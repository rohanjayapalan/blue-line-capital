# Blue Line Capital

Predicts the winner of every NHL regular-season game, locks each pick 10 minutes before puck drop in a
SHA-256 hash chain, and grades itself against the betting market with five simulated bankrolls.
GitHub Actions runs the pipeline; Vercel serves the static site from `site/`. Audience: recruiters for
capital markets, IB, PE/quant and analyst roles, so rigour and honesty matter more than flash.

Read `README.md` for setup, `LEARN.md` for how every component works, `PROGRESS.md` for status.

## How to work in this repo

- Explain changes the way LEARN.md does: plain language, an analogy for any new concept, short
  sections instead of walls of text. Keep it interesting.
- When a component changes, update its section in LEARN.md (and README's results table if numbers move).
- Site copy is direct and understated, in sentence case. No hype.
- **No demo or fake data, ever,** in `public/`, `state/` or the site. Tests write to a temp folder
  (`BLC_TEST_DIR=/some/dir` keeps their output for inspection).
- Never edit `tape/*.jsonl` by hand: every line hashes the previous one, and the site re-verifies the chain.
- The bot commits to `public/ state/ tape/ data/` every 10 minutes on game days. Run `git pull` before
  editing, and don't hand-edit those folders.
- Keep `tape/.gitkeep`: `scripts/commit.sh` only adds folders that exist.
- Run `python -m tests.test_live_flow` before pushing anything in `pipeline/` or `tests/`; CI runs the
  same test, and a broken pipeline means missed locks that can't be redone.
- Bot commits are authored by `blue-line-bot`, so `git log --author=blue-line-bot` lists only theirs.

## Commands

```bash
pip install -r requirements.txt
python -m tests.test_live_flow                   # end-to-end: provisional -> lock -> tamper check -> grade -> learn
python -m pipeline.run pregame [--date YYYY-MM-DD]   # predict, lock at T-10, publish
python -m pipeline.run nightly                   # ingest last night, grade, settle, learn, snapshot
python -m pipeline.run train [--odds] [--skip-backtest]
python -m pipeline.run backtest                  # three-season walk-forward exam
python -m pipeline.run doctor                    # check every data source
python -m http.server 8000                       # then open http://localhost:8000/site/ (reads ../public)
```

## Layout

- `pipeline/`: `config` (every setting), `teams`, `health` (format checks, GitHub issues), `nhl_api`,
  `parse_game`, `store` (parquet tables), `xg_model`, `team_stats` (score/venue + rink adjustments),
  `ratings` (Kalman), `goalies`, `players` (Game Score lineups), `features` (25 inputs), `win_model`,
  `backtest` (+ `market_section`), `market` (Shin, risk desk, Kelly, books), `odds_history` (ESPN),
  `tape`, `live`, `run` (CLI), `publish` (writes `public/*.json`).
- `site/`: `index.html`, `css/style.css`, `css/fonts.css`, `js/config.js` (GitHub owner/repo),
  `js/app.js` (all views, SVG charts, GSAP motion, in-browser tape check), `vendor/gsap.min.js`,
  self-hosted fonts, `vercel.json`.
- `data/tables/<season>/<table>.parquet` for history; the live season uses monthly files
  `<table>-YYYY-MM.parquet` so nightly commits stay small. `data/raw/` is gitignored.
- `state/`: trained models, snapshot, record, provisional picks. `public/`: what the site reads.
- `.github/workflows/`: `pregame` (every 10 min, 6 am to 2 am ET), `nightly` (5:30 am ET + backup),
  `train` (manual), `tests` (on code changes). All data jobs share the `blc-data` concurrency group.
  Cron is in UTC and doesn't follow daylight saving, so the window shifts an hour in winter.
  GitHub can delay scheduled runs by 10+ minutes when busy; the late-run lock rule below covers it.

## Rules the owner set (keep them)

- Regular season only. Overtime and shootout wins count. The winner is shown front and centre.
- A pick locks 10 minutes before puck drop. A late run locks the last pick made *before* the start,
  never one made after.
- Hits get full rows with a goal light; misses stay visible but smaller and muted. Both always count.
- Recent form: the last 20 games at full weight, then decay (half-life tuned, currently 6 games).
  Last season counts at a tuned weight (0.3). Anything older is used only to train the model's
  decision-making, never as a team stat.
- Team-level model plus top-6 forwards, top defence pair and the starting goalie.
- Goalies matter but not excessively. A streak counts only if a t-test says it's consistent (the
  streak weight was tuned and the data chose 0).
- Start the season less confident: phase-specific Platt calibration (early slope about 0.8).
- Bankroll: the always-in book bets every game and sizes up with confidence (fractional Kelly); the
  edge-only book bets with an edge. Benchmarks: favourite, home team, model pick at a flat $100.
- A model/market gap of 7+ points triggers the five-check risk desk; passing all five makes it a steal.
- Cost stays $0 to $10 a month. Public repo, Vercel Hobby, free data sources.
- Design: ice #EDF1F3, ink #102A43/#0D1F33, red #D62839 = model, blue #1B4F9C = market,
  crease #9CC3E6, yellow #F2B705 only for a lock or a steal. Big Shoulders Display + Barlow, self-hosted.
  No "AI look": no gradients or glass, no all-caps labels, no dot-separated meta lines.
  GSAP motion (skate-trail load, puck slides, Zamboni wipe between views); respect prefers-reduced-motion.

## Data sources and quirks

- NHL API `https://api-web.nhle.com/v1`: schedule with sportsbook odds, play-by-play, boxscores,
  rosters, EDGE. `teams.NHL_IDS` includes UTA=68 (Utah) and ARI=53 (history).
- History mirror: `raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main/nhl/json/raw/{id}.json`.
- Daily Faceoff: parsed from `__NEXT_DATA__` (`homeTeamSlug`, `homeGoalieName`, `homeNewsStrengthName`).
  A format change raises `DataFormatError` and opens a GitHub issue; predictions fall back to
  usage-based goalies and last-game lineups.
- ESPN odds history (`train --odds`) runs from Actions; it's optional, failures are logged, never fatal.
  Its "... - Live Odds" feeds hold in-game prices that leak the result, and some 2023-24 books are
  stale (negative margin). Both are filtered: `market.sane` keeps only lines with a 0 to 15% margin,
  live and in the backtest. Never relax that without re-checking the market accuracy (~59%, not 90%).
- The site fetches `public/*.json` and `tape/<season>.jsonl` from `raw.githubusercontent.com` (owner set
  in `site/js/config.js`), so data commits never redeploy Vercel. Vercel's root directory is `site`.
- `publish` skips writes when only `generated_at` changed, and health refreshes `last_ok` at most every
  6 hours, to keep the commit history quiet.

## Results (walk-forward backtest, 3,936 games)

58.0% picks right, log loss 0.672, Brier 0.240. By season: 2023-24 60.6% / 0.664,
2024-25 57.6% / 0.668, 2025-26 55.6% / 0.685 (a parity year: home teams won 52.2%).
Closing line on the same games: 59.3% / 0.665, so the model trails the market slightly; every
simulated book loses roughly the vig over the backtest. Live model CV: log loss 0.661, 59.9%.
Chosen settings: half-life 6, carryover 0.3, Kalman 50% goals / 50% xG, goalie streak weight 0,
C = 0.03, blend 100% logistic (XGBoost didn't beat it out of sample). xG holdout AUC about 0.75.

## Status

**Pushed to `main` on 2026-10-09**, with `site/js/config.js` pointing at `rohanjayapalan`. The
end-to-end test passed on Python 3.12 before the push (the pinned numpy 1.26.4 won't build on 3.13,
so use 3.12 locally, as CI does). The first Nightly run catches up every game since
`LIVE_SEASON_START`, so the games missed before go-live are added to the ratings, but they were never
locked or graded.

First live runs on 2026-10-09: Nightly caught up the season, Pregame published tonight's picks, Train
loaded ESPN odds (the first market comparison leaked results; fixed in the follow-up, see quirks).
The test now skips the bot's live files (`provisional.json`, `record.json`, line caches) when it
copies `state/`.

Still needs the owner (these can't be done from a repo push):

1. Import to Vercel with root directory `site`, framework preset "Other", no build command.
2. Watch the repo (All activity) so data-source issues reach your email.
3. Only if bot commits fail with a 403: Settings > Actions > General > Workflow permissions, read
   and write. The workflows already ask for `contents: write`, which is normally enough.

### Is it running?

```bash
git pull
git log --author=blue-line-bot --since="2 days ago" --oneline | head   # pregame/nightly commits
python -c "import json; d=json.load(open('public/today.json')); print(d['date'], len(d['games']), d['tape'])"
wc -l tape/*.jsonl                                                      # locked picks so far
```

Healthy on a game day: bot commits within the last hour or so, `today.json` dated today (Eastern),
and the tape growing. No commits on a no-game day is normal, apart from a `health.json` refresh
every 6 hours or so. Also check the Actions tab for red runs and open issues for data-source alerts.

## Ideas not built yet

- Closing-line-value study after a full live season.
- A staleness alert: open an issue if no pregame commit lands for N hours on a game day.
- Extra sportsbooks: `live.predict_game` already accepts `ctx["extra_quotes"]`.
- A headless smoke test for the site (jsdom) in CI.
- Tracking-based inputs (zone entries, backchecking) if a free source appears.
- Custom domain.
