# LEARN.md: how Blue Line Capital works, piece by piece

This guide walks through every part of the system in the order the data flows, with the math and
the reasoning. Each section ends with the file to open and a line you can say in an interview.

---

## The 30-second version

Every night the pipeline turns each NHL game into 1,300 events, measures shot quality with its own
expected-goals model, and updates a Kalman filter that rates every team with an uncertainty band.
Before each game it combines 25 home-minus-away inputs (team strength, special teams, goalie talent,
missing stars, rest, travel) in a regularised logistic regression blended with XGBoost, calibrated
separately for early, mid and late season. It compares the result to the vig-free betting market,
sizes simulated bets with fractional Kelly behind a five-check risk desk, and locks the pick into a
SHA-256 hash chain 10 minutes before puck drop. A walk-forward backtest on 3,936 games gives
58.0% accuracy and 0.672 log loss, roughly market-level.

---

## 1. Data: from raw game files to tables

**Where it comes from.** The NHL's public API has a play-by-play file (every shot, hit, faceoff,
giveaway, penalty, with x/y coordinates) and a box score (every player's ice time and stats) for
each game. History since 2017-18 (11,052 games) comes from a public GitHub mirror of the same files.

**Coordinates.** The rink is 200 x 85 feet with centre ice at (0, 0) and the nets at x = +89 and
x = -89. Teams switch ends each period, so the parser flips coordinates so the shooting team always
attacks the net at (+89, 0). It decides who attacks which end by *voting*: events tagged as happening
in a team's offensive zone reveal which end that team is attacking. Check: 95.5% of unblocked shots
land in the attacking zone, and goal counts match every final score.

**Strength states.** Each event carries a code like `1551`: away goalie in net, away skaters, home
skaters, home goalie in net. `1541` means the home team is short-handed. This is how the parser knows
5-on-5 versus power play, and how much time each team spent in each state.

**Files:** `pipeline/parse_game.py`, `pipeline/store.py`.
**Say:** "I built the dataset from raw play-by-play: 1.28 million shot attempts, validated against
official final scores."

---

## 2. Expected goals (xG)

Goals are rare and noisy, so a team can outplay its opponent and lose. **Expected goals** estimate
the chance each shot goes in based on *where and how* it was taken, not whether it went in.

The model is a **logistic regression**:

```
xG = 1 / (1 + e^-(b0 + b1*distance + b2*angle + b3*is_rebound + ...))
```

The S-shaped function turns any weighted sum into a probability between 0 and 1. Inputs: distance,
angle, shot type (wrist, slap, tip...), rebound (within 3 seconds of a teammate's shot), rush
(within 4 seconds of an event outside the offensive zone), after a takeaway, after a faceoff, how
fast the puck moved, power play, empty net, score state and overtime. Blocked shots are left out,
because their coordinates mark the block, not the shot.

**How good is it?** On a season it never trained on:
- **AUC about 0.75.** Pick a random goal and a random non-goal; 75% of the time the goal had the
  higher xG. 0.5 is a coin flip.
- **Log loss about 0.23** (defined in section 10).

**File:** `pipeline/xg_model.py`.
**Say:** "Shot quality predicts future results better than goals, so I trained my own xG model and
validated it out of sample."

---

## 3. Making team stats fair

**Score and venue adjustment.** Trailing teams throw more pucks at the net and leading teams sit
back, so raw shot counts flatter bad teams that are always behind. For each score state (up 1, tied,
down 2...) and venue, the league-wide share of shots is measured, and each shot is weighted by

```
weight = 0.5 / league share of shots in that state
```

A team trailing by 2 usually gets about 58% of shots, so each of its shots counts 0.5 / 0.58 = 0.86.

**Rink adjustment.** Hits, takeaways, giveaways and blocks are counted by people in each arena, and
some arenas count far more generously. Each arena gets a factor = (its average per game) / (league
average), built only from the two previous seasons and **shrunk** toward 1:

```
factor = (n * arena_avg + 20 * league_avg) / (n + 20) / league_avg
```

With few games, the 20 "pretend average games" dominate; with many, the real data does.

**File:** `pipeline/team_stats.py`.
**Say:** "Raw event counts are biased by game state and by scorekeepers, so I corrected both before
modelling."

---

## 4. Recent form: weighting and shrinkage

Your rule: the last 20 games matter most, then it fades. A team's stat before game *k* is a
weighted average over its previous games:

```
weight = 1                              for the last 20 games
weight = 0.5 ^ ((age - 19) / H)         after that (halves every H games)
weight x 0.3                            if the game was last season
weight = 0                              if older than last season
```

`H` and the 0.3 were **tuned**, not guessed: the data picked H = 6 from {6, 10, 16} and 0.3 from
{0.3, 0.6}.

**Shrinkage** stops early-season overreaction. Every stat starts with *k* pretend league-average games:

```
value = (sum of weighted stat + k * league average) / (sum of weights + k)
```

This is the **bias-variance trade-off**: a little bias toward average buys a big cut in noise.

**The 25 inputs** (all home minus away): power rating; 5v5 xG, shot-attempt, high-danger and goal
shares; power play and penalty kill chances per 60; penalties drawn minus taken; finishing (goals
minus xG); puck battles (takeaways minus giveaways); forecheck (offensive-zone hits and takeaways);
own-zone giveaways; faceoff %; rush share; starting goalie; lineup versus usual; stars missing; rest;
two back-to-back flags; travel; time zones; neutral site; and two NHL EDGE tracking stats that only
exist live and start at weight zero.

**File:** `pipeline/features.py`.

---

## 5. Power ratings with a Kalman filter

Each team has a hidden true strength θ (goals per game against an average team) that drifts over
time. We never see θ; we see noisy game results. A **Kalman filter** tracks two things for all 32
teams at once:
- **θ**: the best guess of each team's strength;
- **P**: a 32 x 32 matrix of how unsure we are, including how the guesses are linked.

Each game, from the home team's view:

```
observed margin y = 0.5 * (goal difference) + 0.5 * (xG difference)     (the 50/50 split was tuned)
prediction        = θ_home - θ_away + home_ice
surprise          = y - prediction
Kalman gain K     = P h / (h' P h + R)          h = +1 for home, -1 for away, 0 elsewhere
update            θ = θ + K * surprise
                  P = P - K h' P
```

**Intuition:** the gain K is large when we're unsure (P big) and small when we're confident. So an
upset moves ratings a lot in October and little in March. Between games uncertainty grows a little
every day (injuries, trades), and every summer ratings regress 35% toward average with extra
uncertainty. The model input is the gap divided by its uncertainty, a z-score:

```
rating_z = (θ_home - θ_away) / sqrt(scale^2 + variance of the gap)
```

**File:** `pipeline/ratings.py`.
**Say:** "A Kalman filter gives each rating an error bar, so the model automatically trusts early-season
ratings less. It's the same math used for tracking in aerospace and for estimating hidden factors in
finance."

---

## 6. Goalies: talent, and streaks only if they're real

**GSAx** (goals saved above expected) = xG of the unblocked shots a goalie faced minus the goals he
allowed. +1.0 means one more save than an average goalie would have made.

- **Talent** = GSAx per 60 minutes over his career, recent games weighted more, shrunk toward a
  slightly-below-average prior.
- **Form** = how far above or below his own talent he played in his last 8 starts.
- **Is the streak real?** A **t-test** on those 8 starts:

```
t = mean(form) / (standard deviation / sqrt(n))
gate = 0 if |t| < 1,  1 if |t| > 3,  linear in between
goalie rating = talent + beta * gate * form
```

A random hot week has |t| under 1 and gets ignored. A sustained, consistent run clears the bar.

**The finding:** `beta` was tuned over {0, 0.25, 0.5, 0.75, 1} in every season, and the data chose
**0** every time. Once you know a goalie's talent, recent form adds nothing out of sample. The
machinery stays in, so if streaks ever start to matter the tuning will pick it up.

**File:** `pipeline/goalies.py`.
**Say:** "I tested my own hypothesis that goalie streaks predict results. The out-of-sample data
rejected it, so the model uses goalie talent only."

---

## 7. Players: pricing tonight's lineup

**Game Score** rates each skater's game:
`0.75 G + 0.7 A1 + 0.55 A2 + 0.075 shots + 0.05 blocks + 0.15 penalties drawn - 0.15 penalties taken
+ 0.01 faceoff wins - 0.01 faceoff losses + 0.15 (+/-)`.
A player's rating is Game Score per 60 minutes, recent games weighted more and shrunk toward
replacement level, so three good games from a call-up don't make him a star.

- **Lineup delta** = tonight's top-6 forwards + top-2 defence (ranked by usual ice time) minus the
  team's recent average lineup. It catches injuries, rest days and call-ups.
- **Stars missing** = how many of the team's three most important regulars are not dressed.

Tonight's lineup comes from Daily Faceoff's projected lines, matched to NHL player IDs. If that's
unavailable, the model falls back to the team's last lineup.

**File:** `pipeline/players.py`, plus `lineup_side` in `pipeline/live.py`.

---

## 8. The win model

**Logistic regression** adds up the inputs, each times a weight, and squashes the sum into a
probability:

```
log-odds z = w0 + w1*x1 + ... + w25*x25
P(home wins) = 1 / (1 + e^-z)
```

The inputs are standardised (mean 0, standard deviation 1), so each weight means "how much one
standard deviation of this input moves the log-odds". **Regularisation** (C = 0.03, tuned) penalises
large weights, which stops the model from chasing noise in inputs that overlap.

**XGBoost** builds hundreds of tiny decision trees (depth 2), each correcting the previous ones'
mistakes, and can catch interactions a straight line misses. The two are blended in log-odds space:

```
z = w * z_logistic + (1 - w) * z_xgboost          (w tuned)
```

In the current live model the tuning put **100% on logistic regression**. XGBoost didn't beat the
simpler model out of sample, so it's kept but weighted zero. That's the right call, not a failure.

**Tuning without cheating: leave-one-season-out cross-validation.** For each setting combination,
train on all training seasons but one, predict the held-out one, and repeat. The winning settings are
the ones with the best held-out log loss.

**Calibration by season phase (Platt scaling).** A final correction `p = 1 / (1 + e^-(a*z + b))` is
fitted separately for early season (first 10 games), mid and late. The data gave the early phase
a ≈ 0.8, meaning early-season picks get squeezed toward 50%. That's the "start less confident"
behaviour, learned rather than hard-coded.

**Why ratings dominate the weights.** The five team-strength stats all measure "who controls play",
so once the power rating is in, the others add little. Back-to-backs, missing stars and the goalie
carry information the rating doesn't have, so they get meaningful weight despite small correlations.

**File:** `pipeline/win_model.py`.

---

## 9. Explaining one prediction (the "why")

For logistic regression, each input's contribution to the log-odds is exactly `weight x
standardised value`. For XGBoost, SHAP values do the same job. They're blended and scaled like the
prediction itself, then converted to percentage points using the slope of the S-curve at the
midpoint:

```
points ≈ 100 * contribution * p_mid * (1 - p_mid)
```

The site shows these as a waterfall from the league baseline (home ice) to the final call.

---

## 10. How predictions are graded

- **Accuracy**: share of winners picked. Easy to read, but it ignores confidence.
- **Log loss**: `-average of [y*ln(p) + (1-y)*ln(1-p)]`. A confident wrong call is punished hard.
  0.693 is a coin flip; good NHL models and closing lines sit around 0.66 to 0.68.
- **Brier score**: `average of (p - y)^2`, gentler on confident misses.
- **Calibration**: of all games called at 60%, did about 60% win?

**Walk-forward backtest.** For each test season S, the xG model trains on seasons before S and the
win model on seasons up to S-2. Then S is replayed one day at a time with online learning switched on.
Results:

| Season | Picks right | Log loss | Always-home log loss |
|---|---|---|---|
| 2023-24 | 60.6% | 0.664 | 0.690 |
| 2024-25 | 57.6% | 0.668 | 0.687 |
| 2025-26 | 55.6% | 0.685 | 0.693 |

2025-26 was a parity season: home teams won only 52.2%, and every simple model found it hard.

**File:** `pipeline/backtest.py`.
**Say:** "I never report in-sample numbers. Every result is walk-forward, with a two-season gap
between the training data and the test season."

---

## 11. Learning from mistakes (online learning)

After each graded game, the logistic weights take one small step of **stochastic gradient descent**,
pulled back toward their trained values:

```
w = w - rate * ((p - y) * x  +  anchor * (w - w_trained))       rate 0.004, anchor 0.05
```

`(p - y) * x` nudges each weight in the direction that would have reduced this game's error. The
anchor term stops one strange night from wrecking the model. The learning log on the site records
the biggest weight moves and the biggest miss each night.

---

## 12. The market

**Odds to probability.** American odds -150 imply 150/250 = 60%; +130 implies 100/230 = 43.5%. The
two sides add up to about 103.5%, and the extra is the bookmaker's cut (the **vig**).

**Removing the vig: Shin's method.** Instead of dividing proportionally, Shin assumes a share z of
bettors are insiders, so books shade longshots more than favourites. It solves for z so the fair
probabilities add to 1:

```
fair_i = (sqrt(z^2 + 4(1-z) * implied_i^2 / sum_implied) - z) / (2(1-z))
```

**Why the closing line is the benchmark.** The last price before puck drop reflects all public
information and all the sharp money. Beating it on log loss is the hardest test in sports prediction.
"Market followed us" measures how often the line moved toward the model's number after opening,
a classic sign of real information.

**File:** `pipeline/market.py`.

---

## 13. Betting: Kelly, the risk desk and trust

**Anchoring.** Bets use a blend of model and market, not the raw model:

```
logit(p_bet) = logit(p_market) + alpha * (logit(p_model) - logit(p_market))
alpha = 0.5 normally, 0.8 for a checked steal, 0.2 when a check fails
```

**Kelly criterion.** The stake fraction that maximises the long-run growth rate of the bankroll
(the expected log of wealth) is

```
f* = (p * d - 1) / (d - 1)          d = decimal odds
```

Full Kelly is too aggressive when p is uncertain, so the always-in book bets a quarter of it
(half for a steal), times a trust factor, capped at 3% (5% for a steal), with a 0.25% minimum
because it always has a position.

**Risk desk.** When model and market disagree by 7+ points, five checks run before a big bet:
1. The line hasn't moved 2.5+ points against the model since opening.
2. It's not a goalie-driven gap with unconfirmed starters.
3. No single situational factor explains 60%+ of the gap.
4. Both teams have played at least 5 games.
5. The data is less than 36 hours old.

All pass: it's a **steal**. Any fail: the bet stays small.

**Trust** starts at 50% of normal size and ramps to 100% over the first 200 graded games, but only
as fast as the live log loss keeps up with the market's.

**Five books** make the comparison fair: always-in (Kelly), edge-only (bets when expected value is
above 1%), and three flat $100 benchmarks (always the favourite, always the home team, the model's
pick). Each reports ROI, worst drawdown and a **t-stat** on the per-bet returns, which asks whether
profit is distinguishable from luck (above 2 is the usual bar).

**Say:** "I size positions with fractional Kelly behind a pre-trade checklist, and I measure
performance with drawdown and statistical significance, not just profit."

---

## 14. The tape: why nobody can change a pick

Each locked pick is one line: `{prev, body, hash}` where

```
hash = SHA-256(previous hash + this prediction's text)
```

Change one character of an old pick and its hash changes, which breaks the link to every later
line. The site's "Verify the tape" button downloads the file and recomputes every hash in your
browser. Git commit timestamps are a second, independent witness. Picks are computed before puck
drop and never replaced. If a job runs late, it locks the last prediction made *before* the game
started, and never one made after.

**File:** `pipeline/tape.py`, `verifyTape` in `site/js/app.js`.

---

## 15. Engineering

- **GitHub Actions** runs everything on free servers: pregame every 10 minutes (6 am to 2 am ET),
  nightly at 5:30 am ET, training on demand, tests on every code change.
- The site is static HTML, CSS and JavaScript on **Vercel**. It reads the JSON files straight from
  GitHub, so data updates never need a redeploy.
- **Health checks** validate every data source's shape. A format change opens a GitHub issue (which
  emails you), and the site keeps the last good data.
- The end-to-end test (`tests/test_live_flow.py`) replays a fake opening night. It checks
  provisional picks, locking at T-10, locking a game that already started, catching a tampered pick,
  grading, settling and learning.

---

## 16. Honest limitations

- The NHL is the most random of the big leagues. Even perfect information tops out around 60 to
  62% accuracy, and the market already prices most of it.
- Backchecking isn't in public data. Own-zone giveaways and chances allowed are proxies.
- Three backtest seasons is a small sample for betting results. The t-stat is there to say so.
- Online learning is deliberately slow. In testing, faster rates chased noise.

---

## Questions an interviewer might ask

**Why not just use accuracy?** Accuracy ignores confidence. A model that says 51% on everything can
match a sharper model's accuracy while being useless for pricing. Log loss and calibration measure
whether the probabilities are right, and that's what a bet or a trade depends on.

**How do you know you're not overfitting?** Walk-forward testing with a two-season gap, settings
chosen by leave-one-season-out cross-validation, strong regularisation, and shrinkage everywhere. And
when XGBoost didn't beat logistic regression out of sample, the blend dropped it.

**What would you do with more time?** Player tracking data for zone entries and backchecking,
Bayesian hierarchical player ratings, and a proper closing-line-value study once a full live season
of prices is recorded.

**What's the finance connection?** It's the same stack as a quant desk: a signal (the model), a
benchmark (the closing line), position sizing (Kelly), pre-trade risk checks, P&L attribution (the
why waterfall), and an audit trail (the tape).
