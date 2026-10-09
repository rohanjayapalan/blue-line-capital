# Build progress

## Done
- Data pipeline: NHL API + GitHub mirror + Daily Faceoff + NHL EDGE, health checks, GitHub-issue alerts
- 9 seasons parsed (11,052 games, 1.28M shot attempts); live season stored in monthly files
- xG model, score/venue and rink adjustments, Kalman ratings, goalie talent + streak test, Game Score lineups
- Win model (logistic + XGBoost, leave-one-season-out tuning, Platt calibration by season phase)
- Walk-forward backtest 2023-26: 58.0% accuracy, log loss 0.672 (3,936 games)
- Market module (Shin de-vig, risk desk, Kelly, five books) + ESPN historical odds backfill
- Live loop: provisional picks, lock at T-10, hash-chained tape, grading, online learning
- Website (site/): Tonight, Record, Scoreboard, Model; GSAP motion; in-browser tape verification
- GitHub Actions: pregame (every 10 min), nightly, train (manual), tests
- README.md (setup) and LEARN.md (study guide)

## To go live (Rohan)
- [x] Push to a public GitHub repo (2026-10-09)
- [x] Put the GitHub username in site/js/config.js
- [x] First Nightly, Pregame and Train runs triggered from Actions
- [ ] Import to Vercel with Root Directory = site
- [ ] If bot commits fail with a 403: Settings > Actions > General > Workflow permissions, read and write

## Ideas for later
- Custom domain
- A closing-line-value study after a full live season
- Player tracking inputs (zone entries, backchecking) if a free source appears
