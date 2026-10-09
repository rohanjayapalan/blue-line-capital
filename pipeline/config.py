"""
Every setting for Blue Line Capital lives in this file.
If you want to change how the model behaves, start here.

Units used everywhere:
  - probabilities are 0-1 (0.62 = 62%)
  - "goals" style numbers are per 60 minutes unless the name says otherwise
  - seasons are written the NHL way: 20252026 means 2025-26
"""
from pathlib import Path

PROJECT_NAME = "Blue Line Capital"
MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # tables the model learns from (committed, small)
RAW_DIR = DATA_DIR / "raw"          # raw NHL game files (NOT committed, cached in Actions)
SHOT_DIR = DATA_DIR / "shots"       # one row per shot (NOT committed, rebuilt from raw)
STATE_DIR = ROOT / "state"          # trained models + live state (committed)
TAPE_DIR = ROOT / "tape"            # locked predictions, hash-chained (committed)
PUBLIC_DIR = ROOT / "public"        # the JSON files the website reads (committed)

# ---------------------------------------------------------------- seasons
FIRST_SEASON = 20172018             # oldest season we download
LIVE_SEASON = 20262027              # the season we are predicting live
LIVE_SEASON_START = "2026-09-29"    # opening night
LIVE_SEASON_END = "2027-04-10"      # last day of the regular season

# Regular-season game counts. 2019-20 and 2020-21 were cut short by COVID.
GAMES_IN_SEASON = {
    20172018: 1271, 20182019: 1271, 20192020: 1082, 20202021: 868,
    20212022: 1312, 20222023: 1312, 20232024: 1312, 20242025: 1312,
    20252026: 1312, 20262027: 1344,
}
ALL_HISTORY = [s for s in GAMES_IN_SEASON if s < LIVE_SEASON]

# 2020-21 was a 56-game, division-only, mostly-no-fans season. It is still used
# for stats (it was real hockey) but it does not teach the model decision rules.
SKIP_FOR_TRAINING = {20202021}

# Your rule: "only use seasons more than a year and a half ago to train the model's
# decision making". When predicting season S, the model trains on seasons <= S - 2.
# Season S - 1 is only used as a (faded) stat, never as training data.
TRAIN_GAP = 1

# Seasons we "replay" as an exam (walk-forward backtest). Each one is predicted
# day by day with a model that never saw it.
BACKTEST_SEASONS = [20232024, 20242025, 20252026]

# ---------------------------------------------------------------- recency (your rule)
FULL_WEIGHT_GAMES = 20        # the last 20 games all count 100%...
FADE_HALF_LIFE_OPTIONS = [6, 10, 16]  # ...then weight halves every H games (H is tuned)
CARRYOVER_OPTIONS = [0.3, 0.6]        # last season's games are multiplied by this
LOOKBACK_GAMES = 120          # never look further back than this many games
DEFAULT_HALF_LIFE = 10
DEFAULT_CARRYOVER = 0.3

# Bayesian shrinkage: every stat starts with this many "league average" pretend games.
# Noisy stats (shooting luck) get a big number, stable stats (shot share) a small one.
# This is how the model "starts deliberately less confident and works its way up".
SHRINK_GAMES = {
    "xgf_pct": 8, "cf_pct": 6, "hd_pct": 10, "gf_pct": 20,
    "pp": 15, "pk": 15, "pen": 12, "finishing": 40,
    "battles": 10, "forecheck": 10, "dz_gv": 12, "faceoff": 8, "rush": 15,
}

# ---------------------------------------------------------------- team ratings (Kalman filter)
KALMAN = {
    "obs_goal_weight": 0.4,      # observed margin = 0.4 * goal margin + 0.6 * xG margin
    "obs_noise_sd": 1.55,        # how noisy one game's margin is (in goals); tuned in bootstrap
    "daily_drift_sd": 0.012,     # how much true strength can change per day
    "start_sd": 0.35,            # uncertainty about a team on day 1 of our data
    "offseason_regress": 0.35,   # every summer, pull each team 35% back toward average
    "offseason_extra_sd": 0.15,  # ...and admit we know less about them
    "expansion_mean": -0.10,     # new teams start slightly below average
    "home_ice": 0.13,            # goals of home advantage, re-estimated nightly
    "z_scale": 2.4,              # rating_z = diff / sqrt(z_scale^2 + uncertainty)
}

# The Kalman filter's observed margin = w * goals + (1 - w) * xG. w is tuned from these:
KALMAN_GOAL_WEIGHT_OPTIONS = [0.25, 0.5]

# ---------------------------------------------------------------- goalies
GOALIE = {
    "talent_half_life": 30,      # starts; talent changes slowly
    "talent_prior": -0.05,       # unknown goalies start slightly below average (GSAx/60)
    "talent_shrink_games": 15,   # pretend games of prior
    "form_window": 8,            # a "streak" is judged over the last 8 starts
    "form_half_life": 3,
    "streak_t_low": 1.0,         # t-stat below this = random noise -> streak ignored
    "streak_t_high": 3.0,        # t-stat above this = real streak -> counted fully
    "form_weight_options": [0.0, 0.25, 0.5, 0.75, 1.0],  # beta, tuned in bootstrap
    "cap": 1.0,                  # a goalie can never be worth more than 1 goal/60 either way
    "starter_prob": {"Confirmed": 0.97, "Likely": 0.85, "Unconfirmed": 0.60},
}

# ---------------------------------------------------------------- players
PLAYER = {
    "half_life_games": 40,
    "shrink_minutes": 300,       # pretend minutes of replacement-level play
    "top_forwards": 6,
    "top_defense": 2,
    "stars_per_team": 3,         # "core" players whose absence triggers star_missing
}

# ---------------------------------------------------------------- win model
WIN_MODEL = {
    "C_options": [0.03, 0.1, 0.3],          # logistic regression strength (smaller = simpler)
    "blend_options": [0.5, 0.7, 0.85, 1.0], # weight on logistic regression vs XGBoost
    "xgb_params": {
        "n_estimators": 300, "max_depth": 2, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
        "reg_lambda": 5.0,
    },
    "phases": [(0, 10, "early"), (10, 40, "mid"), (40, 999, "late")],  # games played
    "online_lr": 0.004,           # how fast the model learns from each night's results
    "online_anchor": 0.05,       # pull back toward the offline weights (stops drifting)
    "min_games_for_prediction": 0,
}

# Features that only exist live (NHL EDGE tracking has no clean history).
# They start at weight 0 and must EARN weight through online learning.
LIVE_ONLY_FEATURES = ["edge_speed", "edge_ozone"]

# ---------------------------------------------------------------- market + betting
BOOKS = {7: "FanDuel", 9: "DraftKings", 6: "Veikkaus", 2: "Unibet", 3: "Tipsport", 8: "Sportradar"}
PREFERRED_BOOKS = [7, 9]          # North American books we "bet" at
QUOTE_MARGIN = (0.0, 0.15)        # a real pre-game two-way line carries a 0-15% margin; anything else is stale or in-game
BET = {
    "start_bankroll": 10_000.0,
    "alpha_normal": 0.5,          # how far we move from the market price toward our model
    "alpha_steal": 0.8,
    "alpha_failed_check": 0.2,
    "steal_gap": 0.07,            # 7 percentage points = "major disagreement"
    "min_stake": 0.0025,          # always-in book: never less than 0.25% of bankroll
    "kelly_fraction": 0.25,
    "steal_kelly_fraction": 0.5,
    "cap": 0.03,
    "steal_cap": 0.05,
    "edge_only_min_ev": 0.01,     # edge-only book skips bets under +1% expected value
    "flat_stake": 100.0,          # benchmark books bet $100 flat
    "trust_start": 0.5,           # stakes start at half size...
    "trust_full_after": 200,      # ...and reach full size after 200 graded games (if earned)
}
RISK_DESK = {
    "line_move_against": 0.025,   # market moved 2.5pp against us since open -> fail
    "goalie_share": 0.40,         # goalie explains >40% of the gap AND not confirmed -> fail
    "single_feature_share": 0.60, # one feature explains >60% of the gap -> fail
    "min_games": 5,
    "stale_hours": 36,
}

# ---------------------------------------------------------------- locking
LOCK_LEAD_MINUTES = 10            # lock when puck drop is 10 minutes away (or already happened)
TIMEZONE = "America/Toronto"      # the site's "today" is Eastern time
DAY_ROLLOVER_HOUR = 5             # before 5 am Eastern, "today" still means last night

# ---------------------------------------------------------------- data sources
NHL_API = "https://api-web.nhle.com/v1"
MIRROR = "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-raw/main/nhl/json/raw"
DFO_GOALIES = "https://www.dailyfaceoff.com/starting-goalies"
DFO_LINES = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
ESPN_ODDS = ("https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events/"
             "{event}/competitions/{event}/odds")
ODDS_API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
DOWNLOAD_THREADS = 8


def prev_season(season):
    """20252026 -> 20242025"""
    return season - 10001


def season_label(season):
    """20252026 -> '2025-26'"""
    s = str(season)
    return f"{s[:4]}-{s[6:]}"
