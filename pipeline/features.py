"""
The feature table: one row per game, built ONLY from what was known before puck drop.
Almost every number is "home team minus away team", so the model reads it as an edge.

Recency (your rule): a team's last 20 games count 100%. Older games fade: their weight halves
every H games (H is tuned). Last season's games are multiplied by a carry-over factor, and
anything older than last season is never used as a stat.

Shrinkage ("start less confident"): each stat starts with k pretend league-average games:
    value = (sum of weighted stat + k * league average) / (sum of weights + k)
With 3 games played the pretend games dominate; by game 30 the real ones do.
"""
import numpy as np
import pandas as pd

from . import config, goalies, ratings, teams

# name: (numerator column, denominator column) - every stat is a ratio of two sums
STATS = {
    "xgf_pct": ("xgf5", "xg5_tot"),      # share of 5v5 expected goals (score & venue adjusted)
    "cf_pct": ("cf5", "cf5_tot"),        # share of 5v5 shot attempts (Corsi, adjusted)
    "hd_pct": ("hdf5", "hd5_tot"),       # share of 5v5 high-danger chances
    "gf_pct": ("gf_noen", "g_tot"),      # share of actual goals (no empty-netters)
    "pp": ("pp_xgf", "pp_hours"),        # power-play xG per 60
    "pk": ("pk_xga", "pk_hours"),        # penalty-kill xG allowed per 60 (lower = better)
    "pen": ("pen_net", "one"),           # penalties drawn minus taken, per game
    "finishing": ("fin", "one"),         # goals minus xG per game (skill... or luck)
    "battles": ("battles", "one"),       # takeaways minus giveaways per game (puck battles)
    "forecheck": ("forecheck", "one"),   # offensive-zone hits + takeaways per game
    "dz_gv": ("dz_gv", "one"),           # giveaways in your own zone per game (bad)
    "faceoff": ("fow", "fo_tot"),        # faceoff win %
    "rush": ("rush_att", "fen_att"),     # share of attempts that come off the rush
}
STAT_NAMES = list(STATS)
FEATURES = (["rating_z"] + STAT_NAMES
            + ["goalie", "lineup", "star_missing", "rest", "b2b_home", "b2b_away",
               "travel_away", "tz_away", "neutral"] + config.LIVE_ONLY_FEATURES)

FEATURE_INFO = {
    "rating_z": "Kalman power-rating gap, divided by how unsure we are",
    "xgf_pct": "5v5 expected-goals share (score & venue adjusted)",
    "cf_pct": "5v5 shot-attempt share (Corsi, adjusted)",
    "hd_pct": "5v5 high-danger chance share",
    "gf_pct": "Actual goal share (no empty-netters)",
    "pp": "Power-play chances created per 60",
    "pk": "Penalty-kill chances allowed per 60",
    "pen": "Penalties drawn minus taken",
    "finishing": "Goals above expected (finishing / luck)",
    "battles": "Puck battles: takeaways minus giveaways",
    "forecheck": "Forecheck pressure: offensive-zone hits + takeaways",
    "dz_gv": "Own-zone giveaways",
    "faceoff": "Faceoff win %",
    "rush": "Share of chances off the rush",
    "goalie": "Starting goalie: talent + trusted streak (GSAx/60)",
    "lineup": "Tonight's top-6 F + top-2 D vs usual lineup",
    "star_missing": "Core players out of the lineup",
    "rest": "Rest days advantage",
    "b2b_home": "Home team on a back-to-back",
    "b2b_away": "Away team on a back-to-back",
    "travel_away": "Away team travel since last game (1000 km)",
    "tz_away": "Away team time zones crossed",
    "neutral": "Neutral-site game (no home ice)",
    "edge_speed": "NHL EDGE skating bursts over 20 mph (live only)",
    "edge_ozone": "NHL EDGE offensive-zone time (live only)",
}


def prepare(tg):
    tg = tg.copy()
    tg["xg5_tot"] = tg["xgf5"] + tg["xga5"]
    tg["cf5_tot"] = tg["cf5"] + tg["ca5"]
    tg["hd5_tot"] = tg["hdf5"] + tg["hda5"]
    tg["g_tot"] = tg["gf_noen"] + tg["ga_noen"]
    tg["pp_hours"] = tg["pp_toi"] / 3600.0
    tg["pk_hours"] = tg["pk_toi"] / 3600.0
    tg["pen_net"] = tg["pen_drawn"] - tg["pen_taken"]
    tg["fin"] = tg["gf_noen"] - tg["xgf_all"]
    tg["battles"] = tg["tk"] - tg["gv"]
    tg["forecheck"] = tg["oz_hits"] + tg["oz_tk"]
    tg["one"] = 1.0
    tg["fo_tot"] = tg["fow"] + tg["fol"]
    tg["franchise"] = tg["team"].map(teams.current_franchise)
    return tg.sort_values(["date", "game_id", "is_home"]).reset_index(drop=True)


def league_means(tg):
    return {s: [float(tg[n].mean()), float(tg[d].mean())] for s, (n, d) in STATS.items()}


def _weights(prev_seasons, season_now, H, carry):
    k = np.arange(len(prev_seasons))
    F = config.FULL_WEIGHT_GAMES
    w = np.where(k < F, 1.0, 0.5 ** ((k - (F - 1)) / H))
    same = prev_seasons == season_now
    last = prev_seasons == config.prev_season(season_now)
    return w * np.where(same, 1.0, np.where(last, carry, 0.0))


def _arrays(tg):
    num = tg[[STATS[s][0] for s in STAT_NAMES]].to_numpy(float)
    den = tg[[STATS[s][1] for s in STAT_NAMES]].to_numpy(float)
    return num, den, tg["season"].to_numpy()


def rolling(tg, H, carry):
    """Weighted sums over each team's PREVIOUS games (most recent first)."""
    num, den, seas = _arrays(tg)
    S_num, S_den, gp = np.zeros_like(num), np.zeros_like(den), np.zeros(len(tg))
    for _, idx in tg.groupby("franchise", sort=False).indices.items():
        idx = np.sort(idx)
        for j in range(1, len(idx)):
            prev = idx[max(0, j - config.LOOKBACK_GAMES): j][::-1]
            w = _weights(seas[prev], seas[idx[j]], H, carry)
            S_num[idx[j]], S_den[idx[j]] = w @ num[prev], w @ den[prev]
            gp[idx[j]] = np.sum(seas[prev] == seas[idx[j]])
    return S_num, S_den, gp


def next_game_sums(tg, H, carry, season):
    """The same sums for every team's NEXT game in `season` (used for live predictions)."""
    num, den, seas = _arrays(tg)
    out = {}
    for fr, idx in tg.groupby("franchise", sort=False).indices.items():
        prev = np.sort(idx)[-config.LOOKBACK_GAMES:][::-1]
        w = _weights(seas[prev], season, H, carry)
        out[fr] = {"num": (w @ num[prev]).tolist(), "den": (w @ den[prev]).tolist(),
                   "gp": int(np.sum(seas[prev] == season))}
    return out


def team_values(S_num, S_den, league):
    k = np.array([config.SHRINK_GAMES[s] for s in STAT_NAMES], dtype=float)
    Ln = np.array([league[s][0] for s in STAT_NAMES])
    Ld = np.array([league[s][1] for s in STAT_NAMES])
    return (np.asarray(S_num) + k * Ln) / (np.asarray(S_den) + k * Ld)


def schedule_context(tg):
    """Rest days, travel (km) and time zones crossed since each team's previous game."""
    rest, travel, tz = np.full(len(tg), 4.0), np.zeros(len(tg)), np.zeros(len(tg))
    dates = tg["date"].to_numpy("datetime64[D]")
    seas, arena = tg["season"].to_numpy(), tg["home"].to_numpy()
    for _, idx in tg.groupby("franchise", sort=False).indices.items():
        idx = np.sort(idx)
        for j in range(1, len(idx)):
            p, c = idx[j - 1], idx[j]
            if seas[p] != seas[c]:
                continue
            rest[c] = min(int((dates[c] - dates[p]).astype(int)), 4)
            travel[c] = teams.distance_km(arena[p], arena[c])
            tz[c] = abs(teams.tz_offset(arena[p]) - teams.tz_offset(arena[c]))
    return rest, travel, tz


def build_table(games, tg, H, carry, league, kal_pre, gcomp, lineup, sched=None):
    """One row per game with h_* and a_* columns for both teams (all pregame)."""
    S_num, S_den, gp = rolling(tg, H, carry)
    rest, travel, tz = sched if sched is not None else schedule_context(tg)
    base = pd.DataFrame(team_values(S_num, S_den, league), columns=STAT_NAMES)
    base["game_id"], base["team"], base["is_home"] = tg["game_id"].values, tg["team"].values, tg["is_home"].values
    base["gp"], base["rest"], base["travel_km"], base["tz"] = gp, rest, travel, tz
    st = (gcomp[gcomp["starter"] == 1].drop_duplicates(["game_id", "team"])
          [["game_id", "team", "goalie_id", "g_talent", "g_form", "g_gate"]])
    base = base.merge(st, on=["game_id", "team"], how="left").merge(lineup, on=["game_id", "team"], how="left")
    base = base.fillna({"g_talent": config.GOALIE["talent_prior"], "g_form": 0.0, "g_gate": 0.0,
                        "lineup_delta": 0.0, "stars_missing": 0, "lineup_score": 0.0, "goalie_id": 0})
    h = base[base["is_home"] == 1].drop(columns="is_home").add_prefix("h_").rename(columns={"h_game_id": "game_id"})
    a = base[base["is_home"] == 0].drop(columns="is_home").add_prefix("a_").rename(columns={"a_game_id": "game_id"})
    g = (games[["game_id", "season", "date", "home", "away", "neutral", "home_win", "last_period",
                "home_score", "away_score"]]
         .merge(h, on="game_id").merge(a, on="game_id").merge(kal_pre, on="game_id"))
    g["date"] = pd.to_datetime(g["date"])
    g["gp"] = np.minimum(g["h_gp"], g["a_gp"])
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def make_X(g, beta, ktag="k25"):
    """The model's inputs. beta = how much a trusted goalie streak counts; ktag = Kalman variant."""
    X = pd.DataFrame(index=g.index)
    X["rating_z"] = ratings.rating_z(g[f"rating_diff_{ktag}"].to_numpy(float), g[f"rating_var_{ktag}"].to_numpy(float))
    for s in STAT_NAMES:
        X[s] = g[f"h_{s}"] - g[f"a_{s}"]
    X["goalie"] = (goalies.rating(g["h_g_talent"], g["h_g_gate"], g["h_g_form"], beta)
                   - goalies.rating(g["a_g_talent"], g["a_g_gate"], g["a_g_form"], beta))
    X["lineup"] = g["h_lineup_delta"] - g["a_lineup_delta"]
    X["star_missing"] = g["h_stars_missing"] - g["a_stars_missing"]
    X["rest"] = g["h_rest"].clip(0, 3) - g["a_rest"].clip(0, 3)
    X["b2b_home"] = (g["h_rest"] == 1).astype(float)
    X["b2b_away"] = (g["a_rest"] == 1).astype(float)
    X["travel_away"] = g["a_travel_km"] / 1000.0
    X["tz_away"] = g["a_tz"]
    X["neutral"] = g["neutral"].astype(float)
    for f in config.LIVE_ONLY_FEATURES:
        X[f] = g[f] if f in g.columns else 0.0
    return X[FEATURES].astype(float)
