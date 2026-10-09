"""
Turns shots + event counts into ONE ROW PER TEAM PER GAME with everything we track.

Two adjustments make the numbers fair:

1. Score & venue adjustment (5v5 shots and xG). Teams that are losing attack more and teams
   that are winning sit back, so raw shot counts flatter trailing teams. For each score state
   (leading by 1, tied, trailing by 2, ...) and venue (home/away) we measure the league-wide
   share of shots, then weight each shot by 0.5 / share. A shot by a team trailing by 2
   (who normally get ~58% of shots) counts 0.5/0.58 = 0.86 of a shot.

2. Rink adjustment (hits, takeaways, giveaways, blocks). These are counted by humans in each
   arena, and some arenas count far more than others. We divide by each arena's factor
   (its average count per game / league average), shrunk toward 1 so small samples don't swing.
"""
import numpy as np
import pandas as pd

from . import config

RINK_STATS = ["hits", "oz_hits", "tk", "oz_tk", "gv", "dz_gv", "blocks"]
RINK_SHRINK_GAMES = 20
HD_XG = 0.10    # a "high-danger" chance is any unblocked shot worth at least 0.10 xG


def _is_5v5(s):
    return ((s["sk_for"] == 5) & (s["sk_against"] == 5) & (s["empty_net"] == 0)
            & (s["own_goalie_pulled"] == 0))


def score_adjust(shots):
    """Weights for 5v5 shots by (venue, score state). Learned from the shots you pass in."""
    s = shots[_is_5v5(shots)].copy()
    s["state"] = s["score_diff"].clip(-3, 3)
    s["xg0"] = np.where(s["kind"] == "block", 0.0, s["xg"])
    out = {}
    for metric, col in (("cf", None), ("xg", "xg0")):
        vals = np.ones(len(s)) if col is None else s[col].to_numpy()
        s["_v"] = vals
        home = s[s["is_home"] == 1].groupby("state")["_v"].sum()
        away = s[s["is_home"] == 0].groupby("state")["_v"].sum()
        w = {"home": {}, "away": {}}
        for st in range(-3, 4):
            h, a = home.get(st, 0.0), away.get(-st, 0.0)   # home leading by st == away trailing by st
            tot = h + a
            w["home"][str(st)] = float(0.5 / (h / tot)) if h > 0 else 1.0
            w["away"][str(-st)] = float(0.5 / (a / tot)) if a > 0 else 1.0
        out[metric] = w
    return out


def rink_factors(games, counts):
    """
    Factor for every (season, arena) using only the TWO PREVIOUS seasons (no peeking).
    Arena = the home team's abbreviation.
    """
    c = counts.merge(games[["game_id", "season", "home"]], on="game_id")
    per_game = c.groupby(["game_id", "season", "home"])[RINK_STATS].sum().reset_index()
    seasons = sorted(per_game["season"].unique())
    targets = seasons + [seasons[-1] + 10001]
    rows = []
    for S in targets:
        src = per_game[per_game["season"].isin([S - 10001, S - 20002])]
        if src.empty:
            src = per_game[per_game["season"] == S]          # very first season: best we can do
        league = src[RINK_STATS].mean()
        by_arena = src.groupby("home")[RINK_STATS].agg(["mean", "count"])
        for arena in sorted(set(per_game["home"])):
            row = {"season": int(S), "arena": arena}
            for st in RINK_STATS:
                if arena in by_arena.index:
                    m, n = by_arena.loc[arena, (st, "mean")], by_arena.loc[arena, (st, "count")]
                else:
                    m, n = league[st], 0
                shrunk = (n * m + RINK_SHRINK_GAMES * league[st]) / (n + RINK_SHRINK_GAMES)
                row[st] = float(shrunk / league[st]) if league[st] > 0 else 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def team_games(games, shots, counts, adjust, rink):
    """One row per team per game. `shots` must already have an 'xg' column."""
    s = shots
    unblocked = (s["kind"] != "block").to_numpy()
    is5 = _is_5v5(s).to_numpy()
    xg = s["xg"].to_numpy(float) * unblocked
    state = s["score_diff"].clip(-3, 3).astype(str).to_numpy()
    venue = np.where(s["is_home"].to_numpy() == 1, "home", "away")
    w_cf = np.array([adjust["cf"][v][st] for v, st in zip(venue, state)])
    w_xg = np.array([adjust["xg"][v][st] for v, st in zip(venue, state)])
    not_en = (s["empty_net"].to_numpy() == 0)
    pp = ((s["sk_for"] > s["sk_against"]) & (s["empty_net"] == 0) & (s["own_goalie_pulled"] == 0)).to_numpy()
    agg = pd.DataFrame({
        "game_id": s["game_id"].to_numpy(), "team": s["team"].to_numpy(),
        "cf5": w_cf * is5,
        "xgf5": w_xg * xg * is5,
        "hdf5": (is5 & unblocked & (xg >= HD_XG)).astype(float),
        "xgf_all": xg * not_en,
        "gf_noen": s["is_goal"].to_numpy() * not_en,
        "pp_xgf": xg * pp,
        "rush_att": s["rush"].to_numpy() * unblocked,
        "reb_att": s["rebound"].to_numpy() * unblocked,
        "fen_att": unblocked.astype(float),
    }).groupby(["game_id", "team"], as_index=False).sum()

    tg = counts.merge(agg, on=["game_id", "team"], how="left").fillna(0.0)
    opp = tg[["game_id", "team", "cf5", "xgf5", "hdf5", "xgf_all", "gf_noen", "pp_xgf"]].rename(columns={
        "team": "opp", "cf5": "ca5", "xgf5": "xga5", "hdf5": "hda5", "xgf_all": "xga_all",
        "gf_noen": "ga_noen", "pp_xgf": "pk_xga"})
    tg = tg.merge(games[["game_id", "season", "date", "home", "away", "neutral", "home_win",
                         "home_score", "away_score", "last_period"]], on="game_id")
    tg["opp"] = np.where(tg["team"] == tg["home"], tg["away"], tg["home"])
    tg = tg.merge(opp, on=["game_id", "opp"])
    tg["is_home"] = (tg["team"] == tg["home"]).astype(int)
    tg["win"] = np.where(tg["is_home"] == 1, tg["home_win"], 1 - tg["home_win"])
    tg["gf_final"] = np.where(tg["is_home"] == 1, tg["home_score"], tg["away_score"])
    tg["ga_final"] = np.where(tg["is_home"] == 1, tg["away_score"], tg["home_score"])

    # rink adjustment for the human-counted stats
    r = rink.rename(columns={st: f"rf_{st}" for st in RINK_STATS})
    tg = tg.merge(r, left_on=["season", "home"], right_on=["season", "arena"], how="left")
    for st in RINK_STATS:
        tg[st] = tg[st] / tg[f"rf_{st}"].fillna(1.0)
    tg = tg.drop(columns=[f"rf_{st}" for st in RINK_STATS] + ["arena"])

    # the "observed margin" the Kalman filter learns from (see ratings.py)
    k = config.KALMAN["obs_goal_weight"]
    tg["margin"] = k * (tg["gf_noen"] - tg["ga_noen"]) + (1 - k) * (tg["xgf_all"] - tg["xga_all"])
    tg["date"] = pd.to_datetime(tg["date"])
    return tg.sort_values(["date", "game_id", "is_home"]).reset_index(drop=True)
