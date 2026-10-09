"""
Team power ratings with a Kalman filter.

Idea: every team has a hidden "true strength" (in goals per game vs an average team).
We never see it directly; we see noisy game results. A Kalman filter keeps, for every team,
  - a best guess of its strength (theta), and
  - how UNSURE we are about that guess (the covariance matrix P).
After each game it nudges the two teams' guesses toward what happened. The nudge is big when
we're unsure (early season, after a trade deadline) and small when we're confident.

What it observes per game (home team's point of view):
    margin = 0.4 * (goal difference, no empty-netters) + 0.6 * (xG difference)
Prediction before the game:  theta_home - theta_away + home_ice
Update: theta += K * (margin - prediction), where K (the "Kalman gain") = P h / (h'P h + R).

Because P is a full matrix, the filter also learns that if Team A beat Team B, and B was
recently beaten by C, those results are linked.
"""
import json

import numpy as np
import pandas as pd

from . import config, teams

FRANCHISES = sorted({teams.current_franchise(t) for t in teams.TEAMS})
IDX = {t: i for i, t in enumerate(FRANCHISES)}


def new_state(params=None):
    p = params or config.KALMAN
    n = len(FRANCHISES)
    return {"theta": np.zeros(n), "P": np.eye(n) * p["start_sd"] ** 2, "season": None, "date": None,
            "active": set()}


def advance(state, date, season, params=None):
    """Move the filter forward in time: daily drift, or a summer regression at a new season."""
    p = params or config.KALMAN
    n = len(FRANCHISES)
    if state["season"] is None:
        state["season"] = season
    elif season != state["season"]:
        r = p["offseason_regress"]
        state["theta"] *= (1 - r)
        state["P"] = (1 - r) ** 2 * state["P"] + np.eye(n) * p["offseason_extra_sd"] ** 2
        state["season"] = season
        state["active"] = set()
        for team, first in teams.EXPANSION.items():      # a brand-new team starts fresh
            if first == season:
                i = IDX[team]
                state["theta"][i] = p["expansion_mean"]
                state["P"][i, :] = 0.0
                state["P"][:, i] = 0.0
                state["P"][i, i] = p["start_sd"] ** 2
    elif state["date"] is not None:
        days = max((date - state["date"]).days, 0)
        state["P"] = state["P"] + np.eye(n) * (p["daily_drift_sd"] ** 2) * days
    state["date"] = date
    return state


def pregame(state, home, away):
    h, a = IDX[teams.current_franchise(home)], IDX[teams.current_franchise(away)]
    th, P = state["theta"], state["P"]
    diff = th[h] - th[a]
    var = P[h, h] + P[a, a] - 2 * P[h, a]
    return float(diff), float(var)


def update(state, home, away, margin, neutral, params=None):
    p = params or config.KALMAN
    h, a = IDX[teams.current_franchise(home)], IDX[teams.current_franchise(away)]
    hv = np.zeros(len(FRANCHISES))
    hv[h], hv[a] = 1.0, -1.0
    P = state["P"]
    Ph = P @ hv
    S = hv @ Ph + p["obs_noise_sd"] ** 2
    K = Ph / S
    pred = state["theta"][h] - state["theta"][a] + (0.0 if neutral else p["home_ice"])
    state["theta"] = state["theta"] + K * (margin - pred)
    state["P"] = P - np.outer(K, Ph)
    state["active"].update({h, a})
    # keep the league average at exactly zero
    act = sorted(state["active"])
    state["theta"][act] -= state["theta"][act].mean()
    return state


def run(games, params=None):
    """
    games: one row per game with game_id, season, date, home, away, neutral, margin (home view).
    Returns (pregame table, final state). The pregame numbers use ONLY earlier games.
    """
    state = new_state(params)
    out = []
    g = games.sort_values(["date", "game_id"])
    for row in g.itertuples(index=False):
        state = advance(state, row.date, row.season, params)
        diff, var = pregame(state, row.home, row.away)
        out.append((row.game_id, diff, var))
        state = update(state, row.home, row.away, row.margin, row.neutral, params)
    return pd.DataFrame(out, columns=["game_id", "rating_diff", "rating_var"]), state


def rating_z(diff, var, params=None):
    p = params or config.KALMAN
    return diff / np.sqrt(p["z_scale"] ** 2 + var)


def to_json(state):
    return {"theta": state["theta"].round(5).tolist(), "P": state["P"].round(6).tolist(),
            "season": state["season"], "date": str(pd.Timestamp(state["date"]).date()),
            "active": sorted(int(i) for i in state["active"]), "teams": FRANCHISES}


def from_json(d):
    return {"theta": np.array(d["theta"]), "P": np.array(d["P"]), "season": d["season"],
            "date": pd.Timestamp(d["date"]), "active": set(d["active"])}


def power_table(state):
    sd = np.sqrt(np.diag(state["P"]))
    rows = [{"team": t, "rating": round(float(state["theta"][i]), 3), "sd": round(float(sd[i]), 3)}
            for t, i in IDX.items() if i in state["active"] or state["season"] is None]
    return sorted(rows, key=lambda r: -r["rating"])
