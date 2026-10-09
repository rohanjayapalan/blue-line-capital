"""
Goalie model: talent + streak, where the streak only counts if it's REAL.

GSAx (goals saved above expected) = xG of the unblocked shots he faced - goals he allowed.
    +1.0 means he stopped one more goal than an average goalie would have.

talent  = long-run GSAx per 60 minutes, recent starts weigh more (half-life 30 appearances),
          shrunk toward a slightly-below-average prior so a hot rookie week doesn't fool us.
form    = how far above/below his talent he has played in his last 8 starts (half-life 3).
streak confidence ("gate") = is that form real or random? We run a t-test on the last 8
          starts: t = mean / (std / sqrt(n)).
              |t| < 1  -> looks random    -> gate 0 (ignore the streak)
              |t| > 3  -> clearly real    -> gate 1 (trust it fully)
              between  -> partly trusted
rating  = talent + beta * gate * form     (beta is learned from data; capped at +/- 1 goal/60)
"""
import math

import numpy as np
import pandas as pd

from . import config


def goalie_games(goalies, shots, games):
    """One row per goalie appearance with xGA, GA and GSAx (from OUR xG model)."""
    s = shots[(shots["kind"] != "block") & (shots["empty_net"] == 0) & (shots["goalie"] > 0)]
    faced = s.groupby(["game_id", "goalie"]).agg(xga=("xg", "sum"), ga=("is_goal", "sum")).reset_index()
    faced = faced.rename(columns={"goalie": "goalie_id"})
    gg = goalies.merge(faced, on=["game_id", "goalie_id"], how="left").fillna({"xga": 0.0, "ga": 0})
    gg = gg.merge(games[["game_id", "season", "date"]], on="game_id")
    gg = gg[gg["toi"] > 0].copy()
    gg["date"] = pd.to_datetime(gg["date"])
    gg["gsax"] = gg["xga"] - gg["ga"]
    gg["hours"] = gg["toi"] / 3600.0
    return gg.sort_values(["date", "game_id"]).reset_index(drop=True)


class GoalieBook:
    """Running memory for every goalie, updated one appearance at a time (no peeking ahead)."""

    def __init__(self, params=None):
        self.p = params or config.GOALIE
        self.d = 0.5 ** (1.0 / self.p["talent_half_life"])
        self.g = {}   # goalie_id -> dict

    def _get(self, gid):
        if gid not in self.g:
            self.g[gid] = {"num": 0.0, "den": 0.0, "starts": [], "n_app": 0, "name": "", "team": "",
                           "last_date": None, "recent_start_dates": []}
        return self.g[gid]

    def components(self, gid):
        """talent, form, t-stat, gate for the goalie's NEXT game."""
        p = self.p
        rec = self.g.get(gid)
        k, prior = p["talent_shrink_games"], p["talent_prior"]
        if rec is None:
            return {"talent": prior, "form": 0.0, "t": 0.0, "gate": 0.0, "n": 0}
        talent = (rec["num"] + k * prior) / (rec["den"] + k)
        last = rec["starts"][-p["form_window"]:]
        if len(last) >= 3:
            r = np.array([x - talent for x in last])
            w = 0.5 ** (np.arange(len(r))[::-1] / p["form_half_life"])
            form = float((w * r).sum() / w.sum())
            sd = r.std(ddof=1)
            t = float(r.mean() / (sd / math.sqrt(len(r)))) if sd > 0 else 0.0
        else:
            form, t = 0.0, 0.0
        gate = min(1.0, max(0.0, (abs(t) - p["streak_t_low"]) / (p["streak_t_high"] - p["streak_t_low"])))
        return {"talent": float(talent), "form": form, "t": t, "gate": gate, "n": rec["n_app"]}

    def add(self, gid, gsax, hours, starter, date, name="", team=""):
        rec = self._get(gid)
        rec["num"] = rec["num"] * self.d + gsax
        rec["den"] = rec["den"] * self.d + hours
        rec["n_app"] += 1
        if starter and hours >= 0.5:
            rec["starts"].append(gsax / hours)
            rec["starts"] = rec["starts"][-self.p["form_window"]:]
            rec["recent_start_dates"] = (rec["recent_start_dates"] + [str(pd.Timestamp(date).date())])[-40:]
        rec["name"], rec["team"], rec["last_date"] = name or rec["name"], team or rec["team"], date


def rating(talent, gate, form, beta, cap=None):
    cap = cap if cap is not None else config.GOALIE["cap"]
    return np.clip(np.asarray(talent) + beta * np.asarray(gate) * np.asarray(form), -cap, cap)


def run(gg, params=None):
    """Pregame components for every appearance + the final book (for live predictions)."""
    book = GoalieBook(params)
    out = []
    for r in gg.itertuples(index=False):
        c = book.components(r.goalie_id)
        out.append((r.game_id, r.team, r.goalie_id, r.starter, c["talent"], c["form"], c["t"], c["gate"]))
        book.add(r.goalie_id, r.gsax, r.hours, r.starter, r.date, r.name, r.team)
    comp = pd.DataFrame(out, columns=["game_id", "team", "goalie_id", "starter", "g_talent", "g_form",
                                      "g_t", "g_gate"])
    return comp, book
