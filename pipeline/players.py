"""
Player model: how good is tonight's lineup compared to the team's usual lineup?

Game Score (per skater per game, Dom Luszczyszyn's formula, simplified):
  0.75 G + 0.7 A1 + 0.55 A2 + 0.075 SOG + 0.05 BLK + 0.15 PenDrawn - 0.15 PenTaken
  + 0.01 FOW - 0.01 FOL + 0.15 (+/-)
Player rating = Game Score per 60 minutes, recent games count more (half-life 40 games), shrunk
toward a replacement-level prior (so 3 good games from a call-up don't make him a star).

For each team-game:
  lineup score = sum of ratings of the top-6 forwards + top-2 defense (ranked by usual ice time)
  lineup delta = tonight's lineup score - the team's recent average lineup score
                 (catches injuries, rest days, call-ups)
  stars missing = how many of the team's 3 core players are NOT dressed
"""
from collections import deque

import numpy as np
import pandas as pd

from . import config


class PlayerBook:
    def __init__(self, prior_f, prior_d, params=None):
        self.p = params or config.PLAYER
        self.d = 0.5 ** (1.0 / self.p["half_life_games"])
        self.prior = {"F": prior_f, "D": prior_d}
        self.pl = {}                        # player_id -> [gs_sum, hours_sum, toi_sum, n_sum, pos, name, team]
        self.team_recent = {}               # team -> deque of sets of dressed player ids (last 10)
        self.team_lineups = {}              # team -> deque of recent lineup scores

    def rating(self, pid, pos="F"):
        s = self.p["shrink_minutes"] / 60.0
        rec = self.pl.get(pid)
        prior = self.prior.get(pos, self.prior["F"])
        if rec is None:
            return prior
        return (rec[0] + s * prior) / (rec[1] + s)

    def usual_toi(self, pid, pos="F"):
        rec = self.pl.get(pid)
        if rec is None or rec[3] == 0:
            return 720.0 if pos == "F" else 960.0
        return rec[2] / rec[3]

    def lineup_score(self, dressed):
        """dressed: list of (player_id, pos). Top-6 F + top-2 D by usual ice time."""
        f = sorted([pid for pid, pos in dressed if pos == "F"], key=lambda x: -self.usual_toi(x, "F"))
        d = sorted([pid for pid, pos in dressed if pos == "D"], key=lambda x: -self.usual_toi(x, "D"))
        top = [(x, "F") for x in f[: self.p["top_forwards"]]] + [(x, "D") for x in d[: self.p["top_defense"]]]
        return float(sum(self.rating(pid, pos) for pid, pos in top))

    def core(self, team):
        """The team's 3 most important regulars (rating x ice time), from the last 10 games."""
        recent = self.team_recent.get(team)
        if not recent:
            return []
        counts = {}
        for s in recent:
            for pid in s:
                counts[pid] = counts.get(pid, 0) + 1
        need = max(1, int(0.6 * len(recent)))
        regulars = [pid for pid, c in counts.items() if c >= need and pid in self.pl]
        score = lambda pid: self.rating(pid, self.pl[pid][4]) * self.usual_toi(pid, self.pl[pid][4])
        return sorted(regulars, key=score, reverse=True)[: self.p["stars_per_team"]]

    def usual_lineup(self, team):
        q = self.team_lineups.get(team)
        if not q:
            return None
        w = 0.5 ** (np.arange(len(q))[::-1] / 5.0)
        return float(np.dot(w, list(q)) / w.sum())

    def pregame(self, team, dressed):
        score = self.lineup_score(dressed)
        usual = self.usual_lineup(team)
        ids = {pid for pid, _ in dressed}
        missing = [pid for pid in self.core(team) if pid not in ids]
        return {"lineup_score": score, "lineup_delta": 0.0 if usual is None else score - usual,
                "stars_missing": len(missing), "missing_ids": missing}

    def add_game(self, team, rows):
        """rows: list of (player_id, pos, name, toi_seconds, game_score) for one team-game."""
        dressed = [(r[0], r[1]) for r in rows]
        self.team_lineups.setdefault(team, deque(maxlen=10)).append(self.lineup_score(dressed))
        self.team_recent.setdefault(team, deque(maxlen=10)).append({r[0] for r in rows})
        for pid, pos, name, toi, gs in rows:
            rec = self.pl.get(pid)
            if rec is None:
                rec = [0.0, 0.0, 0.0, 0.0, pos, name, team]
                self.pl[pid] = rec
            rec[0] = rec[0] * self.d + gs
            rec[1] = rec[1] * self.d + toi / 3600.0
            rec[2] = rec[2] * self.d + toi
            rec[3] = rec[3] * self.d + 1.0
            rec[4], rec[5], rec[6] = pos, name, team


def priors(skaters):
    """Replacement level = 70% of the league's Game Score per 60, by position."""
    g = skaters.groupby("pos").apply(lambda d: d["gs"].sum() / (d["toi"].sum() / 3600.0), include_groups=False)
    return 0.7 * float(g.get("F", 1.0)), 0.7 * float(g.get("D", 0.6))


def run(skaters, games, prior_f=None, prior_d=None):
    """Pregame lineup numbers for every team-game (chronological, no peeking) + the final book."""
    if prior_f is None:
        prior_f, prior_d = priors(skaters)
    book = PlayerBook(prior_f, prior_d)
    sk = skaters.merge(games[["game_id", "date"]], on="game_id")
    sk = sk.sort_values(["date", "game_id", "team"])
    out = []
    for (gid, team), grp in sk.groupby(["game_id", "team"], sort=False):
        dressed = list(zip(grp["player_id"], grp["pos"]))
        pre = book.pregame(team, dressed)
        out.append((gid, team, pre["lineup_score"], pre["lineup_delta"], pre["stars_missing"]))
        book.add_game(team, list(zip(grp["player_id"], grp["pos"], grp["name"], grp["toi"], grp["gs"])))
    df = pd.DataFrame(out, columns=["game_id", "team", "lineup_score", "lineup_delta", "stars_missing"])
    return df, book
