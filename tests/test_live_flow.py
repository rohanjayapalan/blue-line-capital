"""
End-to-end test of the live loop with a FAKE opening night (test-only data, written to a temp
folder - the real state/, tape/ and public/ are never touched):
  pregame (provisional) -> lock 10 min before puck drop -> lock a started game -> tape verifies
  -> grade + settle bets + online learning -> publish JSON.
Run:  python -m tests.test_live_flow
"""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import config, live, nhl_api, run, tape, teams
from pipeline.health import SourceDownError

TMP = Path(os.environ.get("BLC_TEST_DIR") or tempfile.mkdtemp())   # set BLC_TEST_DIR to keep the output
for d in ("state", "tape", "public"):
    (TMP / d).mkdir()
for f in config.STATE_DIR.glob("*.json"):
    shutil.copy(f, TMP / "state" / f.name)
config.STATE_DIR, config.TAPE_DIR, config.PUBLIC_DIR = TMP / "state", TMP / "tape", TMP / "public"

GAMES = [("FLA", "CAR", "21:00", "+115", "-135"), ("MTL", "TOR", "23:00", "+140", "-165"),
         ("NYR", "BOS", "00:00", "-105", "-115"), ("VAN", "EDM", "02:00", "+150", "-180"),
         ("CHI", "VGK", "02:30", "+230", "-285")]


def fake_games(state="FUT"):
    out = []
    for i, (a, h, t, ao, ho) in enumerate(GAMES, start=1):
        day = "2026-09-29" if t >= "20:00" else "2026-09-30"
        out.append({"id": 2026020000 + i, "season": 20262027, "gameType": 2, "gameState": state,
                    "gameScheduleState": "OK", "startTimeUTC": f"{day}T{t}:00Z", "neutralSite": False,
                    "venue": {"default": "Arena"},
                    "homeTeam": {"abbrev": h, "id": teams.NHL_IDS[h], "odds": [{"providerId": 7, "value": ho}, {"providerId": 9, "value": ho}]},
                    "awayTeam": {"abbrev": a, "id": teams.NHL_IDS[a], "odds": [{"providerId": 7, "value": ao}, {"providerId": 9, "value": ao}]}})
    return out


SCHED = {"games": fake_games()}
nhl_api.games_on = lambda d: SCHED["games"]
nhl_api.dfo_starting_goalies = lambda d=None: [
    {"homeTeamSlug": teams.DFO_SLUGS["CAR"], "awayTeamSlug": teams.DFO_SLUGS["FLA"],
     "homeGoalieName": "Frederik Andersen", "awayGoalieName": "Sergei Bobrovsky",
     "homeNewsStrengthName": "Confirmed", "awayNewsStrengthName": "Likely"}]


def no_lines(slug):
    raise SourceDownError("Daily Faceoff lines", "offline in test")


nhl_api.dfo_lines = no_lines


class Args:
    date = "2026-09-29"


def at(ts):
    live.now_utc = lambda: datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


# 1) morning run: provisional predictions, nothing locked
at("2026-09-29T15:00:00")
run.cmd_pregame(Args)
prov = json.loads((TMP / "state/provisional.json").read_text())
assert len(prov) == 5, len(prov)
assert tape.verify(config.LIVE_SEASON)["entries"] == 0

# 2) 8 minutes before FLA @ CAR: that game (only) locks
at("2026-09-29T20:52:00")
run.cmd_pregame(Args)
assert list(tape.locked(config.LIVE_SEASON)) == [2026020001]

# 3) MTL @ TOR already started when the job runs: locks with the prediction made BEFORE puck drop
SCHED["games"] = fake_games()
SCHED["games"][1]["gameState"] = "LIVE"
SCHED["games"][0]["gameState"] = "LIVE"
at("2026-09-29T23:04:00")
run.cmd_pregame(Args)
locked = tape.locked(config.LIVE_SEASON)
assert sorted(locked) == [2026020001, 2026020002], sorted(locked)
body2 = locked[2026020002][0]
assert body2["computed_at"] < body2["start_utc"].replace("Z", "+00:00"), body2["computed_at"]
v = tape.verify(config.LIVE_SEASON)
assert v["ok"] and v["entries"] == 2

# 4) tampering is detected
p = tape.path(config.LIVE_SEASON)
orig = p.read_text()
first = json.loads(orig.splitlines()[0])
first["body"] = first["body"].replace('"p_home":', '"p_home":0.99,"x_":', 1)   # quietly edit an old pick
p.write_text(json.dumps(first, separators=(",", ":")) + "\n" + "\n".join(orig.splitlines()[1:]) + "\n")
assert not tape.verify(config.LIVE_SEASON)["ok"]
p.write_text(orig)

# 5) grade: CAR won 4-2 in regulation, TOR lost 2-3 in OT
world = {"games": pd.DataFrame([
    {"game_id": 2026020001, "home_win": 1, "home_score": 4, "away_score": 2, "last_period": "REG"},
    {"game_id": 2026020002, "home_win": 0, "home_score": 2, "away_score": 3, "last_period": "OT"}])}
record, model = run.grade_and_learn(world, set(), "2026-09-30")
assert len(record["graded"]) == 2 and model["online_updates"] == 2
run.publish.run(today="2026-09-29", games=SCHED["games"], log=lambda *a: None)
today = json.loads((TMP / "public/today.json").read_text())
rec = json.loads((TMP / "public/record.json").read_text())

g1 = today["games"][0]["prediction"]
print("PASS: provisional -> lock at T-10 -> lock started game -> tamper check -> grade -> learn -> publish")
print("FLA@CAR pick:", g1["pick"], g1["p_pick"], g1["tier"], "| market:", g1["market"]["p_home"], g1["market"]["home_price"],
      "| edge pp:", g1["edge_pp"], "| bet:", g1["bet"])
print("goalies:", g1["goalies"]["home"]["name"], g1["goalies"]["home"]["status"], "vs", g1["goalies"]["away"]["name"], g1["goalies"]["away"]["status"])
print("why:", [(w["feature"], w["pp_home"], w["detail"]) for w in g1["why"][:4]])
print("projected:", g1["projected"], "| desk:", g1["desk"]["triggered"])
print("record:", rec["season"], "| always-in book:", {k: rec["books"]["always"][k] for k in ("bankroll", "bets", "roi")})
print("learning log:", rec["learning_log"][0]["moves"][0])
print("statuses:", [c["status"] for c in today["games"]], "| tape:", today["tape"]["ok"], today["tape"]["entries"])
