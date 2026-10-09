"""
Turns ONE raw NHL game (play-by-play JSON + boxscore JSON) into clean rows.

Output (a dict):
  game     one row: teams, date, final score, how it ended (REG/OT/SO), neutral site
  shots    one row per shot attempt, with everything the xG model needs
  counts   two rows (one per team): hits, takeaways, faceoffs, penalties, time at 5v5/PP/PK
  skaters  one row per skater: boxscore stats + Game Score
  goalies  one row per goalie: starter?, time on ice, shots and goals against

Rink geometry used below (NHL API coordinates, in feet):
  centre ice is (0, 0); the nets are at x = -89 and x = +89; blue lines at x = -25 and +25.
We flip coordinates so the shooting team ALWAYS attacks the net at (+89, 0).
"""
import math
from collections import Counter, defaultdict

NET_X = 89.0
BLUE_LINE_X = 25.0
SHOT_KINDS = {"goal": "goal", "shot-on-goal": "sog", "missed-shot": "miss", "blocked-shot": "block"}
PENALTY_TYPES = {"MIN", "MAJ", "BEN", "MAT"}   # ones that put a team short-handed


def mmss(s):
    """'12:34' -> 754 seconds. Missing -> 0."""
    if not s or ":" not in str(s):
        return 0
    m, sec = str(s).split(":")[:2]
    return int(m) * 60 + int(sec)


def situation(code):
    """
    situationCode '1551' = [away goalie in net][away skaters][home skaters][home goalie in net].
    '0651' means the away team pulled its goalie for a 6th skater.
    """
    try:
        c = str(code)
        if len(c) != 4:
            raise ValueError
        return int(c[0]), int(c[1]), int(c[2]), int(c[3])
    except (TypeError, ValueError):
        return 1, 5, 5, 1


def neutral_site(pbp):
    """Games played in Europe (Global Series) have a UTC offset of about +1/+2."""
    off = str(pbp.get("venueUTCOffset") or "")
    try:
        return int(off.split(":")[0]) >= -2
    except ValueError:
        return False


def attack_signs(plays, team_of_id, home, away):
    """
    Which way is each team shooting in each period? +1 means toward x = +89.
    We VOTE using events whose zone we know: an event in a team's offensive zone ('O')
    tells us which end that team attacks. This is safer than trusting one field.
    """
    votes = defaultdict(int)
    for p in plays:
        d = p.get("details") or {}
        kind = p.get("typeDescKey")
        x, zone = d.get("xCoord"), d.get("zoneCode")
        team = team_of_id.get(d.get("eventOwnerTeamId"))
        if kind not in ("hit", "takeaway", "giveaway", "shot-on-goal", "missed-shot", "goal"):
            continue
        if x is None or team is None or zone not in ("O", "D") or abs(x) <= BLUE_LINE_X:
            continue
        period = (p.get("periodDescriptor") or {}).get("number") or 0
        side = 1 if x > 0 else -1
        votes[(team, period)] += side if zone == "O" else -side
    signs = {}
    periods = {(p.get("periodDescriptor") or {}).get("number", 0) for p in plays}
    for period in periods:
        # fall back to the NHL's own field if a period has no usable votes
        fallback = None
        for p in plays:
            if (p.get("periodDescriptor") or {}).get("number") == period and p.get("homeTeamDefendingSide"):
                fallback = 1 if p["homeTeamDefendingSide"] == "left" else -1
                break
        h = votes.get((home, period), 0) - votes.get((away, period), 0)
        home_sign = (1 if h > 0 else -1) if h != 0 else (fallback or 1)
        signs[(home, period)] = home_sign
        signs[(away, period)] = -home_sign
    return signs


def parse_game(pbp, box):
    gid = int(pbp["id"])
    season = int(pbp["season"])
    home, away = pbp["homeTeam"]["abbrev"], pbp["awayTeam"]["abbrev"]
    team_of_id = {pbp["homeTeam"]["id"]: home, pbp["awayTeam"]["id"]: away}
    roster = {r["playerId"]: team_of_id.get(r.get("teamId")) for r in pbp.get("rosterSpots", [])}
    plays = sorted(pbp.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    signs = attack_signs(plays, team_of_id, home, away)
    other = {home: away, away: home}

    shots = []
    counts = {t: Counter() for t in (home, away)}
    per_player = defaultdict(Counter)      # a1, a2, pen drawn/taken, faceoffs
    score = {home: 0, away: 0}
    prev = None                            # the previous event (for rebounds, rushes...)
    goals_noso = {home: 0, away: 0}

    for p in plays:
        pd_ = p.get("periodDescriptor") or {}
        period, ptype = pd_.get("number") or 1, pd_.get("periodType") or "REG"
        if ptype == "SO":
            continue                       # shootouts are not hockey we can learn from
        t = (period - 1) * 1200 + mmss(p.get("timeInPeriod"))
        kind = p.get("typeDescKey")
        d = p.get("details") or {}
        ag, ask, hsk, hg = situation(p.get("situationCode"))
        owner = team_of_id.get(d.get("eventOwnerTeamId"))

        # ---- time spent in each strength state (counted until the next event)
        if prev is not None and prev["period"] == period and t >= prev["t"]:
            dt = t - prev["t"]
            pag, pas, phs, phg = prev["sit"]
            if pag and phg:
                if pas == 5 and phs == 5:
                    counts[home]["toi5"] += dt
                    counts[away]["toi5"] += dt
                elif phs > pas:
                    counts[home]["pp_toi"] += dt
                    counts[away]["pk_toi"] += dt
                elif pas > phs:
                    counts[away]["pp_toi"] += dt
                    counts[home]["pk_toi"] += dt
            counts[home]["toi"] += dt
            counts[away]["toi"] += dt

        x, y = d.get("xCoord"), d.get("yCoord")

        if kind in SHOT_KINDS:
            shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId")
            team = roster.get(shooter)
            if team is None:               # unknown shooter: blocked shots are owned by the blocker
                team = other.get(owner) if kind == "blocked-shot" else owner
            if team is None:
                prev = None
                continue
            opp = other[team]
            is_home = team == home
            sk_for, sk_against = (hsk, ask) if is_home else (ask, hsk)
            own_g, opp_g = (hg, ag) if is_home else (ag, hg)
            sign = signs.get((team, period), 1)
            if x is not None and y is not None:
                xn, yn = x * sign, y * sign
                dist = math.hypot(NET_X - xn, yn)
                angle = math.degrees(math.atan2(abs(yn), NET_X - xn))  # >90 means behind the net
            else:
                xn = yn = dist = angle = float("nan")

            # context from the previous event
            dt = t - prev["t"] if prev and prev["period"] == period else 999
            prev_xn = prev["x"] * sign if prev and prev["x"] is not None else None
            rebound = int(prev is not None and dt <= 3 and prev.get("shot_team") == team)
            rush = int(prev is not None and dt <= 4 and prev_xn is not None and prev_xn < BLUE_LINE_X)
            turnover = int(prev is not None and dt <= 5 and (
                (prev["kind"] == "takeaway" and prev["owner"] == team) or
                (prev["kind"] == "giveaway" and prev["owner"] == opp)))
            faceoff = int(prev is not None and dt <= 5 and prev["kind"] == "faceoff")
            if (prev is not None and dt < 999 and None not in (prev["x"], prev["y"], x, y)):
                moved = math.hypot(prev["x"] - x, prev["y"] - y)
                speed = min(moved / max(dt, 1), 60.0)
            else:
                speed = 0.0

            shots.append({
                "game_id": gid, "season": season, "period": period, "t": t,
                "team": team, "opp": opp, "is_home": int(is_home),
                "kind": SHOT_KINDS[kind], "is_goal": int(kind == "goal"),
                "shot_type": d.get("shotType") or "unknown",
                "x": xn, "y": yn, "dist": dist, "angle": angle,
                "rebound": rebound, "rush": rush, "after_turnover": turnover, "after_faceoff": faceoff,
                "secs_prev": min(dt, 60), "speed": speed,
                "sk_for": sk_for, "sk_against": sk_against,
                "empty_net": int(opp_g == 0), "own_goalie_pulled": int(own_g == 0),
                "score_diff": score[team] - score[opp], "ot": int(ptype == "OT"),
                "shooter": shooter or 0, "goalie": d.get("goalieInNetId") or 0,
            })
            if kind == "blocked-shot":
                counts[opp]["blocks"] += 1
            if kind == "goal":
                score[team] += 1
                goals_noso[team] += 1
                if opp_g == 0:
                    counts[team]["gf_en"] += 1
                for key, col in (("assist1PlayerId", "a1"), ("assist2PlayerId", "a2")):
                    if d.get(key):
                        per_player[d[key]][col] += 1
            shot_team = team
        else:
            shot_team = None
            if owner is not None:
                zone = d.get("zoneCode")
                if kind == "hit":
                    counts[owner]["hits"] += 1
                    counts[owner]["oz_hits"] += int(zone == "O")
                elif kind == "takeaway":
                    counts[owner]["tk"] += 1
                    counts[owner]["oz_tk"] += int(zone == "O")
                elif kind == "giveaway":
                    counts[owner]["gv"] += 1
                    counts[owner]["dz_gv"] += int(zone == "D")
                elif kind == "faceoff":
                    counts[owner]["fow"] += 1
                    per_player[d.get("winningPlayerId")]["fow"] += 1
                    per_player[d.get("losingPlayerId")]["fol"] += 1
                elif kind == "penalty":
                    if d.get("typeCode") in PENALTY_TYPES:
                        counts[owner]["pen_taken"] += 1
                    per_player[d.get("committedByPlayerId")]["pt"] += 1
                    per_player[d.get("drawnByPlayerId")]["pd"] += 1

        prev = {"t": t, "period": period, "kind": kind, "owner": owner, "x": x, "y": y,
                "sit": (ag, ask, hsk, hg), "shot_team": shot_team}

    # ------------------------------------------------------------ game row
    outcome = (box.get("gameOutcome") or pbp.get("gameOutcome") or {}).get("lastPeriodType", "REG")
    hs, as_ = box["homeTeam"]["score"], box["awayTeam"]["score"]
    game = {
        "game_id": gid, "season": season, "date": str(pbp.get("gameDate") or box.get("gameDate")),
        "start_utc": pbp.get("startTimeUTC"), "home": home, "away": away,
        "home_score": hs, "away_score": as_, "home_win": int(hs > as_), "last_period": outcome,
        "home_goals": goals_noso[home], "away_goals": goals_noso[away],   # no shootout goals
        "neutral": int(neutral_site(pbp)), "venue": (pbp.get("venue") or {}).get("default", ""),
    }

    # ------------------------------------------------------------ team counts
    count_rows = []
    for team in (home, away):
        row = {"game_id": gid, "team": team}
        for k in ("toi", "toi5", "pp_toi", "pk_toi", "hits", "oz_hits", "tk", "oz_tk", "gv", "dz_gv",
                  "fow", "pen_taken", "blocks", "gf_en"):
            row[k] = counts[team][k]
        row["fol"] = counts[other[team]]["fow"]
        row["pen_drawn"] = counts[other[team]]["pen_taken"]
        count_rows.append(row)

    # ------------------------------------------------------------ skaters + goalies
    skaters, goalies = [], []
    pbgs = box.get("playerByGameStats") or {}
    for side, team in (("homeTeam", home), ("awayTeam", away)):
        tstats = pbgs.get(side) or {}
        for group in ("forwards", "defense"):
            for s in tstats.get(group, []):
                pid = s["playerId"]
                c = per_player[pid]
                a1, a2 = c["a1"], c["a2"]
                if a1 + a2 == 0 and s.get("assists", 0):      # play-by-play missing assists
                    a1 = s["assists"]
                gs = (0.75 * s.get("goals", 0) + 0.7 * a1 + 0.55 * a2 + 0.075 * s.get("sog", 0)
                      + 0.05 * s.get("blockedShots", 0) + 0.15 * c["pd"] - 0.15 * c["pt"]
                      + 0.01 * c["fow"] - 0.01 * c["fol"] + 0.15 * s.get("plusMinus", 0))
                skaters.append({
                    "game_id": gid, "team": team, "player_id": pid,
                    "name": (s.get("name") or {}).get("default", ""),
                    "pos": "D" if group == "defense" else "F", "toi": mmss(s.get("toi")),
                    "goals": s.get("goals", 0), "assists": s.get("assists", 0),
                    "sog": s.get("sog", 0), "gs": round(gs, 4),
                })
        for g in tstats.get("goalies", []):
            goalies.append({
                "game_id": gid, "team": team, "goalie_id": g["playerId"],
                "name": (g.get("name") or {}).get("default", ""), "starter": int(bool(g.get("starter"))),
                "toi": mmss(g.get("toi")), "shots_against": g.get("shotsAgainst", 0) or 0,
                "goals_against": g.get("goalsAgainst", 0) or 0, "decision": g.get("decision") or "",
            })
    # older boxscores sometimes lack the starter flag: the goalie with the most ice time started
    for team in (home, away):
        tg = [g for g in goalies if g["team"] == team]
        if tg and not any(g["starter"] for g in tg):
            max(tg, key=lambda g: g["toi"])["starter"] = 1

    return {"game": game, "shots": shots, "counts": count_rows, "skaters": skaters, "goalies": goalies}
