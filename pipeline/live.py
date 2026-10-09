"""
Live predictions for tonight, built from the nightly snapshot (so the 10-minute pregame job
stays fast). The math is the SAME code used in training - features.team_values, make_X,
win_model.predict - so what was tested is exactly what runs live.
"""
import json
import math
import unicodedata
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config, features, goalies, market, players, ratings, teams, win_model

ET = ZoneInfo(config.TIMEZONE)
STARTED = {"LIVE", "CRIT", "FINAL", "OFF"}
TIE_INFLATE = 1.4           # regulation ties: ~23% of NHL games reach OT, more than plain Poisson says


def now_utc():
    return datetime.now(timezone.utc)


def site_date(now=None):
    """The site's 'today' in Eastern time; before 5 am it still shows last night."""
    t = (now or now_utc()).astimezone(ET)
    if t.hour < config.DAY_ROLLOVER_HOUR:
        t -= timedelta(days=1)
    return t.date().isoformat()


def parse_utc(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def clean(o):
    """numpy numbers/bools -> plain Python (so JSON and the tape hash are exact)."""
    if isinstance(o, dict):
        return {k if isinstance(k, str) else str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def write_json(path, obj, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=indent, default=str, sort_keys=True))


# ================================================================== snapshot (built nightly)
def pbook_to_json(b):
    return {"prior": b.prior, "pl": {str(k): v for k, v in b.pl.items()},
            "recent": {t: [sorted(int(x) for x in s) for s in q] for t, q in b.team_recent.items()},
            "lineups": {t: list(q) for t, q in b.team_lineups.items()}}


def pbook_from_json(d):
    b = players.PlayerBook(d["prior"]["F"], d["prior"]["D"])
    b.pl = {int(k): v for k, v in d["pl"].items()}
    b.team_recent = {t: deque([set(s) for s in q], maxlen=10) for t, q in d["recent"].items()}
    b.team_lineups = {t: deque(q, maxlen=10) for t, q in d["lineups"].items()}
    return b


def gbook_to_json(b):
    out = {}
    for gid, r in b.g.items():
        rr = dict(r)
        rr["last_date"] = str(pd.Timestamp(r["last_date"]).date()) if r["last_date"] is not None else None
        out[str(gid)] = rr
    return out


def gbook_from_json(d):
    b = goalies.GoalieBook()
    b.g = {int(k): v for k, v in d.items()}
    return b


def make_snapshot(world, model, skaters_recent=None):
    """Everything pregame needs, as of the last finished game."""
    st = model["settings"]
    season = config.LIVE_SEASON
    tg = world["tg"]
    sums = features.next_game_sums(tg, st["half_life"], st["carryover"], season)
    team_info = {}
    for fr, grp in tg.groupby("franchise"):
        last = grp.iloc[-1]
        team_info[fr] = {**sums[fr], "last_date": str(pd.Timestamp(last["date"]).date()),
                         "last_arena": last["home"], "last_season": int(last["season"])}
    gg = world["gg"]
    depth = {}
    for team, grp in gg[gg["starter"] == 1].groupby(gg["team"].map(teams.current_franchise)):
        recent = grp.sort_values("date").tail(20)
        depth[team] = [int(x) for x in recent["goalie_id"].value_counts().index[:3]]
    games = world["games"]
    tot = (games["home_goals"] + games["away_goals"]).mean()
    return {
        "built_at": now_utc().isoformat(timespec="seconds"),
        "through": str(pd.Timestamp(tg["date"].max()).date()),
        "season": season, "settings": st, "league": world["league"],
        "teams": team_info, "kalman": ratings.to_json(world["kal_state"][st["kalman"]]),
        "kparams": world["kparams"][st["kalman"]],
        "goalies": gbook_to_json(world["gbook"]), "team_goalies": depth,
        "players": pbook_to_json(world["pbook"]), "league_goals": round(float(tot), 3),
        "edge": {},
    }


# ================================================================== name matching
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return s.replace(".", " ").replace("-", " ").split()


def match_name(full_name, candidates):
    """
    full_name: 'Auston Matthews' (Daily Faceoff). candidates: {id: 'A. Matthews'} (NHL boxscores)
    Matches on last name + first initial, then last name alone if unique.
    """
    parts = _norm(full_name)
    if not parts:
        return None
    first, last = parts[0][0], " ".join(parts[1:]) or parts[0]
    exact, loose = [], []
    for pid, cand in candidates.items():
        c = _norm(cand)
        if not c:
            continue
        c_last = " ".join(c[1:]) if len(c) > 1 else c[0]
        if c_last == last or c[-1] == parts[-1]:
            loose.append(pid)
            if c[0][0] == first:
                exact.append(pid)
    if len(exact) == 1:
        return exact[0]
    if len(loose) == 1:
        return loose[0]
    return exact[0] if exact else None


# ================================================================== pieces of one game
def team_stat_values(snap, fr):
    t = snap["teams"].get(fr)
    lg = snap["league"]
    if t is None:  # never seen: league average
        return {s: lg[s][0] / lg[s][1] if lg[s][1] else 0.0 for s in features.STAT_NAMES}, 0
    vals = features.team_values(t["num"], t["den"], lg)
    return dict(zip(features.STAT_NAMES, vals.tolist())), t["gp"]


def schedule_bits(snap, fr, arena, game_date):
    t = snap["teams"].get(fr)
    if not t or t["last_season"] != config.LIVE_SEASON:
        return {"rest": 4.0, "travel_km": 0.0, "tz": 0.0}
    days = (pd.Timestamp(game_date) - pd.Timestamp(t["last_date"])).days
    return {"rest": float(min(max(days, 0), 4)), "travel_km": teams.distance_km(t["last_arena"], arena),
            "tz": float(abs(teams.tz_offset(t["last_arena"]) - teams.tz_offset(arena)))}


def goalie_side(snap, gbook, team, dfo_name, dfo_status, rest, game_date):
    """Expected goalie numbers, mixing the likely starter and the backup by starter probability."""
    depth = snap["team_goalies"].get(teams.current_franchise(team), [])
    names = {gid: gbook.g[gid]["name"] for gid in depth if gid in gbook.g}
    team_goalies = {gid: r["name"] for gid, r in gbook.g.items() if r.get("team") == team}
    starter, status = None, dfo_status or "Unconfirmed"
    if dfo_name:
        starter = match_name(dfo_name, team_goalies) or match_name(dfo_name, names)
    if starter is None:
        status = "Projected"
        starter = depth[0] if depth else None
        yesterday = str((pd.Timestamp(game_date) - pd.Timedelta(days=1)).date())
        if (starter is not None and rest == 1 and len(depth) > 1 and
                yesterday in gbook.g.get(starter, {}).get("recent_start_dates", [])):
            starter = depth[1]         # back-to-back: the backup usually plays the second night
    p = config.GOALIE["starter_prob"].get(status, 0.6)
    backup = next((g for g in depth if g != starter), None)
    cs = gbook.components(starter) if starter is not None else gbook.components(-1)
    cb = gbook.components(backup) if backup is not None else gbook.components(-1)
    if backup is None:
        p = 1.0
    return {
        "g_talent": p * cs["talent"] + (1 - p) * cb["talent"], "g_gate": 1.0,
        "g_form": p * cs["gate"] * cs["form"] + (1 - p) * cb["gate"] * cb["form"],
        "name": dfo_name or (gbook.g[starter]["name"] if starter in gbook.g else "TBD"),
        "id": starter, "status": status, "prob": p, "talent": round(cs["talent"], 3),
        "form": round(cs["form"], 3), "streak_t": round(cs["t"], 2), "gate": round(cs["gate"], 2),
    }


def lineup_side(snap, pbook, team, dfo_players, roster):
    """Tonight's lineup vs the usual one. dfo_players: Daily Faceoff line list (or None)."""
    dressed, source = [], "last game"
    if dfo_players:
        cands = {int(p["id"]): f'{p["first"]} {p["last"]}' for p in (roster or [])}
        if not cands:
            cands = {pid: r[5] for pid, r in pbook.pl.items() if r[6] == team}
        for p in dfo_players:
            grp = str(p.get("groupIdentifier", ""))
            if p.get("categoryIdentifier") not in (None, "ev") or not grp[:1] in ("f", "d"):
                continue
            pid = match_name(p.get("name") or p.get("playerName") or "", cands)
            if pid is not None:
                dressed.append((pid, "F" if grp.startswith("f") else "D"))
        if len(dressed) >= 10:
            source = "Daily Faceoff"
        else:
            dressed = []
    if not dressed:
        recent = pbook.team_recent.get(team)
        if recent:
            last = recent[-1]
            dressed = [(pid, pbook.pl[pid][4]) for pid in last if pid in pbook.pl]
    pre = pbook.pregame(team, dressed)
    pre["missing_names"] = [pbook.pl[pid][5] for pid in pre["missing_ids"] if pid in pbook.pl]
    pre["source"] = source
    return pre


def scoreline(p_home, total):
    """Projected goals and P(overtime) from a Poisson model matched to the win probability."""
    def pois(lam):
        return [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(16)]

    def outcome(s):
        ph, pa = pois(total * s), pois(total * (1 - s))
        tie = min(0.6, TIE_INFLATE * sum(ph[k] * pa[k] for k in range(16)))
        hw = sum(ph[i] * pa[j] for i in range(16) for j in range(i))
        aw = sum(ph[i] * pa[j] for j in range(16) for i in range(j))
        hw, aw = (1 - tie) * hw / (hw + aw), (1 - tie) * aw / (hw + aw)
        return hw + tie * (0.5 + 0.5 * (s - 0.5)), tie

    lo, hi = 0.2, 0.8
    for _ in range(40):
        mid = (lo + hi) / 2
        if outcome(mid)[0] < p_home:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return {"home": round(total * s, 2), "away": round(total * (1 - s), 2), "p_ot": round(outcome(s)[1], 3)}


# ================================================================== one game, end to end
def predict_game(snap, model, booster, game, ctx):
    """
    game: an NHL schedule entry. ctx: dfo goalies/lines, rosters, lines memory, books, trust.
    Returns the full prediction body (what gets locked into the tape).
    """
    st = snap["settings"]
    home, away = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
    hf, af = teams.current_franchise(home), teams.current_franchise(away)
    start = parse_utc(game["startTimeUTC"])
    gdate = start.astimezone(ET).date().isoformat()
    neutral = bool(game.get("neutralSite", False))
    arena = home

    row = {"neutral": int(neutral)}
    hv, hgp = team_stat_values(snap, hf)
    av, agp = team_stat_values(snap, af)
    for s in features.STAT_NAMES:
        row[f"h_{s}"], row[f"a_{s}"] = hv[s], av[s]
    hs, as_ = schedule_bits(snap, hf, arena, gdate), schedule_bits(snap, af, arena, gdate)
    for k in ("rest", "travel_km", "tz"):
        row[f"h_{k}"], row[f"a_{k}"] = hs[k], as_[k]

    kstate = ratings.from_json(snap["kalman"])
    kstate = ratings.advance(kstate, pd.Timestamp(gdate), config.LIVE_SEASON, snap["kparams"])
    diff, var = ratings.pregame(kstate, home, away)
    row[f"rating_diff_{st['kalman']}"], row[f"rating_var_{st['kalman']}"] = diff, var

    gbook, pbook = ctx["gbook"], ctx["pbook"]
    dfo = ctx["dfo_goalies"].get((home, away), {})
    hg = goalie_side(snap, gbook, home, dfo.get("home_name"), dfo.get("home_status"), hs["rest"], gdate)
    ag = goalie_side(snap, gbook, away, dfo.get("away_name"), dfo.get("away_status"), as_["rest"], gdate)
    for pre, g in (("h", hg), ("a", ag)):
        row[f"{pre}_g_talent"], row[f"{pre}_g_gate"], row[f"{pre}_g_form"] = g["g_talent"], g["g_gate"], g["g_form"]
    hl = lineup_side(snap, pbook, home, ctx["dfo_lines"].get(home), ctx["rosters"].get(home))
    al = lineup_side(snap, pbook, away, ctx["dfo_lines"].get(away), ctx["rosters"].get(away))
    row["h_lineup_delta"], row["a_lineup_delta"] = hl["lineup_delta"], al["lineup_delta"]
    row["h_stars_missing"], row["a_stars_missing"] = hl["stars_missing"], al["stars_missing"]
    edge = snap.get("edge", {})
    row["edge_speed"] = edge.get(home, {}).get("speed", 0.0) - edge.get(away, {}).get("speed", 0.0)
    row["edge_ozone"] = edge.get(home, {}).get("ozone", 0.0) - edge.get(away, {}).get("ozone", 0.0)
    gp = min(hgp, agp)

    X = features.make_X(pd.DataFrame([row]), st["goalie_beta"], st["kalman"])
    res = win_model.predict(model, booster, X, [gp])
    ex = win_model.explain(res, 0, X.iloc[0])
    p = float(res["p"][0])

    # ---------------------------------------------------------- market side
    quotes = market.quotes_from_schedule(game) + ctx.get("extra_quotes", {}).get(game["id"], [])
    cons = market.consensus(quotes)
    mem = ctx["lines"].setdefault(str(game["id"]), {})
    if cons:
        mem.setdefault("open", {"p_home": cons["p_home"], "at": now_utc().isoformat(timespec="seconds")})
        mem["last"] = {"p_home": cons["p_home"], "at": now_utc().isoformat(timespec="seconds")}
    body_market, decision, desk = None, None, {"triggered": False, "passed": False, "checks": []}
    if cons:
        desk = market.risk_desk(p, cons["p_home"], {
            "p_open": mem.get("open", {}).get("p_home"), "explain": ex["all"],
            "goalie_confirmed": hg["status"] == "Confirmed" and ag["status"] == "Confirmed",
            "gp_home": hgp, "gp_away": agp, "data_age_hours": ctx.get("data_age_hours", 0.0)})
        decision = market.decide(p, cons, desk, ctx["bankrolls"], ctx["trust"])
        body_market = {"p_home": round(cons["p_home"], 4), "p_open": round(mem["open"]["p_home"], 4),
                       "best_home": round(cons["best_home"], 3), "best_away": round(cons["best_away"], 3),
                       "best_home_book": cons["best_home_book"], "best_away_book": cons["best_away_book"],
                       "books": cons["books"], "vig": cons["vig"]}

    pick = home if p >= 0.5 else away
    return clean({
        "game_id": int(game["id"]), "season": int(game.get("season", config.LIVE_SEASON)), "date": gdate,
        "start_utc": game["startTimeUTC"], "home": home, "away": away, "neutral": neutral,
        "p_home": round(p, 4), "pick": pick, "p_pick": round(max(p, 1 - p), 4),
        "computed_at": now_utc().isoformat(timespec="seconds"), "model_version": config.MODEL_VERSION,
        "online_updates": int(model.get("online_updates", 0)), "gp": int(gp),
        "x": {k: round(float(v), 5) for k, v in X.iloc[0].items()},
        "why": {"p_base": ex["p_base"], "items": ex["all"]},
        "goalies": {"home": {k: hg[k] for k in ("name", "status", "prob", "talent", "form", "streak_t", "gate")},
                    "away": {k: ag[k] for k in ("name", "status", "prob", "talent", "form", "streak_t", "gate")}},
        "lineups": {"home": {"delta": round(hl["lineup_delta"], 3), "missing": hl["missing_names"], "source": hl["source"]},
                    "away": {"delta": round(al["lineup_delta"], 3), "missing": al["missing_names"], "source": al["source"]}},
        "teams": {"home": {s: round(hv[s], 4) for s in features.STAT_NAMES} | {"gp": hgp, **hs},
                  "away": {s: round(av[s], 4) for s in features.STAT_NAMES} | {"gp": agp, **as_}},
        "rating": {"diff": round(diff, 3), "sd": round(math.sqrt(var), 3)},
        "projected": scoreline(p, snap.get("league_goals", 6.1)),
        "market": body_market, "desk": desk, "decision": decision,
    })


# ================================================================== locking rules
def should_lock(game, now):
    start = parse_utc(game["startTimeUTC"])
    return game.get("gameState") in STARTED or (start - now) <= timedelta(minutes=config.LOCK_LEAD_MINUTES)


def lock_body(provisional_entry, game, now):
    """The locked record = the latest prediction computed BEFORE puck drop."""
    body = dict(provisional_entry)
    body["locked_at"] = now.isoformat(timespec="seconds")
    start = parse_utc(game["startTimeUTC"])
    if parse_utc(body["computed_at"]) > start:
        return None                       # never lock something computed after the game began
    return body
