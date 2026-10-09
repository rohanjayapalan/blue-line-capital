"""
Command line - GitHub Actions runs these:

  python -m pipeline.run train     full rebuild: history, backtest, live model, snapshot (run once / weekly)
  python -m pipeline.run nightly   ingest last night's games, grade the tape, settle bets, learn, rebuild snapshot
  python -m pipeline.run pregame   every 10 min on game days: predict, lock at puck drop, publish
  python -m pipeline.run doctor    check every data source
"""
import argparse
import sys
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import (backtest, config, features, health, live, market, nhl_api, publish, store, tape, teams,
               win_model)
from .health import DataFormatError, SourceDownError

LIVE = config.LIVE_SEASON


def log(*a):
    print(*a, flush=True)


def S(name):
    return config.STATE_DIR / name


# ============================================================== shared helpers
def live_world():
    """Everything rebuilt from the tables using the SAVED xG model (fast and repeatable)."""
    hist = [s for s in store.seasons_on_disk() if s < LIVE]
    return backtest.build_world(hist, max(store.seasons_on_disk()), log=log,
                                xgm=live.read_json(S("xg_model.json"), None),
                                adjust=live.read_json(S("score_adjust.json"), None),
                                league=live.read_json(S("league.json"), None))


def save_snapshot(world, model, edge=None):
    snap = live.make_snapshot(world, model)
    old = live.read_json(S("snapshot.json"), {})
    snap["edge"] = edge if edge is not None else old.get("edge", {})
    live.write_json(S("snapshot.json"), snap)
    log(f"[snapshot] through {snap['through']}, {len(snap['teams'])} teams")
    return snap


def empty_record():
    return {"graded": [], "books": {b: [] for b in market.BOOK_NAMES}, "learning_log": [], "voided": []}


def bankrolls(record):
    start = config.BET["start_bankroll"]
    return {b: start + sum(h["profit"] for h in record["books"].get(b, [])) for b in ("always", "edge")}


def trust_level(record):
    g = [r for r in record["graded"] if not r.get("void")]
    with_mkt = [r for r in g if r.get("market_p_home") is not None]
    if len(with_mkt) >= 30:
        y = np.array([r["home_win"] for r in with_mkt], float)
        pm = np.clip([r["p_home"] for r in with_mkt], 1e-4, 1 - 1e-4)
        pk = np.clip([r["market_p_home"] for r in with_mkt], 1e-4, 1 - 1e-4)
        ll = lambda p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return market.trust(len(g), ll(pm), ll(pk))
    return market.trust(len(g))


# ============================================================== train
def cmd_train(args):
    for s in config.ALL_HISTORY:
        if not store.has(s, "shots"):
            log(f"[train] building {s} from the raw mirror")
            nhl_api.download_season(s)
            store.build_season(s)
    if args.odds:
        try:
            from . import odds_history
            odds_history.backfill(log=log)
        except Exception as e:            # historical odds are a bonus, never a blocker
            log(f"[train] odds backfill skipped: {e!r}")
    if not args.skip_backtest:
        backtest.run(log=log)
    hist = [s for s in store.seasons_on_disk() if s < LIVE]
    world = backtest.build_world(hist, max(store.seasons_on_disk()), log=log)
    train_seasons = [s for s in hist if s <= LIVE - 10001 * (1 + config.TRAIN_GAP)
                     and s not in config.SKIP_FOR_TRAINING]
    model, booster = win_model.train(backtest.tables_for(world), train_seasons, log)
    win_model.save(model, booster)
    live.write_json(S("xg_model.json"), world["xg"], indent=1)
    live.write_json(S("score_adjust.json"), world["adjust"], indent=1)
    live.write_json(S("league.json"), world["league"], indent=1)
    save_snapshot(world, model)
    publish.run(log=log)


# ============================================================== nightly
def finished_games(start, end):
    """Regular-season games between two dates: (finished ids, postponed/cancelled ids)."""
    done, void = [], []
    d = start
    while d <= end:
        js = nhl_api.schedule(d.isoformat())
        for day in js.get("gameWeek", []):
            if start.isoformat() <= day["date"] <= end.isoformat():
                for g in day.get("games", []):
                    if g.get("gameType") != 2:
                        continue
                    if g.get("gameScheduleState") in ("PPD", "CNCL"):
                        void.append(g["id"])
                    elif g.get("gameState") in ("OFF", "FINAL"):
                        done.append(g["id"])
        d += timedelta(days=7)
    return done, void


def grade_and_learn(world, void_ids, today):
    """Grade every locked prediction whose game is over; settle bets; one learning step per game."""
    record = live.read_json(S("record.json"), empty_record())
    model, booster = win_model.load()
    before = model["coef"].copy()
    done = {r["game_id"] for r in record["graded"]}
    g = world["games"].set_index("game_id")
    new = []
    for gid, (body, h) in sorted(tape.locked(LIVE).items(), key=lambda kv: kv[1][0]["start_utc"]):
        if gid in done:
            continue
        if gid in void_ids:
            record["graded"].append({"game_id": gid, "date": body["date"], "home": body["home"],
                                     "away": body["away"], "void": True, "hash": h})
            continue
        if gid not in g.index:
            continue
        r = g.loc[gid]
        y = int(r["home_win"])
        entry = {"game_id": gid, "date": body["date"], "start_utc": body["start_utc"], "home": body["home"],
                 "away": body["away"], "p_home": body["p_home"], "pick": body["pick"], "p_pick": body["p_pick"],
                 "home_win": y, "hit": int((body["p_home"] >= 0.5) == bool(y)),
                 "home_score": int(r["home_score"]), "away_score": int(r["away_score"]),
                 "last_period": r["last_period"], "hash": h, "steal": False,
                 "market_p_home": (body.get("market") or {}).get("p_home"),
                 "market_p_open": (body.get("market") or {}).get("p_open"), "profit": {}}
        dec = body.get("decision")
        if dec:
            entry["steal"] = bool(dec.get("steal"))
            for book, bet in dec["bets"].items():
                pnl = market.settle(bet, y)
                entry["profit"][book] = pnl
                record["books"].setdefault(book, []).append(
                    {"date": body["date"], "game_id": gid, "profit": pnl, "stake": bet["stake"]})
        x = np.array([body["x"][f] for f in model["features"]], float)
        win_model.online_update(model, x, y)
        record["graded"].append(entry)
        new.append(entry)
    if new:
        win_model.save(model, booster)
        moves = sorted(zip(model["features"], before, model["coef"]), key=lambda t: -abs(t[2] - t[1]))[:3]
        miss = max((e for e in new if not e["hit"]), key=lambda e: e["p_pick"], default=None)
        record["learning_log"].append({
            "date": today, "games": len(new), "hits": sum(e["hit"] for e in new),
            "moves": [{"feature": f, "label": features.FEATURE_INFO.get(f, f), "before": round(float(b), 4),
                       "after": round(float(a), 4)} for f, b, a in moves],
            "biggest_miss": None if miss is None else {
                "game": f'{miss["away"]} @ {miss["home"]}', "pick": miss["pick"], "p_pick": miss["p_pick"],
                "score": f'{miss["away_score"]}-{miss["home_score"]}'}})
        record["learning_log"] = record["learning_log"][-120:]
    live.write_json(S("record.json"), record)
    log(f"[grade] {len(new)} newly graded games")
    return record, model


def refresh_rosters():
    out = {}
    for team in sorted({teams.current_franchise(t) for t in teams.TEAMS}):
        try:
            js = nhl_api.roster(team)
        except (DataFormatError, SourceDownError):
            continue
        if not js:
            continue
        rows = []
        for grp, pos in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
            for p in js.get(grp, []):
                rows.append({"id": p["id"], "first": (p.get("firstName") or {}).get("default", ""),
                             "last": (p.get("lastName") or {}).get("default", ""), "pos": pos})
        out[team] = rows
    if len(out) >= 20:
        live.write_json(S("rosters.json"), out)
    return out


def _find_number(js, test):
    stack = [js]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if test(k):
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, dict):
                        for kk in ("value", "total", "count", "pctg"):
                            if isinstance(v.get(kk), (int, float)):
                                return float(v[kk])
                stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return None


def refresh_edge(snap):
    """NHL EDGE tracking (live-only features). Best effort: an empty dict just means weight 0."""
    raw = {}
    for team, tid in teams.NHL_IDS.items():
        if team != teams.current_franchise(team):
            continue
        try:
            js = nhl_api.edge_team(tid)
        except (DataFormatError, SourceDownError):
            continue
        gp = snap["teams"].get(team, {}).get("gp", 0)
        bursts = _find_number(js, lambda k: "burst" in k.lower() and "20" in k)
        oz = _find_number(js, lambda k: "offensive" in k.lower() and "zone" in k.lower())
        if bursts is not None and oz is not None and gp > 0:
            raw[team] = (bursts / gp, oz)
    if len(raw) < 20:
        return {}
    arr = np.array(list(raw.values()))
    z = (arr - arr.mean(0)) / np.where(arr.std(0) > 0, arr.std(0), 1)
    return {t: {"speed": round(float(a), 3), "ozone": round(float(b), 3)} for t, (a, b) in zip(raw, z)}


def cmd_nightly(args):
    today = date.fromisoformat(live.site_date())
    have = store.load("games", [LIVE])
    start = date.fromisoformat(config.LIVE_SEASON_START)
    if len(have):
        start = max(start, date.fromisoformat(str(have["date"].max())) - timedelta(days=2))
    end = today - timedelta(days=1)
    done, void = ([], []) if end < start else finished_games(start, end)
    new_ids = [g for g in done if len(have) == 0 or g not in set(have["game_id"])]
    if new_ids:
        log(f"[nightly] ingesting {store.ingest(new_ids, LIVE)} games")
    world = live_world()
    record = live.read_json(S("record.json"), empty_record())
    record["voided"] = sorted(set(record.get("voided", [])) | set(void))
    live.write_json(S("record.json"), record)
    record, model = grade_and_learn(world, set(record["voided"]), today.isoformat())
    snap = save_snapshot(world, model)
    try:
        refresh_rosters()
    except Exception as e:
        log(f"[nightly] rosters skipped: {e!r}")
    try:
        edge = refresh_edge(snap)
        if edge:
            snap["edge"] = edge
            live.write_json(S("snapshot.json"), snap)
    except Exception as e:
        log(f"[nightly] EDGE skipped: {e!r}")
    try:
        games_today = nhl_api.games_on(live.site_date())
    except (DataFormatError, SourceDownError):
        games_today = None
    publish.run(games=games_today, log=log)


# ============================================================== pregame
def dfo_goalie_map(rows):
    out = {}
    for r in rows or []:
        h = teams.SLUG_TO_ABBREV.get(r.get("homeTeamSlug"))
        a = teams.SLUG_TO_ABBREV.get(r.get("awayTeamSlug"))
        if h and a:
            out[(h, a)] = {"home_name": r.get("homeGoalieName"), "home_status": r.get("homeNewsStrengthName"),
                           "away_name": r.get("awayGoalieName"), "away_status": r.get("awayNewsStrengthName")}
    return out


def dfo_lines_cached(team_list, h=None):
    cache = live.read_json(S("lines_cache.json"), {})
    now = live.now_utc()
    for team in team_list:
        c = cache.get(team)
        fresh = c and (now - live.parse_utc(c["at"])).total_seconds() < 3 * 3600
        if fresh or team not in teams.DFO_SLUGS:
            continue
        try:
            players_ = nhl_api.dfo_lines(teams.DFO_SLUGS[team])
            if players_:
                cache[team] = {"at": now.isoformat(timespec="seconds"), "players": players_}
        except (DataFormatError, SourceDownError) as e:
            if h is not None:
                health.record(h, "Daily Faceoff lines", False, str(e))
            if isinstance(e, DataFormatError):
                health.open_github_issue("Data source changed format: Daily Faceoff lines",
                                         f"Line-combination parsing failed:\n\n```\n{e}\n```\nLineups fall back to each "
                                         "team's last game until this is fixed.")
    live.write_json(S("lines_cache.json"), cache)
    return {t: cache[t]["players"] for t in team_list if t in cache}


def cmd_pregame(args):
    now = live.now_utc()
    today = args.date or live.site_date(now)
    snap = live.read_json(S("snapshot.json"), None)
    if snap is None:
        log("[pregame] no snapshot yet - run `train` first")
        return
    h = health.load_health()
    games = nhl_api.games_on(today)
    health.record(h, "NHL schedule", True)
    locked = tape.locked(LIVE)
    prov = live.read_json(S("provisional.json"), {})
    lines = live.read_json(S("lines.json"), {})
    todo = [g for g in games if g["id"] not in locked and g.get("gameScheduleState", "OK") == "OK"
            and g.get("gameState") not in live.STARTED]
    if todo:
        try:
            dfo = dfo_goalie_map(nhl_api.dfo_starting_goalies(today))
            health.record(h, "Daily Faceoff goalies", True)
        except (DataFormatError, SourceDownError) as e:
            dfo = {}
            health.record(h, "Daily Faceoff goalies", False, str(e))
            if isinstance(e, DataFormatError):
                health.open_github_issue("Data source changed format: Daily Faceoff goalies",
                                         f"Starting-goalie parsing failed:\n\n```\n{e}\n```\nPredictions fall back to "
                                         "usage-based goalie projections until this is fixed.")
        record = live.read_json(S("record.json"), empty_record())
        model, booster = win_model.load()
        age = (now - live.parse_utc(snap["built_at"])).total_seconds() / 3600
        ctx = {"gbook": live.gbook_from_json(snap["goalies"]), "pbook": live.pbook_from_json(snap["players"]),
               "dfo_goalies": dfo,
               "dfo_lines": dfo_lines_cached(sorted({g["homeTeam"]["abbrev"] for g in todo} |
                                                    {g["awayTeam"]["abbrev"] for g in todo}), h),
               "rosters": live.read_json(S("rosters.json"), {}), "lines": lines,
               "bankrolls": bankrolls(record), "trust": trust_level(record), "data_age_hours": age}
        for g in todo:
            prov[str(g["id"])] = live.predict_game(snap, model, booster, g, ctx)
        log(f"[pregame] {len(todo)} provisional predictions for {today}")
    for g in games:
        gid = g["id"]
        if gid in locked or str(gid) not in prov or g.get("gameScheduleState", "OK") != "OK":
            continue
        if live.should_lock(g, now):
            body = live.lock_body(prov[str(gid)], g, now)
            if body is not None:
                tape.append(LIVE, body)
                log(f"[pregame] LOCKED {body['away']} @ {body['home']}: {body['pick']} {body['p_pick']:.1%}")
    keep = (date.fromisoformat(today) - timedelta(days=2)).isoformat()
    prov = {k: v for k, v in prov.items() if v["date"] >= keep}
    live.write_json(S("provisional.json"), prov)
    live.write_json(S("lines.json"), {k: v for k, v in lines.items() if k in prov or int(k) in tape.locked(LIVE)})
    health.save_health(h)
    publish.run(today=today, games=games, log=log)


def cmd_doctor(args):
    h = health.load_health()
    checks = [("NHL schedule", lambda: nhl_api.schedule(live.site_date())),
              ("Daily Faceoff goalies", lambda: nhl_api.dfo_starting_goalies()),
              ("GitHub mirror", lambda: nhl_api.mirror_game(2025020001))]
    for name, fn in checks:
        try:
            fn()
            health.record(h, name, True)
            log(f"OK    {name}")
        except Exception as e:
            health.record(h, name, False, repr(e))
            log(f"FAIL  {name}: {e!r}")
    health.save_health(h)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pipeline.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--skip-backtest", action="store_true")
    t.add_argument("--odds", action="store_true", help="backfill historical odds from ESPN (slow)")
    sub.add_parser("nightly")
    p = sub.add_parser("pregame")
    p.add_argument("--date", default=None)
    sub.add_parser("doctor")
    sub.add_parser("backtest")
    args = ap.parse_args(argv)
    fn = {"train": cmd_train, "nightly": cmd_nightly, "pregame": cmd_pregame, "doctor": cmd_doctor,
          "backtest": lambda a: backtest.run(log=log)}[args.cmd]
    try:
        fn(args)
    except (DataFormatError, SourceDownError) as e:
        h = health.load_health()
        health.record(h, getattr(e, "source", "unknown"), False, str(e))
        health.save_health(h)
        health.open_github_issue(f"Data source problem: {getattr(e, 'source', 'unknown')}",
                                 f"`{args.cmd}` failed:\n\n```\n{e}\n```\n\nThe site keeps showing the last good data.")
        log(f"[{args.cmd}] DATA SOURCE PROBLEM: {e}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
