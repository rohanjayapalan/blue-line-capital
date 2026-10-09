"""
Writes the JSON files the website reads (the site never talks to the NHL directly):
  public/today.json    tonight: picks, win %, why, market, bets, lock status, live scores
  public/record.json   graded record, calibration, model-vs-market, bankroll books, backtest
  public/model.json    weights, settings, power ratings, goalies, learning log
  public/health.json   data-source status (written by health.py)
"""
import json
from datetime import datetime, timezone

import numpy as np

from . import backtest, config, features, health, live, market, ratings, tape, teams


def _state(name, default):
    return live.read_json(config.STATE_DIR / name, default)


def tier(p_pick, steal):
    if steal:
        return "Steal"
    if p_pick >= 0.62:
        return "Strong"
    if p_pick >= 0.55:
        return "Lean"
    return "Toss-up"


def detail(f, b):
    H, A = b["home"], b["away"]
    th, ta = b["teams"]["home"], b["teams"]["away"]
    pct = lambda v: f"{100 * v:.1f}%"
    if f == "rating_z":
        return f"Power-rating gap {b['rating']['diff']:+.2f} goals/game (uncertainty {b['rating']['sd']:.2f})"
    if f in ("xgf_pct", "cf_pct", "hd_pct", "gf_pct", "faceoff", "rush"):
        return f"{H} {pct(th[f])} vs {A} {pct(ta[f])}"
    if f in ("pp", "pk"):
        return f"{H} {th[f]:.2f} vs {A} {ta[f]:.2f} xG per 60"
    if f in ("pen", "finishing", "battles", "forecheck", "dz_gv"):
        return f"{H} {th[f]:+.2f} vs {A} {ta[f]:+.2f} per game"
    if f == "goalie":
        g = b["goalies"]
        return (f"{g['home']['name']} ({g['home']['status']}) {g['home']['talent']:+.2f} vs "
                f"{g['away']['name']} ({g['away']['status']}) {g['away']['talent']:+.2f} GSAx/60")
    if f in ("lineup", "star_missing"):
        L = b["lineups"]
        out = [f"{t} without {', '.join(L[s]['missing'][:3])}" for s, t in (("home", H), ("away", A)) if L[s]["missing"]]
        return "; ".join(out) or f"{H} {L['home']['delta']:+.2f} vs {A} {L['away']['delta']:+.2f} vs usual lineup"
    if f == "rest":
        return f"{H} {th['rest']:.0f} days vs {A} {ta['rest']:.0f} days since last game"
    if f == "b2b_home":
        return f"{H} played yesterday"
    if f == "b2b_away":
        return f"{A} played yesterday"
    if f == "travel_away":
        return f"{A} travelled {ta['travel_km']:.0f} km"
    if f == "tz_away":
        return f"{A} crossed {ta['tz']:.0f} time zones"
    if f == "neutral":
        return "Neutral-site game"
    return ""


def status_of(g, locked):
    if g.get("gameScheduleState", "OK") != "OK":
        return "postponed"
    st = g.get("gameState")
    if st in ("OFF", "FINAL"):
        return "final"
    if st in ("LIVE", "CRIT"):
        return "live"
    return "locked" if locked else "upcoming"


def card(g, body, lock_hash):
    H, A = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
    status = status_of(g, lock_hash is not None)
    c = {"id": g["id"], "start_utc": g["startTimeUTC"], "status": status,
         "venue": (g.get("venue") or {}).get("default", ""),
         "home": {"abbrev": H, "name": teams.display_name(H), "score": g["homeTeam"].get("score")},
         "away": {"abbrev": A, "name": teams.display_name(A), "score": g["awayTeam"].get("score")},
         "period": (g.get("periodDescriptor") or {}).get("periodType"),
         "prediction": None}
    if body is None:
        return c
    mk, dec = body.get("market"), body.get("decision")
    p, pick = body["p_home"], body["pick"]
    pick_home = pick == H
    pred = {
        "p_home": p, "pick": pick, "p_pick": body["p_pick"], "computed_at": body["computed_at"],
        "tier": tier(body["p_pick"], bool(dec and dec["steal"])), "projected": body["projected"],
        "goalies": body["goalies"], "lineups": body["lineups"], "p_base": body["why"]["p_base"],
        "why": [{"feature": it["feature"], "label": features.FEATURE_INFO.get(it["feature"], it["feature"]),
                 "pp_home": it["pp"], "detail": detail(it["feature"], body)}
                for it in body["why"]["items"] if abs(it["pp"]) >= 0.05][:7],
        "market": None, "edge_pp": None, "bet": None, "desk": body.get("desk"),
        "locked": {"at": body.get("locked_at"), "hash": lock_hash} if lock_hash else None,
    }
    if mk:
        pm_pick = mk["p_home"] if pick_home else 1 - mk["p_home"]
        pred["market"] = {"p_home": mk["p_home"], "p_open": mk["p_open"],
                          "home_price": market.to_american(mk["best_home"]),
                          "away_price": market.to_american(mk["best_away"]),
                          "home_book": mk["best_home_book"], "away_book": mk["best_away_book"], "books": mk["books"]}
        pred["edge_pp"] = round(100 * (body["p_pick"] - pm_pick), 1)
    if dec:
        a = dec["bets"]["always"]
        pred["bet"] = {"side": a["side"], "team": H if a["side"] == "home" else A, "stake": a["stake"],
                       "price": market.to_american(a["dec"]), "edge_stake": (dec["bets"].get("edge") or {}).get("stake"),
                       "ev": dec["ev"], "steal": dec["steal"], "alpha": dec["alpha"]}
    if status == "final" and c["home"]["score"] is not None:
        pred["hit"] = int((p >= 0.5) == (c["home"]["score"] > c["away"]["score"]))
    c["prediction"] = pred
    return c


def today_json(today, games):
    locked = tape.locked(config.LIVE_SEASON)
    prov = _state("provisional.json", {})
    snap = _state("snapshot.json", {})
    cards = []
    for g in sorted(games or [], key=lambda x: x["startTimeUTC"]):
        if g["id"] in locked:
            body, h = locked[g["id"]]
        else:
            body, h = prov.get(str(g["id"])), None
        cards.append(card(g, body, h))
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": today,
            "season": config.season_label(config.LIVE_SEASON), "season_start": config.LIVE_SEASON_START,
            "games": cards, "tape": tape.verify(config.LIVE_SEASON), "model_version": config.MODEL_VERSION,
            "snapshot_through": snap.get("through"), "lock_minutes": config.LOCK_LEAD_MINUTES}


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def record_json(record, bt):
    g = [r for r in record["graded"] if not r.get("void")]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "season": {"games": len(g)}, "books": {}, "graded": [], "calibration": [], "by_tier": {},
           "backtest": None, "learning_log": list(reversed(record.get("learning_log", [])))[:40]}
    if g:
        y = np.array([r["home_win"] for r in g])
        p = np.array([r["p_home"] for r in g])
        out["season"] = {"games": len(g), "hits": int(sum(r["hit"] for r in g)),
                         "accuracy": round(float(np.mean([r["hit"] for r in g])), 4),
                         "log_loss": round(_ll(y, p), 4), "brier": round(float(np.mean((p - y) ** 2)), 4)}
        m = [r for r in g if r.get("market_p_home") is not None]
        if m:
            ym = np.array([r["home_win"] for r in m])
            pm = np.array([r["market_p_home"] for r in m])
            pp = np.array([r["p_home"] for r in m])
            moves = [(r["market_p_home"] - r["market_p_open"], r["p_home"] - r["market_p_open"])
                     for r in m if r.get("market_p_open") is not None]
            moved = [mv for mv in moves if abs(mv[0]) >= 0.005]
            out["season"]["market"] = {
                "games": len(m), "accuracy": round(float(((pm >= 0.5) == ym).mean()), 4),
                "log_loss": round(_ll(ym, pm), 4), "model_log_loss": round(_ll(ym, pp), 4),
                "model_accuracy": round(float(((pp >= 0.5) == ym).mean()), 4),
                "followed_us": round(float(np.mean([np.sign(a) == np.sign(b) for a, b in moved])), 3) if moved else None}
        out["calibration"] = backtest.calibration(y, p)
        for r in g:
            t = tier(r["p_pick"], r.get("steal"))
            d = out["by_tier"].setdefault(t, {"games": 0, "hits": 0})
            d["games"] += 1
            d["hits"] += r["hit"]
        start = config.BET["start_bankroll"]
        for b, name in market.BOOK_NAMES.items():
            out["books"][b] = {"name": name, **market.book_stats(record["books"].get(b, []), start)}
    for r in reversed(record["graded"][-400:]):
        if r.get("void"):
            out["graded"].append({"date": r["date"], "home": r["home"], "away": r["away"], "void": True})
            continue
        pick_home = r["pick"] == r["home"]
        mp = r.get("market_p_home")
        out["graded"].append({"date": r["date"], "home": r["home"], "away": r["away"], "pick": r["pick"],
                              "p_pick": r["p_pick"], "hit": r["hit"], "score": f'{r["away_score"]}-{r["home_score"]}',
                              "ot": r["last_period"], "steal": r.get("steal", False),
                              "market_p_pick": None if mp is None else round(mp if pick_home else 1 - mp, 4),
                              "profit": r.get("profit", {}).get("always"), "hash": (r.get("hash") or "")[:12]})
    if bt:
        out["backtest"] = {"overall": bt.get("overall"), "calibration": bt.get("overall_calibration"),
                           "market": bt.get("market"),
                           "folds": [{k: f.get(k) for k in ("label", "model", "baseline_home", "by_phase",
                                                            "settings", "ot_share", "xg", "market")} for f in bt.get("folds", [])]}
    return out


def model_json(model, snap, record):
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "version": config.MODEL_VERSION, "project": config.PROJECT_NAME}
    if model:
        out.update({"settings": model["settings"], "trained_on": [config.season_label(s) for s in model["train_seasons"]],
                    "n_train_games": model["n_train_games"], "cv": model["cv"], "blend": model["blend"],
                    "platt": model["platt"], "tuning": model["tuning"][:8], "online_updates": model.get("online_updates", 0),
                    "weights": sorted([{"feature": f, "label": features.FEATURE_INFO.get(f, f), "weight": round(w, 4),
                                        "offline": round(w0, 4)} for f, w, w0 in zip(model["features"], model["coef"], model["coef0"])],
                                      key=lambda d: -abs(d["offline"]))})
    xg = _state("xg_model.json", None)
    if xg:
        pairs = sorted(zip(xg["features"], xg["coef"]), key=lambda t: -abs(t[1]))[:10]
        out["xg"] = {"report": xg.get("report"), "n_shots": xg.get("n_shots"),
                     "top": [{"feature": f, "weight": round(w, 3)} for f, w in pairs]}
    if snap:
        st = ratings.from_json(snap["kalman"])
        out["power"] = [dict(r, name=teams.display_name(r["team"])) for r in ratings.power_table(st)]
        book = live.gbook_from_json(snap["goalies"])
        current = {teams.current_franchise(t) for t in teams.TEAMS}
        rows = []
        for gid, r in book.g.items():
            if r.get("n_app", 0) < 15 or teams.current_franchise(r.get("team", "")) not in current:
                continue
            if not r.get("last_date") or r["last_date"] < f"{int(str(config.LIVE_SEASON)[:4]) - 1}-07-01":
                continue      # only goalies who played last season or this one
            c = book.components(gid)
            rows.append({"id": gid, "name": r["name"], "team": r["team"], "talent": round(c["talent"], 3),
                         "form": round(c["form"], 3), "t": round(c["t"], 2), "gate": round(c["gate"], 2)})
        rows.sort(key=lambda d: -d["talent"])
        out["goalies"] = rows[:40]
        out["snapshot_through"] = snap.get("through")
    out["learning_log"] = list(reversed(record.get("learning_log", [])))[:30]
    return out


def _write_if_changed(path, obj):
    """Skip the write when only the timestamp would change (keeps the git history quiet)."""
    new = live.clean(obj)
    if path.exists():
        old = live.read_json(path, {})
        if {k: v for k, v in old.items() if k != "generated_at"} == \
                json.loads(json.dumps({k: v for k, v in new.items() if k != "generated_at"}, default=str)):
            return False
    live.write_json(path, new)
    return True


def run(today=None, games=None, log=print):
    today = today or live.site_date()
    record = _state("record.json", {"graded": [], "books": {}, "learning_log": []})
    model = _state("win_model.json", None)
    snap = _state("snapshot.json", None)
    pub = config.PUBLIC_DIR
    _write_if_changed(pub / "today.json", today_json(today, games))
    _write_if_changed(pub / "record.json", record_json(record, _state("backtest.json", None)))
    _write_if_changed(pub / "model.json", model_json(model, snap, record))
    health.save_health(health.load_health())
    log(f"[publish] wrote public/*.json for {today}")
