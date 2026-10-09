"""
The exam (walk-forward backtest).

For each test season S:
  1. the xG model is trained only on seasons before S
  2. the win model is trained only on seasons <= S-2 (the 1.5-year rule) and its settings are
     tuned on those seasons alone
  3. season S is replayed one day at a time: predict the day, then grade it and let the
     model learn (the same online learning the live site uses)
Nothing from the future can leak in, so these numbers are what the live model should do.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config, features, goalies, market, players, ratings, store, team_stats, win_model, xg_model


def build_world(xg_seasons, upto, lineup=None, log=print, xgm=None, adjust=None, league=None):
    """Everything computed from raw tables, with xG trained on `xg_seasons` only."""
    seasons = [s for s in store.seasons_on_disk() if s <= upto]
    games = store.load("games", seasons)
    shots = store.load("shots", seasons)
    counts = store.load("counts", seasons)
    gtab = store.load("goalies", seasons)
    log(f"[world] seasons {seasons[0]}..{seasons[-1]}: {len(games)} games, {len(shots)} shots")
    if xgm is None:          # nightly runs reuse the saved xG model instead of retraining
        xgm = xg_model.train(shots[shots["season"].isin(xg_seasons)], holdout_season=max(xg_seasons))
        log(f"[world] xG model: {xgm['report']}")
    shots = xg_model.add_xg(xgm, shots)
    adjust = adjust or team_stats.score_adjust(shots[shots["season"].isin(xg_seasons)])
    rink = team_stats.rink_factors(games, counts)
    tg = features.prepare(team_stats.team_games(games, shots, counts, adjust, rink))
    league = league or features.league_means(tg[tg["season"].isin(xg_seasons)])
    home_rows = tg[tg["is_home"] == 1][["game_id", "gf_noen", "ga_noen", "xgf_all", "xga_all"]]
    kg = games.merge(home_rows, on="game_id")
    kg["date"] = pd.to_datetime(kg["date"])
    kal_pre, kal_state, kparams = None, {}, {}
    for kw in config.KALMAN_GOAL_WEIGHT_OPTIONS:
        tag = f"k{int(round(kw * 100))}"
        kg["margin"] = kw * (kg["gf_noen"] - kg["ga_noen"]) + (1 - kw) * (kg["xgf_all"] - kg["xga_all"])
        hist = kg[kg["season"].isin(xg_seasons) & (kg["neutral"] == 0)]
        kp = dict(config.KALMAN)
        kp["home_ice"] = float(hist["margin"].mean())
        kp["obs_goal_weight"] = kw
        pre, state = ratings.run(kg, kp)
        pre = pre.rename(columns={"rating_diff": f"rating_diff_{tag}", "rating_var": f"rating_var_{tag}"})
        kal_pre = pre if kal_pre is None else kal_pre.merge(pre, on="game_id")
        kal_state[tag], kparams[tag] = state, kp
    gg = goalies.goalie_games(gtab, shots, games)
    gcomp, gbook = goalies.run(gg)
    if lineup is None:
        lineup, pbook = players.run(store.load("skaters", seasons), games)
    else:
        lineup, pbook = lineup
    sched = features.schedule_context(tg)
    return {"games": games, "shots": shots, "tg": tg, "league": league, "xg": xgm, "adjust": adjust,
            "rink": rink, "kal_pre": kal_pre, "kal_state": kal_state, "kparams": kparams, "gg": gg,
            "gcomp": gcomp, "gbook": gbook, "lineup": lineup, "pbook": pbook, "sched": sched}


def tables_for(world, combos=None):
    combos = combos or [(H, c) for H in config.FADE_HALF_LIFE_OPTIONS for c in config.CARRYOVER_OPTIONS]
    return {(H, c): features.build_table(world["games"], world["tg"], H, c, world["league"],
                                         world["kal_pre"], world["gcomp"], world["lineup"], world["sched"])
            for H, c in combos}


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {}
    return {"games": int(len(y)), "accuracy": round(float(((p >= 0.5) == y).mean()), 4),
            "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4)}


def calibration(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.sum() >= 10:
            out.append({"p": round(float(p[m].mean()), 4), "won": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out


def walk_season(model, booster, table, settings, log=print):
    """Replay a season day by day with online learning. Returns one row per game."""
    rows = []
    for date, day in table.groupby("date", sort=True):
        X = features.make_X(day, settings["goalie_beta"], settings["kalman"])
        res = win_model.predict(model, booster, X, day["gp"].to_numpy())
        for i, (idx, r) in enumerate(day.iterrows()):
            rows.append({"game_id": int(r["game_id"]), "date": str(date.date()), "season": int(r["season"]),
                         "home": r["home"], "away": r["away"], "p_home": float(res["p"][i]),
                         "home_win": int(r["home_win"]), "last_period": r["last_period"], "gp": int(r["gp"])})
        for i in range(len(day)):
            win_model.online_update(model, X.iloc[i].to_numpy(), int(day["home_win"].iloc[i]))
    return pd.DataFrame(rows)


def _thin(curve, keep=160):
    if len(curve) <= keep:
        return curve
    step = len(curve) / keep
    return [curve[int(i * step)] for i in range(keep)] + [curve[-1]]


def market_section(preds):
    """Model vs closing line on the same games + the simulated betting books (needs ESPN odds)."""
    path = config.DATA_DIR / "odds_history.parquet"
    if not path.exists():
        return None
    odds = pd.read_parquet(path).dropna(subset=["home_close", "away_close"])
    rows = []
    for gid, grp in odds.groupby("game_id"):
        op = grp.dropna(subset=["home_open", "away_open"])
        rows.append({"game_id": gid,
                     "p_mkt": float(np.mean([market.shin(h, a) for h, a in zip(grp["home_close"], grp["away_close"])])),
                     "p_open": float(np.mean([market.shin(h, a) for h, a in zip(op["home_open"], op["away_open"])])) if len(op) else None,
                     "best_home": float(grp["home_close"].max()), "best_away": float(grp["away_close"].max())})
    m = preds.merge(pd.DataFrame(rows), on="game_id").sort_values(["date", "game_id"]).reset_index(drop=True)
    if m.empty:
        return None
    out = {"games": int(len(m)), "model": metrics(m["home_win"], m["p_home"]), "market": metrics(m["home_win"], m["p_mkt"]),
           "by_season": {config.season_label(s): {"games": int(len(g)), "model": metrics(g["home_win"], g["p_home"]),
                                                  "market": metrics(g["home_win"], g["p_mkt"])}
                         for s, g in m.groupby("season")}}
    start = config.BET["start_bankroll"]
    bank, hist, steals = {"always": start, "edge": start}, {b: [] for b in market.BOOK_NAMES}, []
    n, ll_model, ll_mkt = 0, 0.0, 0.0
    for date, day in m.groupby("date", sort=True):
        placed = []
        for r in day.itertuples(index=False):
            cons = {"p_home": r.p_mkt, "best_home": r.best_home, "best_away": r.best_away}
            desk = market.risk_desk(r.p_home, r.p_mkt, {"p_open": r.p_open, "explain": [], "goalie_confirmed": True,
                                                        "gp_home": r.gp, "gp_away": r.gp, "data_age_hours": 0})
            tr = market.trust(n, ll_model / n, ll_mkt / n) if n >= 30 else market.trust(n)
            placed.append((r, market.decide(r.p_home, cons, desk, bank, tr)))
        for r, dec in placed:        # bets are sized on the morning bankroll, settled at night
            for b, bet in dec["bets"].items():
                pnl = market.settle(bet, r.home_win)
                hist[b].append({"date": str(date)[:10], "profit": pnl, "stake": bet["stake"]})
                if b in bank:
                    bank[b] += pnl
            if dec["steal"]:
                steals.append(int((dec["side"] == "home") == bool(r.home_win)))
            n += 1
            ll_model -= np.log(r.p_home if r.home_win else 1 - r.p_home)
            ll_mkt -= np.log(r.p_mkt if r.home_win else 1 - r.p_mkt)
    books = {}
    for b, h in hist.items():
        st = market.book_stats(h, start)
        st["curve"] = _thin(st["curve"])
        books[b] = {"name": market.BOOK_NAMES[b], **st}
    mv = m.dropna(subset=["p_open"])
    mv = mv[(mv["p_mkt"] - mv["p_open"]).abs() >= 0.005]
    out.update({"books": books, "steals": {"count": len(steals), "hit_rate": round(float(np.mean(steals)), 4) if steals else None},
                "followed_us": round(float((np.sign(mv["p_mkt"] - mv["p_open"]) == np.sign(mv["p_home"] - mv["p_open"])).mean()), 4) if len(mv) else None})
    return out


def run(test_seasons=None, log=print):
    test_seasons = test_seasons or config.BACKTEST_SEASONS
    lineup = None
    results, preds = [], []
    for S in test_seasons:
        xg_seasons = [s for s in store.seasons_on_disk() if s < S]
        train_seasons = [s for s in xg_seasons if s <= S - 10001 * (1 + config.TRAIN_GAP)
                         and s not in config.SKIP_FOR_TRAINING]
        log(f"\n=== exam season {config.season_label(S)}: model trained on {train_seasons} ===")
        world = build_world(xg_seasons, S, lineup, log)
        lineup = (world["lineup"], world["pbook"])          # lineups don't depend on xG: reuse
        tables = tables_for(world)
        model, booster = win_model.train(tables, train_seasons, log)
        st = model["settings"]
        test = tables[(st["half_life"], st["carryover"])]
        test = test[test["season"] == S]
        home_rate = model["home_rate"]
        pr = walk_season(model, booster, test, st, log)
        m = metrics(pr["home_win"], pr["p_home"])
        base = metrics(pr["home_win"], np.full(len(pr), home_rate))
        by_phase = {ph: metrics(pr.loc[pr["gp"].map(win_model.phase_name) == ph, "home_win"],
                                pr.loc[pr["gp"].map(win_model.phase_name) == ph, "p_home"])
                    for ph in win_model.PHASES}
        log(f"[exam {S}] model {m} | always-home baseline {base}")
        results.append({"season": int(S), "label": config.season_label(S), "train_seasons": train_seasons,
                        "model": m, "baseline_home": base, "by_phase": by_phase,
                        "calibration": calibration(pr["home_win"], pr["p_home"]),
                        "settings": st, "cv": model["cv"], "xg": world["xg"]["report"],
                        "ot_share": round(float((pr["last_period"] != "REG").mean()), 4)})
        preds.append(pr)
    allp = pd.concat(preds, ignore_index=True)
    summary = {"folds": results, "overall": metrics(allp["home_win"], allp["p_home"]),
               "overall_calibration": calibration(allp["home_win"], allp["p_home"])}
    mk = market_section(allp)
    if mk:
        summary["market"] = mk
        for f in results:
            f["market"] = mk["by_season"].get(f["label"])
        log(f"[exam] vs market on {mk['games']} games: model {mk['model']} | market {mk['market']}")
    config.STATE_DIR.mkdir(exist_ok=True)
    allp.to_parquet(config.DATA_DIR / "backtest_predictions.parquet", index=False)
    (config.STATE_DIR / "backtest.json").write_text(json.dumps(summary, indent=1))
    return summary, allp
