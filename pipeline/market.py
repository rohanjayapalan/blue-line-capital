"""
The trading desk: sportsbook odds -> fair probabilities -> bets -> profit and loss.

1. Odds to probability. Decimal odds 2.50 imply 1/2.50 = 40%. Both sides of a game add up to
   more than 100% (about 104%) - that extra is the bookmaker's cut ("vig").
2. Remove the vig with Shin's method, which assumes part of the money comes from insiders and
   so removes a bit more from longshots than from favourites (closer to reality than dividing
   evenly).
3. Our betting probability is ANCHORED to the market, then pulled toward our model:
       logit(p_bet) = logit(p_market) + alpha * (logit(p_model) - logit(p_market))
   alpha = 0.5 normally, 0.8 for a checked "steal", 0.2 when the risk desk has doubts.
4. Stake with fractional Kelly: kelly = (p * odds - 1) / (odds - 1) is the growth-optimal
   fraction of bankroll; we bet a quarter of it (half for a steal), capped.
"""
import math

import numpy as np

from . import config


def to_decimal(value):
    """'+150' -> 2.5, '-120' -> 1.833, '2.05' -> 2.05."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("EVEN", "EV"):
        return 2.0
    try:
        v = float(s)
    except ValueError:
        return None
    if s.startswith(("+", "-")) or abs(v) >= 100:
        if v >= 100:
            return 1 + v / 100
        if v <= -100:
            return 1 + 100 / abs(v)
        return None
    return v if v > 1.0 else None


def to_american(dec):
    if not dec or dec <= 1:
        return ""
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"-{round(100 / (dec - 1))}"


def shin(dec_home, dec_away):
    """Vig-free probability of the HOME side (Shin 1993, two outcomes, solved by bisection)."""
    pi = [1 / dec_home, 1 / dec_away]
    S = sum(pi)
    if S <= 1:
        return pi[0] / S

    def probs(z):
        return [(math.sqrt(z * z + 4 * (1 - z) * p * p / S) - z) / (2 * (1 - z)) for p in pi]

    lo, hi = 0.0, 0.4
    for _ in range(60):
        z = (lo + hi) / 2
        if sum(probs(z)) > 1:
            lo = z
        else:
            hi = z
    p = probs((lo + hi) / 2)
    return p[0] / sum(p)


def consensus(quotes):
    """
    quotes: list of {"book": name, "home": decimal, "away": decimal}.
    Returns fair home probability (average of each book's Shin probability), best prices, vig.
    """
    good = [q for q in quotes if q.get("home") and q.get("away")]
    if not good:
        return None
    probs = [shin(q["home"], q["away"]) for q in good]
    vig = [1 / q["home"] + 1 / q["away"] - 1 for q in good]
    best_home = max(good, key=lambda q: q["home"])
    best_away = max(good, key=lambda q: q["away"])
    return {"p_home": float(np.mean(probs)), "books": [q["book"] for q in good],
            "best_home": best_home["home"], "best_home_book": best_home["book"],
            "best_away": best_away["away"], "best_away_book": best_away["book"],
            "vig": round(float(np.mean(vig)), 4), "quotes": good}


def quotes_from_schedule(game):
    """The NHL schedule lists odds per team per provider; pair them up by provider."""
    h = {o.get("providerId"): to_decimal(o.get("value")) for o in game["homeTeam"].get("odds", []) or []}
    a = {o.get("providerId"): to_decimal(o.get("value")) for o in game["awayTeam"].get("odds", []) or []}
    out = []
    for pid in sorted(set(h) & set(a)):
        if pid in config.PREFERRED_BOOKS and h[pid] and a[pid]:
            out.append({"book": config.BOOKS.get(pid, str(pid)), "home": h[pid], "away": a[pid]})
    if not out:   # fall back to any provider the NHL lists
        for pid in sorted(set(h) & set(a)):
            if h[pid] and a[pid]:
                out.append({"book": config.BOOKS.get(pid, str(pid)), "home": h[pid], "away": a[pid]})
    return out


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def blend(p_model, p_mkt, alpha):
    return sigmoid(logit(p_mkt) + alpha * (logit(p_model) - logit(p_mkt)))


SITUATIONAL = {"goalie", "lineup", "star_missing", "rest", "b2b_home", "b2b_away", "travel_away", "tz_away"}


def risk_desk(p_model, p_mkt, ctx):
    """
    Runs only when the model and market disagree by 7+ percentage points.
    ctx: p_open (first market price we saw), explain items (pp per feature), goalie_confirmed
    (both starters confirmed?), gp_home, gp_away, data_age_hours.
    """
    gap = p_model - p_mkt
    if abs(gap) < config.BET["steal_gap"]:
        return {"triggered": False, "passed": False, "checks": []}
    rd, checks = config.RISK_DESK, []
    p_open = ctx.get("p_open")
    moved = 0.0 if p_open is None else (p_mkt - p_open) * (1 if gap > 0 else -1)
    checks.append({"name": "Line movement", "ok": moved > -rd["line_move_against"],
                   "detail": f"market moved {moved * 100:+.1f} pp toward us since open"})
    items = {d["feature"]: d["pp"] / 100 for d in ctx.get("explain", [])}
    goalie_share = items.get("goalie", 0.0) / gap if gap else 0.0
    checks.append({"name": "Goalie dependence",
                   "ok": not (goalie_share > rd["goalie_share"] and not ctx.get("goalie_confirmed", False)),
                   "detail": f"goalie explains {goalie_share * 100:.0f}% of the gap; "
                             f"starters {'confirmed' if ctx.get('goalie_confirmed') else 'not confirmed'}"})
    sit = max([items.get(f, 0.0) / gap for f in SITUATIONAL] + [0.0])
    checks.append({"name": "One-factor thesis", "ok": sit <= rd["single_feature_share"],
                   "detail": f"largest situational factor explains {sit * 100:.0f}% of the gap"})
    gp = min(ctx.get("gp_home", 99), ctx.get("gp_away", 99))
    checks.append({"name": "Sample size", "ok": gp >= rd["min_games"], "detail": f"fewest games played: {gp}"})
    age = ctx.get("data_age_hours", 0.0)
    checks.append({"name": "Fresh data", "ok": age <= rd["stale_hours"], "detail": f"data is {age:.0f} h old"})
    return {"triggered": True, "passed": all(c["ok"] for c in checks), "checks": checks}


def trust(n_graded, ll_model=None, ll_market=None):
    """Bet sizes start at 50% and grow to 100% as the live record (vs the market) earns it."""
    b = config.BET
    perf = 1.0
    if ll_model is not None and ll_market is not None:
        perf = min(1.0, max(0.0, 1 + 50 * (ll_market - ll_model)))
    ramp = min(1.0, n_graded / b["trust_full_after"])
    return b["trust_start"] + (1 - b["trust_start"]) * ramp * perf


def decide(p_model, cons, desk, bankrolls, trust_level):
    """
    Decide every book's bet for one game. Returns a dict of bets:
      always   - bets every game (at least 0.25% of bankroll), Kelly-sized
      edge     - bets only when expected value > +1%
      fav/home/pick - $100 flat benchmarks
    """
    b = config.BET
    steal = desk["triggered"] and desk["passed"]
    alpha = b["alpha_steal"] if steal else (b["alpha_failed_check"] if desk["triggered"] else b["alpha_normal"])
    p_mkt = cons["p_home"]
    p_bet = blend(p_model, p_mkt, alpha)
    sides = {"home": (p_bet, cons["best_home"]), "away": (1 - p_bet, cons["best_away"])}
    ev = {s: p * d - 1 for s, (p, d) in sides.items()}
    side = max(ev, key=ev.get)
    p, dec = sides[side]
    kelly = max(0.0, (p * dec - 1) / (dec - 1))
    kf, cap = (b["steal_kelly_fraction"], b["steal_cap"]) if steal else (b["kelly_fraction"], b["cap"])
    frac = min(cap, kf * trust_level * kelly)
    bets = {}
    always = max(b["min_stake"], frac)
    bets["always"] = {"side": side, "dec": dec, "stake": round(bankrolls["always"] * always, 2)}
    if ev[side] > b["edge_only_min_ev"] and frac > 0:
        bets["edge"] = {"side": side, "dec": dec, "stake": round(bankrolls["edge"] * frac, 2)}
    fav = "home" if p_mkt >= 0.5 else "away"
    pick = "home" if p_model >= 0.5 else "away"
    price = {"home": cons["best_home"], "away": cons["best_away"]}
    bets["fav"] = {"side": fav, "dec": price[fav], "stake": b["flat_stake"]}
    bets["home"] = {"side": "home", "dec": price["home"], "stake": b["flat_stake"]}
    bets["pick"] = {"side": pick, "dec": price[pick], "stake": b["flat_stake"]}
    return {"p_bet": round(p_bet, 4), "alpha": alpha, "steal": steal, "side": side,
            "ev": round(ev[side], 4), "kelly": round(kelly, 4), "bets": bets}


def settle(bet, home_won):
    won = (bet["side"] == "home") == bool(home_won)
    return round(bet["stake"] * (bet["dec"] - 1), 2) if won else -bet["stake"]


BOOK_NAMES = {"always": "Always-in (Kelly)", "edge": "Edge-only (Kelly)", "fav": "Always the favourite",
              "home": "Always the home team", "pick": "Model pick, flat $100"}


def book_stats(history, start):
    """history: list of {"date", "profit", "stake"} in time order."""
    if not history:
        return {"bankroll": start, "profit": 0.0, "bets": 0, "roi": 0.0, "max_drawdown": 0.0, "t_stat": 0.0,
                "curve": []}
    profits = np.array([h["profit"] for h in history], float)
    stakes = np.array([h["stake"] for h in history], float)
    equity = start + np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[start], equity]))[1:]
    dd = float(((peak - equity) / peak).max())
    r = profits / np.where(stakes > 0, stakes, 1)
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 2 and r.std() > 0 else 0.0
    by_date = {}
    for h, e in zip(history, equity):
        by_date[h["date"]] = round(float(e), 2)
    return {"bankroll": round(float(equity[-1]), 2), "profit": round(float(profits.sum()), 2),
            "bets": int(len(profits)), "roi": round(float(profits.sum() / stakes.sum()), 4) if stakes.sum() else 0.0,
            "max_drawdown": round(dd, 4), "t_stat": round(t, 2),
            "curve": [{"date": d, "equity": v} for d, v in by_date.items()]}
