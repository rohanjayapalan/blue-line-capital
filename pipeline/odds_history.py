"""
Historical sportsbook odds (opening + closing moneylines) from ESPN, used ONLY to grade the
backtest against the market. ESPN blocks many cloud sandboxes but answers GitHub Actions.
Output: data/odds_history.parquet - one row per game per sportsbook.
For a finished game ESPN's "current" line is frozen at puck drop, i.e. it IS the closing line.
Except for the "... - Live Odds" feeds: their close is the in-game price at the final horn (1.005 vs 41),
which leaks the result, so they are dropped here and again in backtest.market_section.
"""
import concurrent.futures as cf
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, market, nhl_api, store, teams

ET = ZoneInfo(config.TIMEZONE)


def out_path():
    return config.DATA_DIR / "odds_history.parquet"


def _dec(v):
    if isinstance(v, dict):
        v = v.get("american", v.get("value"))
    return market.to_decimal(v) if v not in (None, "") else None


def _line(side, key):
    blk = side.get(key)
    return _dec(blk.get("moneyLine")) if isinstance(blk, dict) else None


def _items(js):
    out = []
    for it in (js or {}).get("items", []):
        if "$ref" in it and "homeTeamOdds" not in it:        # some responses only link to the details
            try:
                it = nhl_api.get_json(it["$ref"].replace("http://", "https://"), source="ESPN odds") or {}
            except Exception:
                continue
        out.append(it)
    return out


def is_live_feed(provider):
    return "live" in str(provider).lower()


def event_odds(event_id):
    rows = []
    for it in _items(nhl_api.espn_odds(event_id)):
        if is_live_feed((it.get("provider") or {}).get("name", "")):
            continue
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        hc = _line(h, "close") or _line(h, "current") or _dec(h.get("moneyLine"))
        ac = _line(a, "close") or _line(a, "current") or _dec(a.get("moneyLine"))
        if hc and ac:
            rows.append({"provider": (it.get("provider") or {}).get("name", "unknown"), "home_close": hc,
                         "away_close": ac, "home_open": _line(h, "open"), "away_open": _line(a, "open")})
    return rows


def _events(day):
    try:
        js = nhl_api.espn_scoreboard(day.strftime("%Y%m%d")) or {}
    except Exception:
        return []
    out = []
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        side = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in side or "away" not in side:
            continue
        ab = lambda c: teams.ESPN_ALIASES.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
        d = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
        out.append({"event": ev["id"], "date": d, "home": ab(side["home"]), "away": ab(side["away"])})
    return out


def backfill(seasons=None, log=print):
    seasons = seasons or config.BACKTEST_SEASONS
    games = store.load("games", seasons)
    keymap = dict(zip(games["date"].astype(str) + games["home"] + games["away"], games["game_id"]))
    days = sorted(set(pd.to_datetime(games["date"]).dt.date))
    events = []
    with cf.ThreadPoolExecutor(6) as ex:
        for evs in ex.map(_events, days):
            events.extend(evs)
    matched = []
    for e in events:
        gid = None
        for delta in (0, -1, 1):
            d = (pd.Timestamp(e["date"]) + pd.Timedelta(days=delta)).date().isoformat()
            gid = keymap.get(d + e["home"] + e["away"])
            if gid:
                break
        if gid:
            matched.append((int(gid), e["event"]))
    log(f"[odds] {len(events)} ESPN events, {len(matched)} matched to NHL games")

    def work(pair):
        try:
            return [dict(r, game_id=pair[0]) for r in event_odds(pair[1])]
        except Exception:
            return []

    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(work, matched):
            rows.extend(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ESPN returned no odds")
    df.to_parquet(out_path(), index=False)
    log(f"[odds] saved odds for {df['game_id'].nunique()} games")
    return df
