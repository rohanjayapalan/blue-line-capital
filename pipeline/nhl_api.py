"""
Everything that talks to the internet lives here, so if a website changes,
this is the only file to fix.

Sources (all free):
  NHL web API   api-web.nhle.com     schedule + live odds, boxscores, play-by-play, rosters, EDGE
  History mirror (GitHub)            exact copies of old NHL play-by-play + boxscores (fast bulk download)
  Daily Faceoff                      tonight's starting goalies + projected lines
  ESPN                               closing moneylines for old games (backtest only)
  The Odds API (optional key)        a sharper consensus line (adds Pinnacle)
"""
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from . import config
from .health import DataFormatError, SourceDownError, check_boxscore, check_pbp, check_schedule

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json,text/html"})


def get(url, params=None, retries=4, timeout=25, source="web"):
    """GET with retries. Waits 1s, 2s, 4s... between tries (exponential backoff)."""
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None                      # "doesn't exist" is an answer, not an error
            last = f"HTTP {r.status_code}"
            if r.status_code == 429:             # rate limited: wait longer
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(2 ** attempt)
    raise SourceDownError(source, f"{url} failed after {retries} tries ({last})")


def get_json(url, params=None, source="web", **kw):
    r = get(url, params=params, source=source, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        raise DataFormatError(source, f"{url} did not return JSON")


# ======================================================================= NHL
def schedule(date_str):
    """One week of games starting at date_str (YYYY-MM-DD). Includes live odds."""
    js = get_json(f"{config.NHL_API}/schedule/{date_str}", source="NHL schedule")
    if js is None:
        raise DataFormatError("NHL schedule", f"no schedule for {date_str}")
    check_schedule(js)
    return js


def games_on(date_str):
    """Just the regular-season games on one date."""
    js = schedule(date_str)
    for day in js["gameWeek"]:
        if day["date"] == date_str:
            return [g for g in day.get("games", []) if g.get("gameType") == 2]
    return []


def pbp(game_id):
    js = get_json(f"{config.NHL_API}/gamecenter/{game_id}/play-by-play", source="NHL play-by-play")
    if js is not None:
        check_pbp(js)
    return js


def boxscore(game_id):
    js = get_json(f"{config.NHL_API}/gamecenter/{game_id}/boxscore", source="NHL boxscore")
    if js is not None:
        check_boxscore(js)
    return js


def roster(team):
    return get_json(f"{config.NHL_API}/roster/{team}/current", source="NHL roster")


def edge_team(team_id):
    """NHL EDGE puck/player tracking for one team, season to date."""
    return get_json(f"{config.NHL_API}/edge/team-detail/{team_id}/now", source="NHL EDGE", retries=2)


# ======================================================================= history
def mirror_game(game_id):
    """
    The sportsdataverse project keeps exact copies of NHL API responses on GitHub.
    For bulk history this is faster and kinder than hitting the NHL 20,000 times.
    """
    js = get_json(f"{config.MIRROR}/{game_id}.json", source="history mirror", retries=3, timeout=60)
    if not js or not js.get("pbp_raw") or not js.get("boxscore_raw"):
        return None
    return js["pbp_raw"], js["boxscore_raw"]


def raw_path(game_id):
    season = int(str(game_id)[:4])
    return config.RAW_DIR / f"{season}{season + 1}" / f"{game_id}.json.gz"


def save_raw(game_id, pbp_js, box_js):
    p = raw_path(game_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"pbp": pbp_js, "box": box_js}, separators=(",", ":")).encode()
    p.write_bytes(gzip.compress(blob, compresslevel=6))


def load_raw(game_id):
    p = raw_path(game_id)
    if not p.exists():
        return None
    d = json.loads(gzip.decompress(p.read_bytes()))
    return d["pbp"], d["box"]


def fetch_game(game_id, prefer="nhl"):
    """Get (pbp, boxscore) for a finished game, from cache, the NHL, or the mirror."""
    cached = load_raw(game_id)
    if cached:
        return cached
    order = ["nhl", "mirror"] if prefer == "nhl" else ["mirror", "nhl"]
    for src in order:
        try:
            if src == "nhl":
                p, b = pbp(game_id), boxscore(game_id)
                got = (p, b) if p and b else None
            else:
                got = mirror_game(game_id)
        except SourceDownError:
            got = None
        if got and got[1].get("gameState") in ("OFF", "FINAL"):
            save_raw(game_id, *got)
            return got
    return None


def season_game_ids(season):
    """Regular-season game ids are numbered 0001, 0002, ... in order."""
    start_year = str(season)[:4]
    n = config.GAMES_IN_SEASON[season]
    return [int(f"{start_year}02{i:04d}") for i in range(1, n + 1)]


def download_season(season, prefer="mirror", threads=config.DOWNLOAD_THREADS):
    """Download every game of a season into data/raw/ (skips ones we already have)."""
    todo = [g for g in season_game_ids(season) if not raw_path(g).exists()]
    print(f"[download] {season}: {len(todo)} games to fetch")
    missing = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(fetch_game, g, prefer): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            g = futures[fut]
            try:
                if fut.result() is None:
                    missing.append(g)
            except Exception as e:   # one bad game shouldn't stop the season
                print(f"[download] {g} failed: {e}")
                missing.append(g)
            if i % 200 == 0:
                print(f"[download] {season}: {i}/{len(todo)}")
    if missing:
        print(f"[download] {season}: {len(missing)} games unavailable: {missing[:10]}...")
    return missing


# ======================================================================= Daily Faceoff
def _next_data(html, source):
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise DataFormatError(source, "page no longer has a __NEXT_DATA__ block")
    return json.loads(m.group(1))


def dfo_starting_goalies(date_str=None):
    """
    Returns a list of dicts: home/away team slug, goalie name, and news strength
    (Confirmed / Likely / Unconfirmed).
    """
    url = config.DFO_GOALIES + (f"/{date_str}" if date_str else "")
    r = get(url, source="Daily Faceoff goalies", retries=3)
    if r is None:
        return []
    data = _next_data(r.text, "Daily Faceoff goalies")
    rows = (data.get("props", {}).get("pageProps", {}) or {}).get("data")
    from .health import check_dfo_goalies
    check_dfo_goalies(rows)
    return rows


def dfo_lines(slug):
    """Projected even-strength lines for one team, e.g. slug='toronto-maple-leafs'."""
    r = get(config.DFO_LINES.format(slug=slug), source="Daily Faceoff lines", retries=3)
    if r is None:
        return None
    data = _next_data(r.text, "Daily Faceoff lines")
    combos = data.get("props", {}).get("pageProps", {}).get("combinations")
    if not combos or "players" not in combos:
        raise DataFormatError("Daily Faceoff lines", "missing pageProps.combinations.players")
    return combos["players"]


# ======================================================================= ESPN (backtest odds)
def espn_scoreboard(yyyymmdd):
    return get_json(config.ESPN_SCOREBOARD, params={"dates": yyyymmdd}, source="ESPN", retries=3)


def espn_odds(event_id):
    return get_json(config.ESPN_ODDS.format(event=event_id), source="ESPN odds", retries=3)


# ======================================================================= The Odds API (optional)
def odds_api(api_key):
    """Costs 1 credit per call on the free plan (500/month). Pinnacle lives in region 'eu'."""
    return get_json(config.ODDS_API, source="The Odds API", retries=2,
                    params={"apiKey": api_key, "regions": "us,eu", "markets": "h2h",
                            "oddsFormat": "decimal", "bookmakers": "pinnacle,draftkings,fanduel"})
