"""
Health checks = the smoke detector.

Every time we download something, we check it still looks the way we expect.
If a website changes its format, we raise DataFormatError. run.py catches it,
writes a warning banner into public/health.json (the site shows it), opens a
GitHub issue (GitHub emails you), and fails the run (GitHub emails you again).
"""
import json
import os
from datetime import datetime, timezone

import requests

from . import config


class DataFormatError(Exception):
    """A data source answered, but not in the shape we expect."""

    def __init__(self, source, detail):
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


class SourceDownError(Exception):
    """A data source did not answer at all (after retries)."""

    def __init__(self, source, detail):
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


def require(obj, path, source):
    """
    Walk into a nested dict/list and fail loudly if something is missing.
    require(game, "homeTeam.abbrev", "schedule") returns game["homeTeam"]["abbrev"].
    """
    cur = obj
    for key in path.split("."):
        if isinstance(cur, list):
            if not cur:
                raise DataFormatError(source, f"'{path}' hit an empty list")
            cur = cur[0]
        if not isinstance(cur, dict) or key not in cur:
            raise DataFormatError(source, f"missing '{path}' (stopped at '{key}')")
        cur = cur[key]
    return cur


def check_schedule(js):
    require(js, "gameWeek", "NHL schedule")
    for day in js["gameWeek"]:
        require(day, "date", "NHL schedule")
        for g in day.get("games", []):
            for p in ["id", "gameType", "startTimeUTC", "gameState",
                      "homeTeam.abbrev", "awayTeam.abbrev", "homeTeam.id"]:
                require(g, p, "NHL schedule")


def check_pbp(js):
    for p in ["id", "season", "gameType", "homeTeam.abbrev", "awayTeam.abbrev", "plays"]:
        require(js, p, "NHL play-by-play")
    for play in js["plays"][:50]:
        for p in ["typeDescKey", "periodDescriptor.number", "timeInPeriod"]:
            require(play, p, "NHL play-by-play")
    kinds = {p.get("typeDescKey") for p in js["plays"]}
    if js.get("gameState") in ("OFF", "FINAL") and "shot-on-goal" not in kinds:
        raise DataFormatError("NHL play-by-play", "finished game has no 'shot-on-goal' events")


def check_boxscore(js):
    for p in ["id", "homeTeam.score", "awayTeam.score", "playerByGameStats.homeTeam.forwards",
              "playerByGameStats.homeTeam.defense", "playerByGameStats.homeTeam.goalies"]:
        require(js, p, "NHL boxscore")
    g = js["playerByGameStats"]["homeTeam"]["goalies"]
    if g and "starter" not in g[0]:
        raise DataFormatError("NHL boxscore", "goalies no longer have a 'starter' field")


def check_dfo_goalies(rows):
    if not isinstance(rows, list):
        raise DataFormatError("Daily Faceoff goalies", "pageProps.data is not a list")
    for r in rows[:5]:
        for k in ["homeTeamSlug", "awayTeamSlug", "homeGoalieName", "awayGoalieName"]:
            if k not in r:
                raise DataFormatError("Daily Faceoff goalies", f"missing '{k}'")


# ------------------------------------------------------------------ reporting
def _health_path():
    return config.PUBLIC_DIR / "health.json"


def load_health():
    p = _health_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"sources": {}, "runs": {}, "banner": None}


def record(health, source, ok, detail=""):
    """Remember the last good / bad time for each source."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    s = health["sources"].setdefault(source, {})
    if ok:
        if s.get("status") == "ok" and s.get("last_ok"):
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(s["last_ok"])
                if age.total_seconds() < 6 * 3600:
                    return
            except ValueError:
                pass
        s["last_ok"] = now
        s["status"] = "ok"
        s.pop("error", None)
    else:
        s["last_error"] = now
        s["status"] = "error"
        s["error"] = str(detail)[:300]


def save_health(health):
    config.PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    _health_path().write_text(json.dumps(health, indent=1, sort_keys=True))


def open_github_issue(title, body):
    """
    Opens an issue on your repo so GitHub emails you. Needs the GITHUB_TOKEN that
    Actions provides automatically (see permissions: issues: write in the workflows).
    Skips quietly when running on your laptop.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print(f"[health] (not on GitHub) would open issue: {title}")
        return
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    api = f"https://api.github.com/repos/{repo}/issues"
    try:
        existing = requests.get(api, headers=headers, params={"state": "open", "labels": "data-source"},
                                timeout=20).json()
        if isinstance(existing, list) and any(i.get("title") == title for i in existing):
            return  # already open, don't spam
        requests.post(api, headers=headers, timeout=20,
                      json={"title": title, "body": body, "labels": ["data-source"]})
    except requests.RequestException as e:
        print(f"[health] could not open issue: {e}")
