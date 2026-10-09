"""
Saving and loading our tables (parquet files, one folder per season).

History seasons:  data/tables/<season>/<table>.parquet            (written once)
Live season:      data/tables/<season>/<table>-<YYYY-MM>.parquet  (one file per month, so each
                  night only rewrites the current month and the git repo stays small)
Tables: games, counts (team non-shot stats), skaters, goalies, shots.
"""
import pandas as pd

from . import config, nhl_api
from .parse_game import parse_game

TABLES = ["games", "counts", "skaters", "goalies", "shots"]
TABLE_DIR = config.DATA_DIR / "tables"

SHOT_TYPES = {"period": "int8", "t": "int16", "is_home": "int8", "is_goal": "int8", "rebound": "int8",
              "rush": "int8", "after_turnover": "int8", "after_faceoff": "int8", "secs_prev": "int16",
              "sk_for": "int8", "sk_against": "int8", "empty_net": "int8", "own_goalie_pulled": "int8",
              "score_diff": "int8", "ot": "int8", "x": "float32", "y": "float32", "dist": "float32",
              "angle": "float32", "speed": "float32"}


def files(season, table):
    d = TABLE_DIR / str(season)
    if not d.exists():
        return []
    return sorted(d.glob(f"{table}.parquet")) + sorted(d.glob(f"{table}-*.parquet"))


def has(season, table):
    return bool(files(season, table))


def load(table, seasons):
    frames = [pd.read_parquet(f) for s in seasons for f in files(s, table)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def seasons_on_disk():
    if not TABLE_DIR.exists():
        return []
    return sorted(int(p.name) for p in TABLE_DIR.iterdir() if p.name.isdigit() and has(int(p.name), "games"))


def _frames_from_parsed(parsed):
    rows = {t: [] for t in TABLES}
    for g in parsed:
        rows["games"].append(g["game"])
        for t in ("shots", "counts", "skaters", "goalies"):
            rows[t].extend(g[t])
    frames = {t: pd.DataFrame(v) for t, v in rows.items()}
    if len(frames["shots"]):
        frames["shots"] = frames["shots"].astype(SHOT_TYPES)
    return frames


def write(season, frames, append=False):
    d = TABLE_DIR / str(season)
    d.mkdir(parents=True, exist_ok=True)
    if not append:
        for t, df in frames.items():
            df.sort_values("game_id", kind="stable").reset_index(drop=True).to_parquet(d / f"{t}.parquet", index=False)
        return
    month = dict(zip(frames["games"]["game_id"], frames["games"]["date"].astype(str).str[:7]))
    for t, df in frames.items():
        if not len(df):
            continue
        for mon, part in df.groupby(df["game_id"].map(month)):
            p = d / f"{t}-{mon}.parquet"
            if p.exists():
                old = pd.read_parquet(p)
                part = pd.concat([old[~old["game_id"].isin(set(part["game_id"]))], part], ignore_index=True)
            part.sort_values("game_id", kind="stable").reset_index(drop=True).to_parquet(p, index=False)


def build_season(season):
    """Parse every raw game of a season (from data/raw) into the five tables."""
    parsed, bad = [], []
    for gid in nhl_api.season_game_ids(season):
        raw = nhl_api.load_raw(gid)
        if raw is None:
            continue
        try:
            parsed.append(parse_game(*raw))
        except Exception as e:  # a single broken file should not stop the build
            bad.append((gid, repr(e)))
    if bad:
        print(f"[store] {season}: {len(bad)} games failed to parse, e.g. {bad[:3]}")
    write(season, _frames_from_parsed(parsed))
    print(f"[store] {season}: {len(parsed)} games saved")
    return len(parsed)


def ingest(game_ids, season):
    """Add newly finished games (live season). Returns how many were added."""
    parsed = []
    for gid in game_ids:
        raw = nhl_api.fetch_game(gid, prefer="nhl")
        if raw is not None:
            parsed.append(parse_game(*raw))
    if parsed:
        write(season, _frames_from_parsed(parsed), append=True)
    return len(parsed)
