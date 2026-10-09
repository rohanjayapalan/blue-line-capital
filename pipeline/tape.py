"""
"The tape": every prediction is locked into an append-only, hash-chained ledger.

Each line = {"prev": <hash of the line before>, "body": <the prediction as text>, "hash": <sha256>}
where hash = sha256(prev + body). Changing ANY old prediction changes its hash, which breaks
every hash after it - so nobody (including us) can quietly edit history. The website re-checks
the whole chain in your browser. Git commit timestamps are a second, independent witness.
"""
import hashlib
import json

from . import config

GENESIS = "0" * 64


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))


def path(season):
    return config.TAPE_DIR / f"{config.season_label(season)}.jsonl"


def read(season):
    p = path(season)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append(season, body):
    entries = read(season)
    prev = entries[-1]["hash"] if entries else GENESIS
    b = canonical(body)
    h = hashlib.sha256((prev + b).encode("utf-8")).hexdigest()
    config.TAPE_DIR.mkdir(parents=True, exist_ok=True)
    with path(season).open("a") as f:
        f.write(json.dumps({"prev": prev, "body": b, "hash": h}, separators=(",", ":")) + "\n")
    return h


def verify(season):
    prev = GENESIS
    entries = read(season)
    for i, e in enumerate(entries):
        if e["prev"] != prev or hashlib.sha256((prev + e["body"]).encode("utf-8")).hexdigest() != e["hash"]:
            return {"ok": False, "entries": len(entries), "broken_at": i}
        prev = e["hash"]
    return {"ok": True, "entries": len(entries), "head": prev}


def locked(season):
    """game_id -> (locked prediction body, hash)"""
    return {json.loads(e["body"])["game_id"]: (json.loads(e["body"]), e["hash"]) for e in read(season)}
