"""
Expected goals (xG): the chance a shot becomes a goal, judged only by WHERE and HOW it was
taken - not by who took it or whether it went in. Adding up xG tells you who created the
better chances, which predicts future results better than goals do (goals are noisy).

Model: logistic regression. For each shot:
    xG = 1 / (1 + e^-(b0 + b1*x1 + b2*x2 + ...))
where the x's are distance, angle, shot type, rebound, rush, power play, etc.
Blocked shots are left out (their coordinates are where the BLOCK happened).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]
FEATURES = (["dist", "log_dist", "angle", "angle_sq", "dist_x_angle", "behind_net"]
            + [f"type_{t}" for t in SHOT_TYPES]
            + ["rebound", "rebound_x_angle", "rush", "after_turnover", "after_faceoff",
               "secs_prev", "speed", "pp", "sh", "even_other", "empty_net", "extra_attacker",
               "score_lead", "score_trail", "ot"])


def design(shots):
    """Turn a shots table into the number grid (matrix) the model reads."""
    f = lambda c: shots[c].to_numpy(float)
    d = np.clip(np.nan_to_num(f("dist"), nan=35.0), 1.0, 200.0)
    a = np.nan_to_num(f("angle"), nan=30.0)
    x = np.nan_to_num(f("x"), nan=60.0)
    skf, ska, en, pulled, sd = f("sk_for"), f("sk_against"), f("empty_net"), f("own_goalie_pulled"), f("score_diff")
    reb = f("rebound")
    st = shots["shot_type"].to_numpy()
    cols = {
        "dist": d / 10, "log_dist": np.log(d), "angle": a / 10, "angle_sq": (a / 90) ** 2,
        "dist_x_angle": d * a / 1000, "behind_net": (x > 89).astype(float),
        "rebound": reb, "rebound_x_angle": reb * a / 90,
        "rush": f("rush"), "after_turnover": f("after_turnover"), "after_faceoff": f("after_faceoff"),
        "secs_prev": np.minimum(f("secs_prev"), 60) / 60, "speed": f("speed") / 60,
        "pp": ((skf > ska) & (en == 0) & (pulled == 0)).astype(float),
        "sh": ((skf < ska) & (en == 0) & (pulled == 0)).astype(float),
        "even_other": ((skf == ska) & (skf < 5)).astype(float),
        "empty_net": en, "extra_attacker": pulled,
        "score_lead": np.clip(sd, 0, 3), "score_trail": np.clip(-sd, 0, 3), "ot": f("ot"),
    }
    for t in SHOT_TYPES:
        cols[f"type_{t}"] = (st == t).astype(float)
    return np.column_stack([cols[c] for c in FEATURES])


def usable(shots):
    """Unblocked shots (goals, saves, misses) that have a location."""
    return shots[(shots["kind"] != "block") & shots["dist"].notna()]


def _fit(s):
    X, y = design(s), s["is_goal"].to_numpy()
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    lr = LogisticRegression(C=1.0, max_iter=3000)
    lr.fit((X - mean) / scale, y)
    return {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def _predict(m, s):
    z = ((design(s) - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"]) + m["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def train(shots, holdout_season=None):
    """Fit the xG model. If holdout_season is given, first grade it on that unseen season."""
    s = usable(shots)
    report = {}
    if holdout_season is not None and holdout_season in set(s["season"]):
        tr, te = s[s["season"] != holdout_season], s[s["season"] == holdout_season]
        p = _predict(_fit(tr), te)
        y = te["is_goal"].to_numpy()
        report = {"holdout_season": int(holdout_season), "log_loss": round(float(log_loss(y, p)), 4),
                  "auc": round(float(roc_auc_score(y, p)), 4), "goals": int(y.sum()),
                  "xg": round(float(p.sum()), 1), "shots": int(len(y))}
    m = _fit(s)
    m["report"] = report
    m["n_shots"] = int(len(s))
    m["seasons"] = sorted(int(x) for x in set(s["season"]))
    return m


def add_xg(model, shots):
    """Adds an 'xg' column. Blocked shots get 0 (they never reached the net)."""
    xg = np.zeros(len(shots), dtype="float32")
    mask = (shots["kind"] != "block").to_numpy()
    if mask.any():
        xg[mask] = _predict(model, shots[mask])
    out = shots.copy()
    out["xg"] = xg
    return out


def top_weights(model, n=8):
    pairs = sorted(zip(model["features"], model["coef"]), key=lambda p: -abs(p[1]))
    return [{"feature": f, "weight": round(w, 3)} for f, w in pairs[:n]]
