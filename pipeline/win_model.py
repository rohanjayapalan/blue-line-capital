"""
The win model: turns the features into P(home team wins), overtime and shootouts included.

Two models vote, then get calibrated:
  1. Logistic regression - a weighted sum of the features pushed through the S-curve.
     Easy to explain: each weight says how much that feature moves the odds.
  2. XGBoost - hundreds of tiny decision trees (depth 2) that catch interactions the
     straight line misses (e.g. rest matters more for a tired goalie).
  Blend in log-odds space:  z = w * z_logistic + (1 - w) * z_xgboost   (w is tuned)

Calibration (Platt scaling), separately for early / mid / late season:
  p = 1 / (1 + e^-(a*z + b))
If early-season predictions are over-confident, the data gives a < 1 and squeezes them toward
50%. That is how the model "starts less confident and works its way up" - measured, not guessed.

Online learning ("learns from its mistakes"): after every graded game the logistic weights take
one small step against the error, but are pulled back toward their offline values so one weird
night can't wreck them:
  w <- w - lr * ((p - y) * x + anchor * (w - w_offline))
"""
import json

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from . import config
from .features import FEATURES, make_X

XGB_FEATURES = [f for f in FEATURES if f not in config.LIVE_ONLY_FEATURES]
PHASES = [name for _, _, name in config.WIN_MODEL["phases"]]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def phase_name(gp):
    for lo, hi, name in config.WIN_MODEL["phases"]:
        if lo <= gp < hi:
            return name
    return PHASES[-1]


def _ll(y, z):
    return float(log_loss(y, sigmoid(z), labels=[0, 1]))


def fit_lr(X, y, C):
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    lr = LogisticRegression(C=C, max_iter=3000).fit((X - mean) / scale, y)
    return {"mean": mean, "scale": scale, "coef": lr.coef_[0].copy(), "intercept": float(lr.intercept_[0])}


def lr_logit(m, X):
    return ((X - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"]


def fit_xgb(X, y, params):
    p = {"objective": "binary:logistic", "eta": params["learning_rate"], "max_depth": params["max_depth"],
         "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
         "min_child_weight": params["min_child_weight"], "lambda": params["reg_lambda"],
         "eval_metric": "logloss", "nthread": 2, "seed": 7}
    return xgb.train(p, xgb.DMatrix(X, label=y, feature_names=XGB_FEATURES),
                     num_boost_round=params["n_estimators"])


def xgb_logit(booster, X):
    return booster.predict(xgb.DMatrix(X, feature_names=XGB_FEATURES), output_margin=True)


def oof(X, y, seasons, C, xgb_params=None):
    """Leave-one-season-out predictions: every game is predicted by a model that never saw its season."""
    z_lr, z_xgb = np.zeros(len(y)), np.zeros(len(y))
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        z_lr[te] = lr_logit(fit_lr(X[tr].to_numpy(), y[tr], C), X[te].to_numpy())
        if xgb_params is not None:
            b = fit_xgb(X.loc[tr, XGB_FEATURES].to_numpy(), y[tr], xgb_params)
            z_xgb[te] = xgb_logit(b, X.loc[te, XGB_FEATURES].to_numpy())
    return z_lr, z_xgb


def train(tables, train_seasons, log=print):
    """
    tables: {(half_life, carryover): game table}. Tries every setting and keeps the one with the
    best leave-one-season-out log loss on the TRAINING seasons only.
    """
    wm = config.WIN_MODEL
    tried, best = [], None
    for (H, carry), g in tables.items():
        tr = g[g["season"].isin(train_seasons)].reset_index(drop=True)
        y, seasons = tr["home_win"].to_numpy(), tr["season"].to_numpy()
        ktags = sorted(c[len("rating_diff_"):] for c in g.columns if c.startswith("rating_diff_"))
        for ktag in ktags:
            for beta in config.GOALIE["form_weight_options"]:
                X = make_X(tr, beta, ktag)
                for C in wm["C_options"]:
                    score = _ll(y, oof(X, y, seasons, C)[0])
                    tried.append({"half_life": H, "carryover": carry, "kalman": ktag, "goalie_beta": beta,
                                  "C": C, "log_loss": round(score, 5)})
                    if best is None or score < best[0]:
                        best = (score, H, carry, beta, C, ktag)
    _, H, carry, beta, C, ktag = best
    log(f"[train] best settings: half-life {H}, carry-over {carry}, Kalman {ktag}, goalie beta {beta}, C {C}")
    tr = tables[(H, carry)]
    tr = tr[tr["season"].isin(train_seasons)].reset_index(drop=True)
    y, seasons = tr["home_win"].to_numpy(), tr["season"].to_numpy()
    X = make_X(tr, beta, ktag)
    z_lr, z_xgb = oof(X, y, seasons, C, wm["xgb_params"])
    blend = min(wm["blend_options"], key=lambda w: _ll(y, w * z_lr + (1 - w) * z_xgb))
    z = blend * z_lr + (1 - blend) * z_xgb
    phases = np.array([phase_name(v) for v in tr["gp"]])
    platt, z_cal = {}, z.copy()
    for name in PHASES:
        m = phases == name
        if m.sum() >= 150:
            cal = LogisticRegression(C=1e6, max_iter=1000).fit(z[m].reshape(-1, 1), y[m])
            platt[name] = [float(cal.coef_[0][0]), float(cal.intercept_[0])]
        else:
            platt[name] = [1.0, 0.0]
        z_cal[m] = platt[name][0] * z[m] + platt[name][1]
    lr = fit_lr(X.to_numpy(), y, C)
    booster = fit_xgb(X[XGB_FEATURES].to_numpy(), y, wm["xgb_params"])
    model = {
        "version": config.MODEL_VERSION, "features": FEATURES, "xgb_features": XGB_FEATURES,
        "mean": lr["mean"], "scale": lr["scale"], "coef": lr["coef"], "intercept": lr["intercept"],
        "coef0": lr["coef"].copy(), "intercept0": lr["intercept"], "blend": float(blend), "platt": platt,
        "settings": {"half_life": H, "carryover": carry, "kalman": ktag, "goalie_beta": beta, "C": C},
        "train_seasons": [int(s) for s in train_seasons], "n_train_games": int(len(y)),
        "home_rate": float(y.mean()),
        "cv": {"logistic": round(_ll(y, z_lr), 5), "xgboost": round(_ll(y, z_xgb), 5),
               "blend": round(_ll(y, z), 5), "calibrated": round(_ll(y, z_cal), 5),
               "accuracy": round(float(((z_cal > 0) == y).mean()), 4)},
        "tuning": sorted(tried, key=lambda r: r["log_loss"])[:12],
        "online_updates": 0,
    }
    log(f"[train] CV log loss: logistic {model['cv']['logistic']}, xgboost {model['cv']['xgboost']}, "
        f"blend (w={blend}) {model['cv']['blend']}, calibrated {model['cv']['calibrated']}")
    return model, booster


def predict(model, booster, X, gp):
    """Probabilities + an explanation: how much each feature pushed the log-odds."""
    Xn = X[FEATURES].to_numpy(float)
    zs = (Xn - model["mean"]) / model["scale"]
    lr_c = zs * model["coef"]
    z_lr = lr_c.sum(1) + model["intercept"]
    xc = booster.predict(xgb.DMatrix(X[XGB_FEATURES].to_numpy(float), feature_names=XGB_FEATURES),
                         pred_contribs=True)
    z_xgb = xc.sum(1)
    w = model["blend"]
    z = w * z_lr + (1 - w) * z_xgb
    ab = np.array([model["platt"][phase_name(v)] for v in np.atleast_1d(gp)])
    a, b = ab[:, 0], ab[:, 1]
    z_cal = a * z + b
    contrib = w * lr_c
    idx = [FEATURES.index(f) for f in XGB_FEATURES]
    contrib[:, idx] += (1 - w) * xc[:, :-1]
    contrib *= a[:, None]
    base = a * (w * model["intercept"] + (1 - w) * xc[:, -1]) + b
    return {"p": sigmoid(z_cal), "z": z_cal, "base": base, "contrib": contrib, "z_lr": z_lr, "z_xgb": z_xgb}


def explain(result, i, X_row_raw, top=6):
    """Turn log-odds pushes into percentage points (linearised around the midpoint)."""
    p = float(result["p"][i])
    base = float(result["base"][i])
    mid = sigmoid((base + float(result["z"][i])) / 2)
    slope = mid * (1 - mid)
    items = []
    for j, f in enumerate(FEATURES):
        c = float(result["contrib"][i, j])
        items.append({"feature": f, "logit": round(c, 4), "pp": round(100 * c * slope, 2),
                      "value": round(float(X_row_raw[f]), 4)})
    items.sort(key=lambda d: -abs(d["logit"]))
    return {"p_base": round(float(sigmoid(base)), 4), "p": round(p, 4), "items": items[:top],
            "all": items}


def online_update(model, x_row, y, rate=None, anchor=None):
    """One small learning step after a graded game (logistic part only)."""
    rate = rate if rate is not None else config.WIN_MODEL["online_lr"]
    anchor = anchor if anchor is not None else config.WIN_MODEL["online_anchor"]
    zs = (np.asarray(x_row, float) - model["mean"]) / model["scale"]
    p = float(sigmoid(zs @ model["coef"] + model["intercept"]))
    err = p - y
    model["coef"] = model["coef"] - rate * (err * zs + anchor * (model["coef"] - model["coef0"]))
    model["intercept"] = model["intercept"] - rate * (err + anchor * (model["intercept"] - model["intercept0"]))
    model["online_updates"] = model.get("online_updates", 0) + 1
    return p


def to_json(model):
    out = {}
    for k, v in model.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def from_json(d):
    m = dict(d)
    for k in ("mean", "scale", "coef", "coef0"):
        m[k] = np.array(d[k], float)
    return m


def save(model, booster, folder=None):
    folder = folder or config.STATE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "win_model.json").write_text(json.dumps(to_json(model), indent=1))
    booster.save_model(str(folder / "win_xgb.json"))


def load(folder=None):
    folder = folder or config.STATE_DIR
    model = from_json(json.loads((folder / "win_model.json").read_text()))
    booster = xgb.Booster()
    booster.load_model(str(folder / "win_xgb.json"))
    return model, booster
