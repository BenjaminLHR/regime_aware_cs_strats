#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BOCPD features with XGBoost, SHAP, and date-balanced, class-weighted permutation tests.

For the selected window (--window or SLURM_ARRAY_TASK_ID), this script:

* loads features from the HDF (hard-coded path/key),
* builds 3-class labels per date (top/bottom deciles of risk-adjusted return),
* trains XGBoost only if booster.json is missing (sigma monotone constraint = −1),
* scores OOS class probabilities and SHAP tail-margin (class2 − class0),
* saves arrays plus an OOS SHAP summary plot/CSV,
* runs grouped within-date permutation with repeats and writes <win>/<win>_perm_importance.csv with Δ(date-balanced, class-weighted logloss),
  ΔrankIC, and ΔTop–Bottom (medians with 95% CIs).

Notes:
* training uses inverse-frequency class weights;
* evaluation uses the same scheme (date-balanced weighted logloss);
* permutation reuses the exact prediction pipeline.

Optional env vars: OUT_DIR, MODEL_DIR, TAIL_FRAC, PY_SEED, N_PERM, BATCH, NTHREADS, EVAL_WEIGHT_SOURCE.
"""


import os, sys, gc, json, argparse, pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr

LTR_HDF_PATH = "features/ltr_df.h5"
LTR_HDF_KEY  = "ltr_df"

# Output locations
OUT_BASE   = Path(os.environ.get("OUT_DIR", "./shap_results_mom_bocd")).resolve()
MODEL_BASE = Path(os.environ.get("MODEL_DIR", "./models_mom_bocd_exp")).resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)
MODEL_BASE.mkdir(parents=True, exist_ok=True)

# Features & windows
FEATURE_COLS = [
    'sigma_lag_span_63_z',
    
    # _z refers to winsorizing, then z-scoring (subtract mean return and divide by std of that day)
    # short & medium momentum/m
    '21_day_raw_ret_z',
    '63_day_raw_ret_z',

    # long momentum
    '252_day_raw_ret_z',

    # vol-and-time-adj the above
    '21_day_vol_time_scaled_ret_z',
    '63_day_vol_time_scaled_ret_z',
    '252_day_vol_time_scaled_ret_z',

    # (252, 42)_day_raw_ret are 
    '(252, 42)_day_raw_ret_z',
    '(252, 42)_day_vol_time_scaled_ret_z',

    # MACD(S, L) as per Daniel's papers
    '8_24_curr_z',
    '16_48_curr_z',
    '32_96_curr_z',

    '8_24_lag_1m_z',
    '16_48_lag_1m_z',
    '32_96_lag_1m_z',
    
    
    '8_24_lag_3m_z',
    '16_48_lag_3m_z',
    '32_96_lag_3m_z',
    
    '8_24_lag_6m_z',
    '16_48_lag_6m_z',
    '32_96_lag_6m_z',
    
    '8_24_lag_12m_z',
    '16_48_lag_12m_z',
    '32_96_lag_12m_z',

    # BOCD
    'p_t',
    'E_rt',
    'dE_rt',
    'Pr_le_5',
    'Pr_le_10',
    'Pr_le_14',
    'Pr_le_21',
    'Pr_le_42',
    'Pr_le_63',
    'Pr_le_126',
    'Pr_le_252',
    'H_rt',
    'Var_rt',
    'r_med',
    'p_t_z',
    'E_rt_z',
    'dE_rt_z',
    'Pr_le_5_z',
    'Pr_le_10_z',
    'Pr_le_14_z',
    'Pr_le_21_z',
    'Pr_le_42_z',
    'Pr_le_63_z',
    'Pr_le_126_z',
    'Pr_le_252_z',
    'H_rt_z',
    'Var_rt_z',
    'r_med_z',
]


# (train, test)
TRAIN_TEST_LSTS = [
    [("1999-01-01", "2004-11-25"), ("2004-12-01", "2009-11-30")],
    [("1999-01-01", "2009-11-25"), ("2009-12-01", "2014-11-30")],
    [("1999-01-01", "2014-11-25"), ("2014-12-01", "2019-11-30")],
    [("1999-01-01", "2019-11-25"), ("2019-12-01", "2024-12-31")],
]
WIN_LABELS = ["win0_cfgN", "win1_cfgO", "win2_cfgO", "win3_cfgO"]

# Pre-selected boosting rounds per window
ROUNDS = {"win0_cfgN": 47, "win1_cfgO": 56, "win2_cfgO": 256, "win3_cfgO": 370}

# Monotone constraints: σ non-increasing; others unconstrained
MONO = [-1 if f == "sigma_lag_span_63_z" else 0 for f in FEATURE_COLS]

# Grouped permutation sets
GROUPS: List[List[str]] = [
    ["sigma_lag_span_63_z"],                               # Risk (σ, monotone −1)
    ["E_rt","E_rt_z","r_med","r_med_z"],                   # Regime age / persistence
    ["H_rt","H_rt_z"],                                     # Ambiguity (entropy)
    ["Var_rt","Var_rt_z"],                                 # Dispersion (variance)
    ["p_t","p_t_z","dE_rt","dE_rt_z"],                     # Instant break prob & slope
    [s for k in [5,10,14] for s in (f"Pr_le_{k}", f"Pr_le_{k}_z")],     # Short horizons
    [s for k in [21,42]    for s in (f"Pr_le_{k}", f"Pr_le_{k}_z")],     # Medium horizons
    [s for k in [63,126,252] for s in (f"Pr_le_{k}", f"Pr_le_{k}_z")],   # Long horizons
    ['21_day_raw_ret_z', '21_day_vol_time_scaled_ret_z'], # Short horizon lagged returns
    ['63_day_raw_ret_z', '63_day_vol_time_scaled_ret_z'], # Medium horizon lagged returns
    ['252_day_raw_ret_z', '252_day_vol_time_scaled_ret_z', '(252, 42)_day_raw_ret_z', '(252, 42)_day_vol_time_scaled_ret_z'], # Long horizon lagged returns
    ['8_24_curr_z', '8_24_lag_1m_z', '8_24_lag_3m_z', '8_24_lag_6m_z', '8_24_lag_12m_z'], # MACD short timescale
    ['16_48_curr_z', '16_48_lag_1m_z', '16_48_lag_3m_z', '16_48_lag_6m_z', '16_48_lag_12m_z'], # MACD medium timescale
    ['32_96_curr_z', '32_96_lag_1m_z', '32_96_lag_3m_z', '32_96_lag_6m_z', '32_96_lag_12m_z'], # MACD long timescale
]


# Labels, random, plotting
RET_COL   = "fwd_ret_div_sigma_lag_63" # risk-adjusted forward 21D return
TAIL_FRAC = float(os.environ.get("TAIL_FRAC", 0.10))
RANDOM_SEED = int(os.environ.get("PY_SEED", 123))
np.random.seed(RANDOM_SEED)

# Permutation controls
N_PERM   = int(os.environ.get("N_PERM", 20))     # repeats per group
BATCH    = int(os.environ.get("BATCH", 150_000)) # streaming batch size
NTHREADS = int(os.environ.get("NTHREADS", os.cpu_count() or 8))

# Eval weighting controls
USE_CLASS_WEIGHTS   = True
EVAL_WEIGHT_SOURCE  = os.environ.get("EVAL_WEIGHT_SOURCE", "train").lower()  # 'train' or 'test'
EVAL_SMOOTHING      = 1.0  # Laplace smoothing

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def make_class3_fast(df: pd.DataFrame, date_col="date", ret_col=RET_COL, frac=TAIL_FRAC) -> pd.DataFrame:
    out = df.copy()
    r_desc = out.groupby(date_col)[ret_col].rank(method="first", ascending=False)
    r_asc  = out.groupby(date_col)[ret_col].rank(method="first", ascending=True)
    n      = out.groupby(date_col)[ret_col].transform("size").astype(int)
    k      = np.maximum(1, np.rint(frac * n).astype(int))
    out["class3"] = 1
    out.loc[r_desc <= k, "class3"] = 2
    out.loc[r_asc  <= k, "class3"] = 0
    return out

# Class-weight helpers (eval)
def inverse_frequency_weights(y: np.ndarray, smoothing: float = 1.0) -> Dict[int, float]:
    """Return inverse-frequency weights with Laplace smoothing, mean≈1 across 3 classes."""
    vc = pd.Series(y).value_counts().reindex([0, 1, 2]).fillna(0).astype(float) + smoothing
    w  = vc.sum() / (3.0 * vc)
    return w.to_dict()

def date_balanced_weighted_logloss_from_proba(
    P: np.ndarray, y: np.ndarray, dates: np.ndarray, sample_w: np.ndarray
) -> float:
    """Date-balanced *weighted* logloss: mean across dates of (sum w * nll / sum w)."""
    eps = 1e-15
    P = np.clip(P, eps, 1 - eps)
    p_y = P[np.arange(len(y)), y]
    nll = -np.log(p_y)
    df = pd.DataFrame({"date": pd.to_datetime(dates), "nll": nll, "w": sample_w.astype(float)})
    per_date = (df.groupby("date", sort=False)
                  .apply(lambda g: (g["w"] * g["nll"]).sum() / max(g["w"].sum(), 1e-12))
                  .astype(float))
    return float(per_date.mean())


# Date-balanced helpers
def net_score_from_proba(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return np.clip(P[:, 2], eps, 1.0) - np.clip(P[:, 0], eps, 1.0)

def date_balanced_rank_ic(net: np.ndarray, y_cont: np.ndarray, dates: np.ndarray) -> Tuple[float, pd.Series]:
    """Spearman per date (skip degenerate), then average."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "net": net, "y": y_cont})
    def _ic(g: pd.DataFrame):
        if g["net"].nunique() < 2 or g["y"].nunique() < 2:
            return np.nan
        return spearmanr(g["net"], g["y"]).correlation
    ics = df.groupby("date", sort=False).apply(_ic).astype(float)
    return float(np.nanmean(ics)), ics

def date_balanced_topbottom(net: np.ndarray, y_cont: np.ndarray, dates: np.ndarray, q: float = TAIL_FRAC) -> Tuple[float, pd.Series]:
    """Mean(Y_top) - Mean(Y_bot) per date by net-score deciles; then average."""
    d = pd.DataFrame({"date": pd.to_datetime(dates), "net": net, "y": y_cont})
    out = []
    for _, g in d.groupby("date", sort=False):
        n = len(g)
        if n < 5: 
            continue
        k = max(1, int(round(q * n)))
        g_sorted = g.sort_values("net")
        bot = g_sorted.head(k)["y"].mean()
        top = g_sorted.tail(k)["y"].mean()
        out.append(top - bot)
    return (float(np.mean(out)) if out else float("nan")), pd.Series(out)

def date_balanced_means(phi: np.ndarray, dates: np.ndarray):
    dts = pd.to_datetime(dates).values
    Tcodes, inv = np.unique(dts, return_inverse=True)
    T = len(Tcodes); n, p = phi.shape
    sums_abs = np.zeros((T, p), dtype=np.float64)
    sums_sig = np.zeros((T, p), dtype=np.float64)
    counts   = np.bincount(inv, minlength=T).astype(np.float64)
    np.add.at(sums_abs, inv, np.abs(phi))
    np.add.at(sums_sig, inv, phi)
    per_abs = sums_abs / counts[:, None]
    per_sig = sums_sig / counts[:, None]
    return Tcodes, per_abs, per_sig

def bootstrap_date_balanced(per_date_matrix: np.ndarray, B: int = 300, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, _ = per_date_matrix.shape
    counts = rng.multinomial(T, np.full(T, 1.0 / T), size=B).astype(np.float64)
    boots = (counts @ per_date_matrix) / T
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return np.vstack([lo, hi]).T

def corr_phi_net(phi: np.ndarray, net: np.ndarray) -> np.ndarray:
    x = phi - phi.mean(axis=0, keepdims=True)
    y = net - net.mean()
    num = (x * y[:, None]).sum(axis=0)
    den = np.sqrt((x**2).sum(axis=0) * (y**2).sum())
    return num / np.maximum(den, 1e-30)

# XGBoost helpers
def booster_for_window(win_label: str, device: str | None = None) -> xgb.Booster:
    model_dir = MODEL_BASE / win_label
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "booster.json"
    bst = xgb.Booster()
    if model_path.exists():
        bst.load_model(str(model_path))
    if device is None:
        device = "cuda"  # will fall back if CUDA not available
    try:
        bst.set_param({"device": device, "nthread": NTHREADS})
    except Exception:
        pass
    return bst

def train_if_needed(win_label: str, df: pd.DataFrame, train_span: Tuple[str,str]) -> None:
    model_dir = MODEL_BASE / win_label
    model_path = model_dir / "booster.json"
    meta_path  = model_dir / "train_meta.json"

    if model_path.exists():
        print(f"[{win_label}] booster.json exists — skipping training.")
        return

    d0 = pd.to_datetime(train_span[0]).date()
    d1 = pd.to_datetime(train_span[1]).date()
    tr = df.loc[(df["date"] >= d0) & (df["date"] <= d1)].reset_index(drop=True)
    Xtr = tr[FEATURE_COLS]
    ytr = tr["class3"].values

    # inverse-frequency class weights for training
    cls, cnt = np.unique(ytr, return_counts=True)
    freq = cnt / cnt.sum()
    wmap = {c: (1.0 / f) for c, f in zip(cls, freq)}
    wtr = np.array([wmap[c] for c in ytr], dtype=np.float32)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED,
        "monotone_constraints": "(" + ",".join(map(str, MONO)) + ")",
        "device": "cuda",
    }
    rounds = ROUNDS[win_label]
    print(f"[{win_label}] Training for {rounds} rounds on {len(Xtr):,} rows ...")
    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr, nthread=NTHREADS)
    bst = xgb.train(params, dtrain, num_boost_round=rounds)
    model_dir.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(model_path))
    with open(meta_path, "w") as f:
        json.dump({"params": params, "rounds": rounds, "n_train": int(len(Xtr))}, f, indent=2)

def contribs_to_margin_shap(contribs: np.ndarray, n_features: int, n_classes: int) -> np.ndarray:
    """Accepts (n,(p+1)*C) OR (n,C,p+1); returns (n,p) margin SHAP = class2 - class0."""
    if contribs.ndim == 2:
        n = contribs.shape[0]
        arr = contribs.reshape(n, n_classes, n_features + 1)
    elif contribs.ndim == 3:
        arr = contribs
    else:
        raise ValueError(f"Unsupported contribs ndim={contribs.ndim}")
    return arr[:, 2, :n_features] - arr[:, 0, :n_features]

def predict_and_shap(bst: xgb.Booster, X: pd.DataFrame, batch: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and margin SHAP (class2 - class0) in batches."""
    preds, shap = [], []
    n = len(X)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        d = xgb.DMatrix(X.iloc[s:e], nthread=NTHREADS)
        preds.append(bst.predict(d))                      # (m, C)
        shap.append(bst.predict(d, pred_contribs=True))   # (m, C, p+1) or (m,(p+1)*C)
        if (s // batch) % 5 == 0:
            print(f"  ... SHAP batch {s:,}/{n:,}")
    proba = np.vstack(preds)
    C = np.vstack(shap) if shap[0].ndim == 2 else np.concatenate(shap, axis=0)
    phi_margin = contribs_to_margin_shap(C, n_features=X.shape[1], n_classes=proba.shape[1])
    return proba, phi_margin

# unified streaming predictor (for base & perm)
def predict_proba_streaming(bst: xgb.Booster, X_np: np.ndarray, batch: int = BATCH) -> np.ndarray:
    n = X_np.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    use_inplace = hasattr(bst, "inplace_predict")
    for s in range(0, n, batch):
        e = min(n, s + batch)
        if use_inplace:
            P = bst.inplace_predict(X_np[s:e], predict_type="probabilities")
            P = np.asarray(P, dtype=np.float32).reshape(-1, 3)
        else:
            d = xgb.DMatrix(X_np[s:e])
            P = bst.predict(d)
        out[s:e] = P
        if (s // batch) % 10 == 0:
            print(f"    predict {s:,}/{n:,}")
    return out

# OOS summaries
def get_slice(df: pd.DataFrame, span: Tuple[str,str]) -> pd.DataFrame:
    d0 = pd.to_datetime(span[0]).date()
    d1 = pd.to_datetime(span[1]).date()
    return df.loc[(df["date"] >= d0) & (df["date"] <= d1)].reset_index(drop=True)

def per_window_summary(win: Dict[str, Any], features: List[str], B_boot: int = 300, tail_q: float = 0.10) -> pd.DataFrame:
    dates, phi, proba, net = win["dates"], win["phi"], win["proba"], win["net"]
    _, per_abs, per_sig = date_balanced_means(phi, dates)
    mean_abs, mean_sig = per_abs.mean(axis=0), per_sig.mean(axis=0)
    ci_abs = bootstrap_date_balanced(per_abs, B=B_boot)
    ci_sig = bootstrap_date_balanced(per_sig, B=B_boot)
    align  = corr_phi_net(phi, net)

    # tails (top-bottom by net_score within date)
    d = pd.DataFrame({"date": pd.to_datetime(dates), "net": net})
    qs = d.groupby("date")["net"].quantile([tail_q, 1 - tail_q]).unstack()
    qs.columns = ["low_q", "high_q"]
    is_low  = net <= qs.loc[d["date"].values, "low_q"].values
    is_high = net >= qs.loc[d["date"].values, "high_q"].values

    Tcodes2, inv = np.unique(pd.to_datetime(dates).values, return_inverse=True)
    T = len(Tcodes2); p = len(features)
    sums_high = np.zeros((T, p)); sums_low = np.zeros((T, p))
    cnt_high  = np.bincount(inv[is_high], minlength=T).astype(np.float64)
    cnt_low   = np.bincount(inv[is_low],  minlength=T).astype(np.float64)
    np.add.at(sums_high, inv[is_high], phi[is_high])
    np.add.at(sums_low,  inv[is_low],  phi[is_low])
    mean_high_dates = sums_high / np.maximum(cnt_high[:, None], 1.0)
    mean_low_dates  = sums_low  / np.maximum(cnt_low[:,  None], 1.0)
    high_db = np.nanmean(mean_high_dates, axis=0)
    low_db  = np.nanmean(mean_low_dates,  axis=0)

    df = (pd.DataFrame({
            "feature": features,
            "mean_abs_shap_margin": mean_abs,
            "ci_lo_abs": ci_abs[:, 0], "ci_hi_abs": ci_abs[:, 1],
            "mean_signed_shap_margin": mean_sig,
            "ci_lo_signed": ci_sig[:, 0], "ci_hi_signed": ci_sig[:, 1],
            "corr_vs_net": align,
            "tail_high_minus_low": high_db - low_db})
          .sort_values("mean_abs_shap_margin", ascending=False, kind="mergesort")
          .reset_index(drop=True))
    return df

def plot_global_importance(df: pd.DataFrame, title: str, out_png: Path) -> None:
    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap_margin"])
    xerr = np.vstack([top["mean_abs_shap_margin"] - top["ci_lo_abs"],
                      top["ci_hi_abs"] - top["mean_abs_shap_margin"]])
    ax.errorbar(top["mean_abs_shap_margin"], np.arange(len(top)), xerr=xerr, fmt="none", capsize=3)
    ax.set_xlabel("Date-balanced mean |SHAP| (tail margin)")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

def run_window(win_label: str, df: pd.DataFrame, train_span: Tuple[str,str], test_span: Tuple[str,str]) -> Dict[str, Any]:
    out_dir = OUT_BASE / win_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train if needed
    train_if_needed(win_label, df, train_span)

    # Train & test slices (for eval weight map choice)
    tr = get_slice(df, train_span)
    te = get_slice(df, test_span)
    Xte = te[FEATURE_COLS]
    yte = te["class3"].values
    Yte = te[RET_COL].values
    dates = te["date"].values

    # Eval weights for weighted logloss
    if USE_CLASS_WEIGHTS:
        if EVAL_WEIGHT_SOURCE == "train":
            w_map = inverse_frequency_weights(tr["class3"].values, smoothing=EVAL_SMOOTHING)
        else:  # "test"
            w_map = inverse_frequency_weights(yte, smoothing=EVAL_SMOOTHING)
        sample_w = pd.Series(yte).map(w_map).to_numpy(dtype=float)
    else:
        sample_w = np.ones_like(yte, dtype=float)

    # Predict + SHAP
    bst = booster_for_window(win_label, device="cuda")  # falls back if no CUDA
    proba, phi = predict_and_shap(bst, Xte, batch=200_000)
    net = net_score_from_proba(proba)

    # Save raw outputs
    np.save(out_dir / "proba.npy", proba)
    np.save(out_dir / "phi_margin.npy", phi)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({
            "dates": dates, "n": len(Xte), "p": Xte.shape[1],
            "features": FEATURE_COLS, "window": win_label,
            "eval_weight_source": EVAL_WEIGHT_SOURCE, "use_class_weights": USE_CLASS_WEIGHTS
        }, f)

    # Summaries + plot
    summ = per_window_summary({"dates": dates, "phi": phi, "proba": proba, "net": net},
                              FEATURE_COLS, B_boot=300, tail_q=0.10)
    summ.to_csv(out_dir / f"{win_label}_shap_summary.csv", index=False)
    plot_global_importance(summ, f"{win_label} — Global importance (OOS)",
                           out_dir / f"{win_label}_global_importance.png")

    # Numeric features for permutation
    X_np = Xte.to_numpy(copy=True).astype(np.float32, copy=False)

    return {
        "X": Xte, "X_np": X_np, "y_clf": yte, "y_cont": Yte,
        "dates": dates, "proba": proba, "phi": phi, "net": net,
        "sample_w": sample_w, "out_dir": out_dir
    }

# Permutation with repeats
def build_perm_index_per_date(dates: np.ndarray, seed: int = RANDOM_SEED):
    """Return (perm_index, inv_date) to permute within each date efficiently."""
    dts = pd.to_datetime(dates).values
    inv = pd.factorize(dts)[0]
    n = len(inv)
    perm_index = np.empty(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]
    boundaries = np.flatnonzero(np.r_[True, np.diff(inv_sorted) != 0, True])
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        seg = order[b0:b1]
        perm_index[seg] = rng.permutation(seg)
    return perm_index, inv

def predict_metrics_permuted(
    bst: xgb.Booster,
    X_np: np.ndarray,
    y_clf: np.ndarray,
    y_cont: np.ndarray,
    dates: np.ndarray,
    inv_date: np.ndarray,
    perm_index: np.ndarray,
    cols_idx: np.ndarray,
    sample_w: np.ndarray,
    batch: int = BATCH,
):
    """
    Streamed permutation: apply within-date permutation to cols_idx, then compute:
      • date-balanced *weighted* logloss
      • full net vector for rank metrics
    """
    n, p = X_np.shape
    tmp = np.empty((batch, p), dtype=X_np.dtype)
    use_inplace = hasattr(bst, "inplace_predict")

    T = inv_date.max() + 1
    sum_w_nll = np.zeros(T, dtype=np.float64)
    sum_w     = np.zeros(T, dtype=np.float64)
    net = np.empty(n, dtype=np.float32)

    for s in range(0, n, batch):
        e = min(n, s + batch); m = e - s
        rows = np.arange(s, e)

        tmp[:m] = X_np[s:e]
        if cols_idx.size > 0:
            src = perm_index[rows]
            tmp[:m, cols_idx] = X_np[src[:, None], cols_idx]

        if use_inplace:
            P = bst.inplace_predict(tmp[:m], predict_type="probabilities")
            P = np.asarray(P, dtype=np.float64).reshape(m, 3)
        else:
            d = xgb.DMatrix(tmp[:m])
            P = bst.predict(d)

        # logloss accumulators (weighted)
        p_y = P[np.arange(m), y_clf[s:e]]
        nll = -np.log(np.clip(p_y, 1e-15, 1.0))
        w_batch = sample_w[s:e].astype(np.float64)
        b_inv = inv_date[s:e]
        sum_w_nll += np.bincount(b_inv, weights=w_batch * nll, minlength=T)
        sum_w     += np.bincount(b_inv, weights=w_batch,       minlength=T)

        # net for ranking metrics
        net[s:e] = (P[:, 2] - P[:, 0]).astype(np.float32)

        if (s // batch) % 10 == 0:
            print(f"    perm batch {s:,}/{n:,}")

    per_date = sum_w_nll / np.maximum(sum_w, 1e-12)
    return float(np.nanmean(per_date)), net

def permutation_block(win_label: str, win: Dict[str, Any], groups=GROUPS, repeats: int = N_PERM, batch: int = BATCH) -> pd.DataFrame:
    """
    For each group:
      • compute BASE metrics via unified streaming predictor
      • do 'repeats' independent within-date shuffles
      • report medians + 95% CIs of:
          Δ (date-balanced weighted logloss), Δ rankIC, Δ Top–Bottom
    """
    out_dir = Path(win["out_dir"])
    cache_csv = out_dir / f"{win_label}_perm_importance.csv"
    if cache_csv.exists():
        print(f"[{win_label}] permutation CSV exists — skipping recompute.")
        return pd.read_csv(cache_csv)

    X_np  = win["X_np"]
    y_clf = win["y_clf"]; y_cont = win["y_cont"]; dates = win["dates"]
    sample_w = win["sample_w"]

    # Booster for both base and perm
    bst = booster_for_window(win_label, device="cuda")
    try:
        bst.set_param({"nthread": NTHREADS})
    except Exception:
        pass

    # Baseline via the same streaming path
    print(f"[{win_label}] Computing BASE metrics via unified streaming predictor ...")
    base_proba = predict_proba_streaming(bst, X_np, batch=batch)
    base_loss  = date_balanced_weighted_logloss_from_proba(base_proba, y_clf, dates, sample_w)
    base_net   = (base_proba[:, 2] - base_proba[:, 0]).astype(np.float32)
    base_ic, _ = date_balanced_rank_ic(base_net, y_cont, dates)
    base_tb, _ = date_balanced_topbottom(base_net, y_cont, dates, q=TAIL_FRAC)
    print(f"[{win_label}] BASE: wLogLoss={base_loss:.6f} | rankIC={base_ic:.4f} | Top-Bottom={base_tb:.6f}")

    col_index = {f: i for i, f in enumerate(FEATURE_COLS)}
    rows = []

    for g in groups:
        cols_idx = np.array([col_index[c] for c in g], dtype=np.int32)

        dl, dic, dtb = [], [], []
        for r in range(repeats):
            print(f"[{win_label}] perm group {g} (rep {r+1}/{repeats}) ...")
            perm_index, inv_date = build_perm_index_per_date(dates, seed=RANDOM_SEED + r)
            perm_loss, perm_net  = predict_metrics_permuted(
                bst, X_np, y_clf, y_cont, dates, inv_date, perm_index, cols_idx, sample_w, batch=batch
            )
            # ranking diagnostics from perm_net
            perm_ic, _ = date_balanced_rank_ic(perm_net, y_cont, dates)
            perm_tb, _ = date_balanced_topbottom(perm_net, y_cont, dates, q=TAIL_FRAC)

            dl.append(perm_loss - base_loss)
            dic.append(perm_ic   - base_ic)
            dtb.append(perm_tb   - base_tb)

        def med(a): return float(np.median(a))
        def lo(a):  return float(np.percentile(a, 2.5))
        def hi(a):  return float(np.percentile(a, 97.5))

        rows.append({
            "group": "+".join(g), "k": len(g), "n_perm": repeats,
            "delta_logloss_med": med(dl), "delta_logloss_lo": lo(dl), "delta_logloss_hi": hi(dl),
            "delta_rankIC_med":  med(dic), "delta_rankIC_lo":  lo(dic), "delta_rankIC_hi":  hi(dic),
            "delta_TopBottom_med": med(dtb), "delta_TopBottom_lo": lo(dtb), "delta_TopBottom_hi": hi(dtb),
        })

    df = pd.DataFrame(rows).sort_values("delta_logloss_med", ascending=False)
    df.to_csv(cache_csv, index=False)
    print(f"[{win_label}] permutation results saved -> {cache_csv}")
    return df


# Data loading
def load_ltr_df_from_hdf() -> pd.DataFrame:
    path = Path(LTR_HDF_PATH)
    assert path.exists(), f"HDF file not found: {path}"
    df = pd.read_hdf(path, key=LTR_HDF_KEY)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=None, help="0..3; else use SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()

    # pick window
    if args.window is None:
        arr = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if arr is None:
            raise SystemExit("Pass --window 0..3 (or set SLURM_ARRAY_TASK_ID).")
        win_idx = int(arr)
    else:
        win_idx = int(args.window)
    assert 0 <= win_idx <= 3
    win_label = WIN_LABELS[win_idx]
    train_span, test_span = TRAIN_TEST_LSTS[win_idx]

    print(f"[INFO] Loading ltr_df from HDF: {LTR_HDF_PATH} [key={LTR_HDF_KEY}]")
    ltr_df = load_ltr_df_from_hdf()
    for c in [RET_COL, "date", *FEATURE_COLS]:
        assert c in ltr_df.columns, f"Missing column: {c}"

    if "class3" not in ltr_df.columns:
        print("[INFO] Building class3 labels ...")
        ltr_df = make_class3_fast(ltr_df, date_col="date", ret_col=RET_COL, frac=TAIL_FRAC)

    print(f"[RUN] {win_label} | Train {train_span} | Test {test_span}")
    res = run_window(win_label, ltr_df, train_span, test_span)

    print(f"[PERM] {win_label} | within-date groups ({N_PERM} repeats)")
    _ = permutation_block(win_label, res, groups=GROUPS, repeats=N_PERM, batch=BATCH)

    print("[DONE]", win_label)

if __name__ == "__main__":
    # Thread discipline to avoid oversubscription 
    os.environ["OMP_NUM_THREADS"]      = str(NTHREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"
    main()
