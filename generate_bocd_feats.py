"""
BOCD per-stock feature pass (resumable; no look-ahead).

What it does
- Runs Bayesian Online Changepoint Detection on each stock’s daily returns with a constant hazard.
- Warms up on 252 trading days. At each month-end, refits a Student-t to an as-of top-1000 set and
  shrinks only scale/tails toward that market anchor (mean is left alone).
- Keeps a tail-truncated run-length posterior (MAX_RUN). Writes one HDF per stock; safe to resume.

Features we emit per day t
Core regime signals
- p_t = Pr(r_t = 0)     # break probability today
- E_rt (and dE_rt)      # expected run length; day-to-day change
- P_r_le_k              # Pr(r_t ≤ k) for k in {5,10,14,21,42,63,126,252}
- r_med                 # median run length
- H_rt, Var_rt          # entropy / variance of the run-length posterior

Prediction-leaning extras
Pre-update (for x_t):
- log_pred, pred_mu_1d_prior, pred_var_1d_prior
- std_surprise2 = ((x_t − pred_mu_1d_prior)**2) / pred_var_1d_prior

Post-update (for x_{t+1}):
- pred_mu_1d_post, pred_var_1d_post, pred_sr_1d

21-day hazard-aware proxies (continue vs reset mix)
- E21, V21, V21_ms, SR21

Slow rolls
- mean_log_pred_21, mean_std_surprise2_21

Notes
- First output appears at day 252 so it lines up with the rest of the feature set.
- Shrinkage is skipped on days when the top-1000 mask is too thin.
- r=0 prior is kept; any excess tail mass is folded into r=MAX_RUN.
- N_JOBS = CPU−1; progress via tqdm. Z-scoring happens downstream.
"""


# Thread guards 
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Imports
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import t as student_t

# Configuration
INPUT_HDF = "data/returns_matrix_core_stocks.h5"
KEY = "returns_matrix_core_stocks"

# As-of-date Top-1000 mask (bool DataFrame), saved with key==filename
TOPK_MASK_HDF = "features/top_1000_for_bocd_regularization.h5"
TOPK_MASK_KEY = os.path.splitext(os.path.basename(TOPK_MASK_HDF))[0]

# Where to save outputs (Windows absolute path)
BASE_FEAT_DIR = r"features"
OUTPUT_DIR = os.path.join(BASE_FEAT_DIR, "bocd_per_stock_feats")

# master consolidation
MASTER_OUT_HDF = os.path.join(BASE_FEAT_DIR, "bocd_per_stock_feats.h5")
MASTER_OUT_KEY = os.path.splitext(os.path.basename(MASTER_OUT_HDF))[0]

# Algorithm settings
try:
    CPU_COUNT = len(os.sched_getaffinity(0))
except Exception:
    CPU_COUNT = os.cpu_count() or 1
N_JOBS = max(1, CPU_COUNT - 1)

SPAN = 252
N0_DEFAULT = 200.0
HAZARD = 0.005 
MAX_RUN = 800
TAU_SHRINK = 200.0

# Shrinkage coverage threshold: if day t has < this many True in mask, NO SHRINK that day
MIN_TOPK_PER_DAY = 500

# BOCD Student-t emission
# -----------------------------
from bocd.distribution import StudentT  # type: ignore

# Important: Patch library due to a bug
def _update_params_fixed(self, x):
    """
    Correct NIG one-step update for StudentT sufficient statistics.
    Appends a new state (r->r+1) and keeps r=0 prior at the head.
    """
    x = float(x)
    self.betaT = np.concatenate([
        self.beta0,
        self.betaT + 0.5 * (self.kappaT * (x - self.muT) ** 2) / (self.kappaT + 1),
    ])
    self.muT = np.concatenate([
        self.mu0,
        (self.kappaT * self.muT + x) / (self.kappaT + 1),
    ])
    self.kappaT = np.concatenate([self.kappa0, self.kappaT + 1])
    self.alphaT = np.concatenate([self.alpha0, self.alphaT + 0.5])

StudentT.update_params = _update_params_fixed

# Helpers
# -----------------------------
def ewma_weights(n: int, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    w = np.array([alpha * ((1 - alpha) ** (n - 1 - i)) for i in range(n)], dtype=float)
    return w / w.sum()

def fit_student_t_weighted_mle(y: np.ndarray, w: Optional[np.ndarray] = None) -> tuple[float, float, float]:
    """
    Weighted Student-t MLE with robust init and bounds.
    Optimize over (mu, log_s, log_nu_shift) with nu = exp(log_nu_shift) + 2.1 (> 2).
    """
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    y = y[m]
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)[m]
        if not np.isfinite(w).all() or np.sum(w) <= 0:
            w = np.ones_like(y, dtype=float)
        w = w / np.sum(w)

    if len(y) < 20:
        mu_hat = float(np.nanmean(y)) if len(y) else 0.0
        s_hat = float(np.nanstd(y)) if len(y) else 1e-3
        return mu_hat, max(s_hat, 1e-8), 8.0

    mu0 = np.median(y)
    mad = np.median(np.abs(y - mu0)) or 1e-6
    s0 = max(mad * 1.4826, np.std(y) or 1e-6)
    nu0 = 8.0

    def nll(params):
        mu, log_s, log_nu = params
        s = np.exp(log_s)
        nu = np.exp(log_nu) + 2.1
        ll = student_t.logpdf(y, df=nu, loc=mu, scale=s)
        return -np.sum(w * ll)

    bounds = [
        (-np.inf, np.inf),
        (np.log(1e-8), np.log(1.0)),           # s in [1e-8, 1.0] (returns in decimals)
        (np.log(4 - 2.1), np.log(50 - 2.1)),   # nu in [4, 50] (variance exists)
    ]

    res = minimize(
        nll,
        x0=np.array([mu0, np.log(s0), np.log(max(nu0 - 2.1, 0.1))]),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400},
    )

    if (not res.success) or (not np.all(np.isfinite(res.x))):
        return float(mu0), max(float(s0), 1e-8), 8.0

    mu_hat, log_s_hat, log_nu_hat = res.x
    s_hat = float(np.exp(log_s_hat))
    nu_hat = float(np.exp(log_nu_hat) + 2.1)
    nu_hat = float(np.clip(nu_hat, 4.0, 50.0))
    s_hat = max(s_hat, 1e-8)
    return mu_hat, s_hat, nu_hat

def map_t_to_nig(mu_hat: float, s_hat: float, nu_hat: float, n0: float) -> tuple[float, float, float, float]:
    """
    Map Student-t params to Normal-Inverse-Gamma prior hyperparameters.
    """
    mu0 = float(mu_hat)
    kappa0 = float(n0)
    alpha0 = float(nu_hat / 2.0)
    beta0 = float(alpha0 * (s_hat ** 2) * (kappa0 / (kappa0 + 1.0)))
    return mu0, kappa0, alpha0, beta0

def month_end_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    g = s.groupby(idx.to_period("M")).max()
    return pd.DatetimeIndex(g.values)

def _base_key_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _ensure_bool_mask(mask: pd.DataFrame, like: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    first_date = pd.to_datetime(mask.index.min())
    mask = mask.reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)
    return mask, first_date

def nig_predictive_params(mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For NIG(mu, kappa, alpha, beta), the one-step predictive is Student-t with:
        df = 2*alpha
        loc = mu
        scale^2 = beta * (kappa + 1) / (alpha * kappa)
    Returns (loc, var, df) with var computed as scale^2 * df / (df - 2) if df > 2, else NaN.
    """
    mu = mu.astype(float)
    kappa = kappa.astype(float)
    alpha = alpha.astype(float)
    beta = beta.astype(float)

    df = 2.0 * alpha
    safe_kappa = np.maximum(kappa, 1e-12)
    safe_alpha = np.maximum(alpha, 1e-12)
    scale2 = beta * (safe_kappa + 1.0) / (safe_alpha * safe_kappa)
    var = np.where(df > 2.0, scale2 * (df / (df - 2.0)), np.nan)
    return mu, var, df

def preupdate_mixture_prior_weights(beliefs: np.ndarray, h: float) -> np.ndarray:
    """
    Prior over run length at time t before observing x_t:
        w[0]   = h * sum(beliefs)
        w[r+1] = (1 - h) * beliefs[r]
    Sums to 1 when beliefs sums to 1.
    """
    n = beliefs.size
    w = np.empty(n + 1, dtype=float)
    s = float(beliefs.sum())
    w[0] = h * s
    w[1:] = (1.0 - h) * beliefs
    return w

def mixture_mean_var(weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> Tuple[float, float]:
    """
    Compute mixture mean and variance from component weights, means, variances.
    weights assumed to sum to 1.
    """
    w = weights.astype(float)
    m = means.astype(float)
    v = variances.astype(float)
    mu_mix = float(np.sum(w * m))
    ex2 = np.sum(w * (v + m * m))
    var_mix = float(max(ex2 - mu_mix * mu_mix, 0.0))
    return mu_mix, var_mix

CDF_THRESHOLDS = np.array([5, 10, 14, 21, 42, 63, 126, 252], dtype=int)

def posterior_entropy(p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))

def posterior_variance(p: np.ndarray) -> float:
    idx = np.arange(p.size, dtype=np.float64)
    mean = float(np.dot(idx, p))
    ex2 = float(np.dot(idx * idx, p))
    return max(ex2 - mean * mean, 0.0)

def posterior_median(p: np.ndarray) -> float:
    c = np.cumsum(p.astype(float))
    return float(np.searchsorted(c, 0.5, side="left"))

def posterior_cdf_buckets(p: np.ndarray, thresholds: np.ndarray = CDF_THRESHOLDS) -> dict:
    """
    Inclusive CDF: P(r <= k). If k exceeds vector length-1, returns 1.0.
    """
    c = np.cumsum(p.astype(float))
    out = {}
    last = p.size - 1
    for k in thresholds:
        kk = min(k, last)
        out[int(k)] = float(c[kk])
    return out

# As-of top-1000 market fits (no look-ahead)
# -----------------------------
def precompute_market_fits_idx(
    returns_matrix: pd.DataFrame,
    span: int = 252,
    topk_mask: Optional[pd.DataFrame] = None,
    mask_first_date: Optional[pd.Timestamp] = None,
    min_topk_per_day: int = MIN_TOPK_PER_DAY,
) -> tuple[dict, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = returns_matrix.index
    month_end_idx = month_end_trading_days(dates)
    if month_end_idx[0] != dates[0]:
        month_end_idx = month_end_idx.insert(0, dates[0])
    month_end_idx = month_end_idx.unique()

    market_fit: dict[pd.Timestamp, tuple[float, float, float]] = {}
    low_cov_dates: list[pd.Timestamp] = []
    pre_mask_dates: list[pd.Timestamp] = []

    for dt in tqdm(month_end_idx, desc="Computing market fits (as-of top-1000)"):
        end_loc = dates.get_loc(dt)
        start_loc = max(0, end_loc - span + 1)

        # Decide shrink eligibility
        can_shrink = False
        if (topk_mask is not None) and (mask_first_date is not None) and (dt >= mask_first_date):
            try:
                day_count = int(topk_mask.loc[dt].sum())
            except KeyError:
                day_count = 0
            if day_count >= min_topk_per_day:
                can_shrink = True
            else:
                low_cov_dates.append(pd.Timestamp(dt))
        else:
            pre_mask_dates.append(pd.Timestamp(dt))

        if can_shrink:
            window = returns_matrix.iloc[start_loc:end_loc + 1]
            mwin = topk_mask.iloc[start_loc:end_loc + 1]
            window = window.where(mwin)
            vals = window.values.ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 50:
                w = ewma_weights(len(vals), span=span)
                mu, s, nu = fit_student_t_weighted_mle(vals, w=w)
                market_fit[dt] = (mu, s, nu)
            else:
                market_fit[dt] = (np.nan, np.nan, np.nan)
        else:
            market_fit[dt] = (np.nan, np.nan, np.nan)

    return (
        market_fit,
        pd.DatetimeIndex(month_end_idx),
        pd.DatetimeIndex(sorted(set(low_cov_dates))),
        pd.DatetimeIndex(sorted(set(pre_mask_dates))),
    )

def nearest_market_fit_idx(
    dt: pd.Timestamp,
    month_end_idx: pd.DatetimeIndex,
    market_fit: dict
) -> tuple[float, float, float]:
    pos = month_end_idx.get_indexer([pd.Timestamp(dt)], method="pad")[0]
    if pos < 0:
        pos = 0
    dt_anchor = month_end_idx[pos]
    return market_fit[dt_anchor]

def trim_studentt_state(st: StudentT, keep: int) -> None:
    keep = max(1, int(keep))
    for name in ("muT", "kappaT", "alphaT", "betaT"):
        arr = getattr(st, name, None)
        if arr is None:
            continue
        if arr.size > keep:
            head = arr[:1]
            tail = arr[-(keep - 1):] if keep > 1 else arr[:0]
            setattr(st, name, np.concatenate([head, tail]))

# Core BOCD step (tail-absorb) with summaries
# -----------------------------
def bocd_step_tail_absorb_with_summaries(
    st: StudentT,
    beliefs: np.ndarray,  # prior at t-1
    ret: float,
    h: float,
    max_run: int
) -> tuple[np.ndarray, dict]:
    """
    One BOCD update with:
      - Correct recursion (growth uses π_{r+1}, CP uses π_0 * sum(beliefs))
      - Tail-absorb truncation of posterior vector
      - Predictive mixture stats computed from the **pre-update** prior
      - Posterior summaries (entropy, variance, median, CDF buckets)
      - post-update predictive stats for x_{t+1}
      - hazard-aware 21d approximations (E21, V21, V21_ms, SR21)
    Returns:
      post (np.ndarray), features (dict)
    """
    EPS = 1e-12

    # --- Pre-update predictive (for x_t) ---
    prior_w = preupdate_mixture_prior_weights(beliefs, h)  # length = beliefs.size + 1
    mu_pred_r, var_pred_r, _df = nig_predictive_params(st.muT, st.kappaT, st.alphaT, st.betaT)
    if mu_pred_r.size != prior_w.size:
        pad_n = prior_w.size - mu_pred_r.size
        mu_pred_r = np.concatenate([mu_pred_r, np.repeat(mu_pred_r[-1], pad_n)])
        var_pred_r = np.concatenate([var_pred_r, np.repeat(var_pred_r[-1], pad_n)])
    pred_mu_1d_prior, pred_var_1d_prior = mixture_mean_var(prior_w, mu_pred_r, var_pred_r)

    # --- Likelihoods for x_t and posterior over r_t ---
    pi_full = st.pdf(float(ret)).astype(float)
    if pi_full.size < beliefs.size + 1:
        pad = np.full(beliefs.size + 1 - pi_full.size, pi_full[-1], dtype=float)
        pi_full = np.concatenate([pi_full, pad])

    idxs = np.minimum(np.arange(beliefs.size) + 1, pi_full.size - 1)  # growth uses π_{r+1}
    growth = beliefs * (1.0 - h) * pi_full[idxs]
    cp_unnorm = h * float(pi_full[0]) * beliefs.sum()

    post_unnorm = np.empty(beliefs.size + 1, dtype=float)
    post_unnorm[0] = cp_unnorm
    post_unnorm[1:] = growth

    evidence = float(post_unnorm.sum())
    post = post_unnorm / evidence

    # Tail-absorb
    if post.size > max_run:
        tail_mass = post[max_run:].sum()
        post = post[:max_run].copy()
        post[-1] += tail_mass

    # Posterior summaries
    p_t = float(post[0])
    idx = np.arange(post.size, dtype=np.float64)
    E_rt = float(np.dot(idx, post))
    log_pred = np.log(evidence + EPS)
    P_r_le_5 = float(post[:min(6, post.size)].sum())

    H_rt = posterior_entropy(post)
    Var_rt = posterior_variance(post)
    r_med = posterior_median(post)
    cdf_buckets = posterior_cdf_buckets(post)

    #  Update state arrays to time t (needed for x_{t+1} predictive) 
    st.update_params(float(ret))
    trim_studentt_state(st, keep=max_run)

    # Post-update predictive for x_{t+1}
    # Mixture weights for next step (unconditional):
    w_next = preupdate_mixture_prior_weights(post, h)  # length = post.size + 1
    mu_post_r, var_post_r, _ = nig_predictive_params(st.muT, st.kappaT, st.alphaT, st.betaT)
    if mu_post_r.size != w_next.size:
        pad_n = w_next.size - mu_post_r.size
        mu_post_r = np.concatenate([mu_post_r, np.repeat(mu_post_r[-1], pad_n)])
        var_post_r = np.concatenate([var_post_r, np.repeat(var_post_r[-1], pad_n)])

    pred_mu_1d_post, pred_var_1d_post = mixture_mean_var(w_next, mu_post_r, var_post_r)

    # NaN-safe denominators for pred_sr_1d
    den_ps = np.sqrt(pred_var_1d_post)
    den_ps = den_ps if np.isfinite(den_ps) and den_ps > EPS else EPS
    pred_sr_1d = float(pred_mu_1d_post / den_ps)

    # Standardized surprise for x_t (relative to pre-update mixture) with NaN-safe denom
    den_prior = pred_var_1d_prior
    den_prior = den_prior if np.isfinite(den_prior) and den_prior > EPS else EPS
    std_surprise2 = float(((float(ret) - pred_mu_1d_prior) ** 2) / den_prior)

    # Continue vs Reset scenario stats for horizon features 
    #  Build w_cont explicitly as [0, post] so it sums to 1 by construction
    w_cont = np.concatenate(([0.0], post))
    mu_cont, var_cont = mixture_mean_var(w_cont, mu_post_r, var_post_r)

    # "Reset" (new-regime) mean/var from r=0 component
    mu0 = float(mu_post_r[0])
    var0 = float(var_post_r[0])

    # Clamp negatives/NaNs to safe values
    var_cont = var_cont if np.isfinite(var_cont) and var_cont >= 0.0 else 0.0
    var0 = var0 if np.isfinite(var0) and var0 >= 0.0 else 0.0

    # Hazard survival for k=21
    k = 21
    i = np.arange(1, k + 1, dtype=float)
    p_surv = (1.0 - h) ** i  # p_i
    S_k = float(p_surv.sum())
    R_k = k - S_k

    # 21d expectation and variance approximations
    E21 = float(mu_cont * S_k + mu0 * R_k)
    V21 = float(var_cont * S_k + var0 * R_k)

    # Mean-switching corrected variance
    V21_ms = float(np.sum(p_surv * var_cont + (1.0 - p_surv) * var0 + (mu_cont - mu0) ** 2 * p_surv * (1.0 - p_surv)))

    # NaN-safe SR21 using V21_ms
    den_sr21 = np.sqrt(V21_ms)
    den_sr21 = den_sr21 if np.isfinite(den_sr21) and den_sr21 > EPS else EPS
    SR21 = float(E21 / den_sr21)

    feats = {
        "p_t": p_t,
        "E_rt": E_rt,
        "P_r_le_5": P_r_le_5,
        "log_pred": log_pred,
        "H_rt": H_rt,
        "Var_rt": Var_rt,
        "r_med": r_med,
        "pred_mu_1d_prior": float(pred_mu_1d_prior),
        "pred_var_1d_prior": float(pred_var_1d_prior),

        "pred_mu_1d_post": float(pred_mu_1d_post),
        "pred_var_1d_post": float(pred_var_1d_post),
        "pred_sr_1d": float(pred_sr_1d),

        "E21": E21,
        "V21": V21,
        "V21_ms": V21_ms,
        "SR21": SR21,

        # For rolling stat
        "std_surprise2": std_surprise2,
    }
    feats.update({f"P_r_le_{k}": cdf_buckets[k] for k in CDF_THRESHOLDS})
    return post, feats

# Resumable per-stock worker
# -----------------------------
PER_STOCK_FILE_PREFIX = "bocd_asof_top1k_mle__summaries"

def stock_out_path(permno: str, output_dir: str) -> str:
    base = f"{PER_STOCK_FILE_PREFIX}__{permno}"
    return os.path.join(output_dir, f"{base}.h5")

def already_done(permno: str, output_dir: str) -> bool:
    path = stock_out_path(permno, output_dir)
    return os.path.exists(path) and os.path.getsize(path) > 0

def write_atomic_hdf(df: pd.DataFrame, path: str, key: Optional[str] = None) -> None:
    if key is None:
        key = _base_key_from_path(path)
    tmp = path + ".tmp"
    df.to_hdf(tmp, key=key, mode="w")
    os.replace(tmp, path)

def process_one_stock_tailabsorb(
    permno: str,
    returns: pd.Series,
    month_end_trd: pd.DatetimeIndex,
    market_fit: dict,
    month_end_idx: pd.DatetimeIndex,
    output_dir: str,
    span: int = SPAN,
    n0_default: float = N0_DEFAULT,
    hazard: float = HAZARD,
    max_run: int = MAX_RUN,
    tau_shrink: float = TAU_SHRINK,
) -> Optional[pd.DataFrame]:
    """
    Process a single stock; write per-stock HDF atomically; return df or None if skipped.
    """
    out_path = stock_out_path(permno, output_dir)
    if already_done(permno, output_dir):
        return None

    valid = returns.notna().astype(int)
    roll = valid.rolling(span, min_periods=span).sum()
    elig_date = roll.index[roll.ge(span) & valid.eq(1)].min()
    if pd.isna(elig_date):
        os.makedirs(output_dir, exist_ok=True)
        # (#2) include std_surprise2 in empty schema
        empty_cols = [
            "date","permno",
            "p_t","E_rt","dE_rt","P_r_le_5","log_pred",
            "P_r_le_10","P_r_le_14","P_r_le_21","P_r_le_42",
            "P_r_le_63","P_r_le_126","P_r_le_252",
            "H_rt","Var_rt","r_med",
            "pred_mu_1d_prior","pred_var_1d_prior",
            "pred_mu_1d_post","pred_var_1d_post","pred_sr_1d",
            "E21","V21","V21_ms","SR21",
            "std_surprise2",
            "mean_log_pred_21","mean_std_surprise2_21"
        ]
        empty = pd.DataFrame(columns=empty_cols)
        write_atomic_hdf(empty, out_path, key=_base_key_from_path(out_path))
        return None

    h = float(np.clip(hazard, 1e-12, 1 - 1e-12))

    # Warm-up window up to elig_date (inclusive)
    win = returns.loc[:elig_date].dropna().iloc[-span:]
    y_full = win.values
    w = ewma_weights(len(y_full), span=span)
    s_mu, s_s, s_nu = fit_student_t_weighted_mle(y_full, w=w)

    # Market shrinkage at elig_date (skip if target NaN)
    _, m_s, m_nu = nearest_market_fit_idx(elig_date, month_end_idx, market_fit)
    n_i = len(y_full)
    if np.isfinite(m_s) and np.isfinite(m_nu):
        lam_sh = n_i / (n_i + tau_shrink)
        s_s_star = lam_sh * s_s + (1 - lam_sh) * m_s
        s_nu_star = lam_sh * s_nu + (1 - lam_sh) * m_nu
    else:
        s_s_star, s_nu_star = s_s, s_nu

    mu0, kappa0, alpha0, beta0 = map_t_to_nig(s_mu, s_s_star, s_nu_star, n0=n0_default)
    st = StudentT(mu0, kappa0, alpha0, beta0)

    # Warm-up replay (no emission)
    beliefs = np.array([1.0], dtype=np.float64)
    for z in y_full[:-1]:
        beliefs, _ = bocd_step_tail_absorb_with_summaries(st, beliefs, float(z), h, max_run)

    # Emit from elig_date onward
    records: list[dict] = []
    prev_E = np.nan

    z_star = float(y_full[-1])
    beliefs, feats = bocd_step_tail_absorb_with_summaries(st, beliefs, z_star, h, max_run)
    record = {
        "date": elig_date,
        "permno": permno,
        "p_t": feats["p_t"],
        "E_rt": feats["E_rt"],
        "dE_rt": np.nan,
        "P_r_le_5": feats["P_r_le_5"],
        "P_r_le_10": feats["P_r_le_10"],
        "P_r_le_14": feats["P_r_le_14"],
        "P_r_le_21": feats["P_r_le_21"],
        "P_r_le_42": feats["P_r_le_42"],
        "P_r_le_63": feats["P_r_le_63"],
        "P_r_le_126": feats["P_r_le_126"],
        "P_r_le_252": feats["P_r_le_252"],
        "log_pred": feats["log_pred"],
        "H_rt": feats["H_rt"],
        "Var_rt": feats["Var_rt"],
        "r_med": feats["r_med"],
        "pred_mu_1d_prior": feats["pred_mu_1d_prior"],
        "pred_var_1d_prior": feats["pred_var_1d_prior"],
        "pred_mu_1d_post": feats["pred_mu_1d_post"],
        "pred_var_1d_post": feats["pred_var_1d_post"],
        "pred_sr_1d": feats["pred_sr_1d"],
        "E21": feats["E21"],
        "V21": feats["V21"],
        "V21_ms": feats["V21_ms"],
        "SR21": feats["SR21"],
        "std_surprise2": feats["std_surprise2"],
    }
    records.append(record)
    prev_E = feats["E_rt"]

    me_set = set(month_end_trd)

    for dt, ret in returns.loc[elig_date:].iloc[1:].items():
        # Refresh prior on last trading day regardless of ret NaN
        if dt in me_set:
            tail = returns.loc[:dt].dropna().iloc[-span:]
            if len(tail) >= 100:
                y = tail.values
                w = ewma_weights(len(y), span=span)
                s_mu, s_s, s_nu = fit_student_t_weighted_mle(y, w=w)

                # Market shrinkage (skip if target NaN)
                _, m_s, m_nu = nearest_market_fit_idx(dt, month_end_idx, market_fit)
                if np.isfinite(m_s) and np.isfinite(m_nu):
                    n_i = len(y)
                    lam_sh = n_i / (n_i + tau_shrink)
                    s_s_star = lam_sh * s_s + (1 - lam_sh) * m_s
                    s_nu_star = lam_sh * s_nu + (1 - lam_sh) * m_nu
                else:
                    s_s_star, s_nu_star = s_s, s_nu

                mu0, kappa0, alpha0, beta0 = map_t_to_nig(s_mu, s_s_star, s_nu_star, n0=n0_default)
                st.mu0 = np.array([mu0], dtype=float)
                st.kappa0 = np.array([kappa0], dtype=float)
                st.alpha0 = np.array([alpha0], dtype=float)
                st.beta0 = np.array([beta0], dtype=float)

        if not np.isfinite(ret):
            continue

        beliefs, feats = bocd_step_tail_absorb_with_summaries(st, beliefs, float(ret), h, max_run)
        dE_rt = feats["E_rt"] - prev_E if np.isfinite(prev_E) else np.nan
        prev_E = feats["E_rt"]

        record = {
            "date": dt,
            "permno": permno,
            "p_t": feats["p_t"],
            "E_rt": feats["E_rt"],
            "dE_rt": dE_rt,
            "P_r_le_5": feats["P_r_le_5"],
            "P_r_le_10": feats["P_r_le_10"],
            "P_r_le_14": feats["P_r_le_14"],
            "P_r_le_21": feats["P_r_le_21"],
            "P_r_le_42": feats["P_r_le_42"],
            "P_r_le_63": feats["P_r_le_63"],
            "P_r_le_126": feats["P_r_le_126"],
            "P_r_le_252": feats["P_r_le_252"],
            "log_pred": feats["log_pred"],
            "H_rt": feats["H_rt"],
            "Var_rt": feats["Var_rt"],
            "r_med": feats["r_med"],
            "pred_mu_1d_prior": feats["pred_mu_1d_prior"],
            "pred_var_1d_prior": feats["pred_var_1d_prior"],
            "pred_mu_1d_post": feats["pred_mu_1d_post"],
            "pred_var_1d_post": feats["pred_var_1d_post"],
            "pred_sr_1d": feats["pred_sr_1d"],
            "E21": feats["E21"],
            "V21": feats["V21"],
            "V21_ms": feats["V21_ms"],
            "SR21": feats["SR21"],
            "std_surprise2": feats["std_surprise2"],
        }
        records.append(record)

    # Build DF and add rolling 21d aggregates
    df = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)

    # 21-day rolling means (require full window to avoid look-ahead)
    df["mean_log_pred_21"] = df["log_pred"].rolling(window=21, min_periods=21).mean()
    df["mean_std_surprise2_21"] = df["std_surprise2"].rolling(window=21, min_periods=21).mean()

    # Atomic per-stock write (key == filename)
    os.makedirs(output_dir, exist_ok=True)
    write_atomic_hdf(df, out_path, key=_base_key_from_path(out_path))
    return df

# Orchestrator (resumable)
# -----------------------------
def run_pipeline_resumable(
    input_hdf: str,
    key: str,
    output_dir: str,
    write_master: bool = True,
    master_hdf: str = MASTER_OUT_HDF,
    master_key: str = MASTER_OUT_KEY
) -> pd.DataFrame:
    # Load returns panel
    returns_matrix = pd.read_hdf(input_hdf, key=key)
    if not isinstance(returns_matrix.index, pd.DatetimeIndex):
        if "date" in returns_matrix.columns:
            returns_matrix["date"] = pd.to_datetime(returns_matrix["date"])
            returns_matrix = returns_matrix.set_index("date")
        else:
            returns_matrix.index = pd.to_datetime(returns_matrix.index)
    returns_matrix = returns_matrix.sort_index()

    # Load and align top-1000 mask (bool); keep original first mask date for 'pre-mask' detection
    topk_mask_raw = pd.read_hdf(TOPK_MASK_HDF, key=TOPK_MASK_KEY).astype(bool)
    topk_mask, mask_first_date = _ensure_bool_mask(topk_mask_raw, returns_matrix)

    # Month-end trading days (once)
    month_end_trd = month_end_trading_days(returns_matrix.index)

    # Market fits (on month-end trading days) via as-of-date top-1000, with per-day threshold
    market_fit, month_end_idx, low_cov_dates, pre_mask_dates = precompute_market_fits_idx(
        returns_matrix, span=SPAN, topk_mask=topk_mask,
        mask_first_date=mask_first_date, min_topk_per_day=MIN_TOPK_PER_DAY
    )

    if len(pre_mask_dates) > 0:
        print(f"[INFO] Shrinkage disabled for {len(pre_mask_dates)} month-ends before mask start "
              f"({mask_first_date.date()}): {[d.date() for d in pre_mask_dates]}")
    if len(low_cov_dates) > 0:
        print(f"[INFO] Shrinkage disabled for {len(low_cov_dates)} month-ends with "
              f"top-k count < {MIN_TOPK_PER_DAY}: {[d.date() for d in low_cov_dates]}")

    tickers = list(returns_matrix.columns)

    # Skip already-completed stocks
    to_process = [c for c in tickers if not already_done(str(c), output_dir)]
    skipped = len(tickers) - len(to_process)
    if skipped > 0:
        print(f"Skipping {skipped} already-processed stocks.")

    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {
            executor.submit(
                process_one_stock_tailabsorb,
                permno=str(c),
                returns=returns_matrix[c],
                month_end_trd=month_end_trd,
                market_fit=market_fit,
                month_end_idx=month_end_idx,
                output_dir=output_dir,
                span=SPAN,
                n0_default=N0_DEFAULT,
                hazard=HAZARD,
                max_run=MAX_RUN,
                tau_shrink=TAU_SHRINK,
            ): c
            for c in to_process
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing stocks (resumable, MLE, as-of top-1000, summaries v2)"):
            c = futures[future]
            try:
                df = future.result()
                if df is not None:
                    results.append(df)
            except Exception as e:
                errors.append((c, repr(e)))

    if errors:
        print(f"{len(errors)} workers failed (showing up to 5): {errors[:5]}")

    # consolidation into a single HDF (single-threaded IO)
    if write_master:
        print("Consolidating per-stock files into master HDF (may take time)...")
        frames = []
        for c in tqdm(tickers, desc="Reading per-stock files"):
            path = stock_out_path(str(c), output_dir)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    di = pd.read_hdf(path, key=_base_key_from_path(path))
                    frames.append(di)
                except Exception:
                    pass
        if frames:
            master_df = pd.concat(frames, ignore_index=True).sort_values(["date", "permno"])
            master_key = _base_key_from_path(master_hdf)  # ensure key == filename
            os.makedirs(os.path.dirname(master_hdf), exist_ok=True)
            master_df.to_hdf(master_hdf, key=master_key, mode="w")
            return master_df

    if results:
        return pd.concat(results, ignore_index=True).sort_values(["date", "permno"])
    else:
        # Return an empty DF with all expected columns
        return pd.DataFrame(columns=[
            "date","permno",
            "p_t","E_rt","dE_rt","P_r_le_5","log_pred",
            "P_r_le_10","P_r_le_14","P_r_le_21","P_r_le_42",
            "P_r_le_63","P_r_le_126","P_r_le_252",
            "H_rt","Var_rt","r_med",
            "pred_mu_1d_prior","pred_var_1d_prior",
            "pred_mu_1d_post","pred_var_1d_post","pred_sr_1d",
            "E21","V21","V21_ms","SR21",
            "std_surprise2",
            "mean_log_pred_21","mean_std_surprise2_21"
        ])

if __name__ == "__main__":
    try:
        print("N_JOBS:", N_JOBS)
        features_df = run_pipeline_resumable(
            input_hdf=INPUT_HDF,
            key=KEY,
            output_dir=OUTPUT_DIR,
            write_master=True,
            master_hdf=MASTER_OUT_HDF,
            master_key=MASTER_OUT_KEY,
        )
        print("Done. Sample:")
        print(features_df.head(10))
    except Exception as e:
        print("FATAL:", repr(e))
        sys.exit(1)
