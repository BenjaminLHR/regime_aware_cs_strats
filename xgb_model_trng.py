from __future__ import annotations

from typing import Dict, List, Tuple
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


model_description = 'mlogloss_mom_vol_bocd'


XGB_PARAMS_MC_A = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.015,
    max_depth=2,
    min_child_weight=160,
    gamma=8.0,
    subsample=0.45,
    colsample_bytree=0.35,
    colsample_bylevel=0.55,
    colsample_bynode=0.55,
    reg_alpha=1.20,
    reg_lambda=25.0,
    max_delta_step=3,
    tree_method="hist", max_bin=24,
    seed=44,
)

XGB_PARAMS_MC_B = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.018,
    max_depth=2,
    min_child_weight=140,
    gamma=7.0,
    subsample=0.48,
    colsample_bytree=0.40,
    colsample_bylevel=0.60,
    colsample_bynode=0.60,
    reg_alpha=1.00,
    reg_lambda=22.0,
    max_delta_step=3,
    tree_method="hist", max_bin=28,
    seed=44,
)

XGB_PARAMS_MC_C = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.020,
    max_depth=2,
    min_child_weight=120,
    gamma=6.0,
    subsample=0.50,
    colsample_bytree=0.42,
    colsample_bylevel=0.62,
    colsample_bynode=0.62,
    reg_alpha=0.85,
    reg_lambda=18.0,
    max_delta_step=2,
    tree_method="hist", max_bin=32,
    seed=44,
)

XGB_PARAMS_MC_D = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.022,
    max_depth=3,
    min_child_weight=90,
    gamma=5.5,
    subsample=0.53,
    colsample_bytree=0.46,
    colsample_bylevel=0.66,
    colsample_bynode=0.66,
    reg_alpha=0.70,
    reg_lambda=15.0,
    max_delta_step=2,
    tree_method="hist", max_bin=56,
    seed=44,
)

XGB_PARAMS_MC_E = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.025,
    max_depth=3,
    min_child_weight=70,
    gamma=5.0,
    subsample=0.58,
    colsample_bytree=0.48,
    colsample_bylevel=0.68,
    colsample_bynode=0.68,
    reg_alpha=0.60,
    reg_lambda=13.0,
    max_delta_step=2,
    tree_method="hist", max_bin=60,
    seed=44,
)

XGB_PARAMS_MC_F = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.030,
    max_depth=3,
    min_child_weight=60,
    gamma=4.0,
    subsample=0.60,
    colsample_bytree=0.50,
    colsample_bylevel=0.70,
    colsample_bynode=0.70,
    reg_alpha=0.50,
    reg_lambda=12.0,
    max_delta_step=2,
    tree_method="hist", max_bin=64,
    seed=44,
)

XGB_PARAMS_MC_G = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.032,
    max_depth=4,
    min_child_weight=45,
    gamma=3.0,
    subsample=0.65,
    colsample_bytree=0.60,
    colsample_bylevel=0.75,
    colsample_bynode=0.75,
    reg_alpha=0.20,
    reg_lambda=8.0,
    max_delta_step=1,
    tree_method="hist", max_bin=80,
    seed=44,
)

XGB_PARAMS_MC_H = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.028,
    max_depth=3,
    min_child_weight=75,
    gamma=4.5,
    subsample=0.58,
    colsample_bytree=0.50,
    colsample_bylevel=0.70,
    colsample_bynode=0.70,
    reg_alpha=0.60,
    reg_lambda=13.0,
    max_delta_step=2,
    tree_method="hist", max_bin=56,
    seed=44,
)

XGB_PARAMS_MC_I = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.030,
    max_depth=3,
    min_child_weight=65,
    gamma=4.2,
    subsample=0.60,
    colsample_bytree=0.50,
    colsample_bylevel=0.70,
    colsample_bynode=0.70,
    reg_alpha=0.55,
    reg_lambda=12.0,
    max_delta_step=2,
    tree_method="hist", max_bin=60,
    seed=44,
)

XGB_PARAMS_MC_J = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.030,
    max_depth=3,
    min_child_weight=60,
    gamma=4.0,
    subsample=0.60,
    colsample_bytree=0.50,
    colsample_bylevel=0.70,
    colsample_bynode=0.70,
    reg_alpha=0.50,
    reg_lambda=12.0,
    max_delta_step=2,
    tree_method="hist", max_bin=64,
    seed=44,
)

XGB_PARAMS_MC_K = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.032,
    max_depth=4,
    min_child_weight=45,
    gamma=3.0,
    subsample=0.65,
    colsample_bytree=0.60,
    colsample_bylevel=0.75,
    colsample_bynode=0.75,
    reg_alpha=0.20,
    reg_lambda=8.0,
    max_delta_step=1,
    tree_method="hist", max_bin=80,
    seed=44,
)

XGB_PARAMS_MC_L = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.035,
    max_depth=4,
    min_child_weight=35,
    gamma=2.0,
    subsample=0.70,
    colsample_bytree=0.70,
    colsample_bylevel=0.80,
    colsample_bynode=0.80,
    reg_alpha=0.10,
    reg_lambda=4.0,
    max_delta_step=1,
    tree_method="hist", max_bin=96,
    seed=44,
)

XGB_PARAMS_MC_M = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.045,
    max_depth=5,
    min_child_weight=20,
    gamma=1.0,
    subsample=0.80,
    colsample_bytree=0.80,
    colsample_bylevel=0.90,
    colsample_bynode=0.80,
    reg_alpha=0.05,
    reg_lambda=3.0,
    max_delta_step=1,
    tree_method="hist", max_bin=128,
    seed=44,
)

XGB_PARAMS_MC_N = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.060,
    max_depth=6,
    min_child_weight=10,
    gamma=0.5,
    subsample=0.90,
    colsample_bytree=0.90,
    colsample_bylevel=0.90,
    colsample_bynode=0.90,
    reg_alpha=0.02,
    reg_lambda=2.0,
    max_delta_step=1,
    tree_method="hist", max_bin=256,
    seed=44,
)

XGB_PARAMS_MC_O = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    eta=0.040,
    grow_policy="lossguide",
    max_depth=0,
    max_leaves=40,
    min_child_weight=18,
    gamma=1.5,
    subsample=0.80,
    colsample_bytree=0.80,
    colsample_bylevel=0.80,
    colsample_bynode=0.80,
    reg_alpha=0.10,
    reg_lambda=4.0,
    max_delta_step=1,
    tree_method="hist", max_bin=128,
    seed=44,
)

PARAM_GRID = {
    "A": XGB_PARAMS_MC_A,
    "B": XGB_PARAMS_MC_B,
    "C": XGB_PARAMS_MC_C,
    "D": XGB_PARAMS_MC_D,
    "E": XGB_PARAMS_MC_E,
    "F": XGB_PARAMS_MC_F,
    "G": XGB_PARAMS_MC_G,
    "H": XGB_PARAMS_MC_H,
    "I": XGB_PARAMS_MC_I,
    "J": XGB_PARAMS_MC_J,
    "K": XGB_PARAMS_MC_K,
    "L": XGB_PARAMS_MC_L,
    "M": XGB_PARAMS_MC_M,
    "N": XGB_PARAMS_MC_N,
    "O": XGB_PARAMS_MC_O,
}



# For GPU config
for cfg in PARAM_GRID.values():
    cfg["tree_method"] = "gpu_hist"
    cfg["predictor"] = "gpu_predictor"



returns_matrix = pd.read_csv(
    'data/returns_matrix.csv',
    index_col='date',
    parse_dates=['date']
)

returns_matrix.columns = returns_matrix.columns.astype(int)
sigma = pd.read_hdf("features/ewma_std_span_63.h5", "ewma_std_span_63")
meta_cols = ["date", "permno", "fwd_ret_div_sigma_lag_63", "decile_fwd_ret_div_sigma_lag_63"]
bocd_cols = [
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
    'log_pred',
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
    'log_pred_z',
    'H_rt_z',
    'Var_rt_z',
    'r_med_z',
]


ltr_df = pd.read_hdf('features/ltr_df.h5', key='ltr_df')

cols_to_select = [c for c in ltr_df.columns if c in meta_cols] + [c for c in ltr_df.columns if '_z' in c[-2:]] + bocd_cols
cols_to_select = list(set(cols_to_select))

ltr_df = ltr_df[cols_to_select]


'''
XGB long/short backtest pipeline
Assumes train_test_lsts: list of [(trn_start, trn_end), (tst_start, tst_end)] per window.
For each training window:
  1) Take the earliest 80% of dates as sub-train.
  2) Skip the next 21 calendar days (gap).
  3) Use the remainder as validation and early-stop on validation.
  4) Retrain on the full training window using the best iteration.
'''


#  parameters
TAIL_FRAC          = 0.10 # ≈ top / bottom decile for class-label
USE_CLASS_WEIGHTS  = True
TRAIN_FRAC         = 0.80 # share of rows that go into sub-train
GAP_DAYS           = 21
NUM_BOOST_ROUNDS   = 10000
EARLY_STOP_ROUNDS  = 200

def to_dt_lsts(
    raw_lsts: List[List[Tuple[str, str]]]
) -> List[List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Convert nested (start, end) date strings to Timestamps; preserve nesting."""
    return [
        [(pd.to_datetime(a), pd.to_datetime(b)) for (a, b) in period]
        for period in raw_lsts
    ]

def make_class3(
    df: pd.DataFrame,
    date_col: str = "date",
    ret_col: str = "fwd_ret_cs",
    frac: float = TAIL_FRAC,
) -> pd.DataFrame:
    out = df.copy()
    out["class3"] = 1  # default (middle)
    for _, g in out.groupby(date_col):
        n = len(g)
        if n == 0:
            continue
        k = max(1, int(round(frac * n)))
        g_sorted = g.sort_values(ret_col, ascending=False)
        top_idx  = g_sorted.head(k).index
        bot_idx  = g_sorted.tail(k).index
        out.loc[top_idx, "class3"] = 2
        out.loc[bot_idx, "class3"] = 0
    return out

# split sub-train / validation inside ONE training window
def split_train_valid(
    window_df: pd.DataFrame,
    date_col: str  = "date",
    train_frac: float = TRAIN_FRAC,
    gap_days: int = GAP_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split: earliest train_frac share of dates is train; skip gap_days; remainder is validation."""
    window_df = window_df.sort_values(date_col)
    uniq_dates = window_df[date_col].unique()
    cut_idx    = int(np.floor(len(uniq_dates) * train_frac)) # 80 % date
    train_end  = pd.to_datetime(uniq_dates[cut_idx - 1])
    gap_end    = train_end + pd.Timedelta(days=gap_days)
    # masks
    trn_mask   = window_df[date_col] <= train_end
    val_mask   = (window_df[date_col] >  gap_end)
    return window_df[trn_mask].copy(), window_df[val_mask].copy()

# 3)  optional class-weight mapping 
def add_class_weights(
    df_trn: pd.DataFrame,
    df_val: pd.DataFrame,
    label_col: str = "class3",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inverse-frequency weights, mean ≈ 1."""
    cls_cnt  = df_trn[label_col].value_counts().reindex([0, 1, 2]).fillna(0)
    inv_freq = cls_cnt.sum() / (3 * cls_cnt.replace(0, np.nan))
    w_map    = inv_freq.fillna(0).to_dict()
    df_trn["sample_weight"] = df_trn[label_col].map(w_map)
    df_val["sample_weight"] = df_val[label_col].map(w_map)
    return df_trn, df_val

# single-window train → best_iter → retrain on whole window 
def train_one_window(
    train_df: pd.DataFrame,
    params: Dict = None,
    feature_cols: List[str] = None,
    use_class_weights: bool = USE_CLASS_WEIGHTS,
    col_for_regime_based_weighting=None,
    num_boost_round: int = NUM_BOOST_ROUNDS,
    early_stop_rounds: int = EARLY_STOP_ROUNDS,
) -> xgb.Booster:
    """
    Train one XGBoost window with early stopping on an internal train/valid split,
    then retrain on all rows using the best iteration. Imposes a monotonicity
    constraint of -1 (non-increasing) on 'sigma_lag_span_63_z'.
    """
     
    label_col = "class3"
    min_weight = 1e-2

    assert feature_cols and len(feature_cols) > 0, "feature_cols must be provided"
    if SIGMA_FEAT not in feature_cols:
        raise ValueError(f"{SIGMA_FEAT} must be present in feature_cols to apply the constraint")

    # Inject monotonicity constraint for SIGMA_FEAT (-1); others 0.
    params_local = dict(params or {})
    mono_vec = [-1 if f == SIGMA_FEAT else 0 for f in feature_cols]
    # Only set if caller didn't already provide one
    params_local.setdefault("monotone_constraints", "(" + ",".join(map(str, mono_vec)) + ")")

    # Create sub-train/validation split for early stopping.
    df_subtrn, df_val = split_train_valid(train_df)
    if use_class_weights:
        df_subtrn, df_val = add_class_weights(df_subtrn, df_val, label_col)
    else:
        df_subtrn["sample_weight"] = 1.0
        df_val["sample_weight"] = 1.0

    # Optional regime-based weighting
    if col_for_regime_based_weighting is not None:
        df_subtrn["sample_weight"] *= (min_weight + df_subtrn[col_for_regime_based_weighting])
        df_val["sample_weight"] *= (min_weight + df_val[col_for_regime_based_weighting])

    df_subtrn.dropna(subset=["sample_weight"], inplace=True)
    df_val.dropna(subset=["sample_weight"], inplace=True)

    # Fit early-stopped model on sub-train; evaluate on validation.
    dtrn = xgb.DMatrix(
        df_subtrn[feature_cols],
        label=df_subtrn[label_col],
        weight=df_subtrn["sample_weight"]
    )
    dval = xgb.DMatrix(
        df_val[feature_cols],
        label=df_val[label_col],
        weight=df_val["sample_weight"]
    )
    evals = [(dtrn, "train"), (dval, "valid")]

    booster_es = xgb.train(
        params_local,
        dtrn,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stop_rounds,
        verbose_eval=50,
    )
    best_iter = booster_es.best_iteration  # 0-based

    # retrain on all rows with best_iter 
    if use_class_weights:
        train_df, _ = add_class_weights(train_df, train_df, label_col)
    else:
        train_df["sample_weight"] = 1.0

    if col_for_regime_based_weighting is not None:
        train_df["sample_weight"] *= (min_weight + train_df[col_for_regime_based_weighting])

    train_df.dropna(subset=["sample_weight"], inplace=True)

    dall = xgb.DMatrix(
        train_df[feature_cols],
        label=train_df[label_col],
        weight=train_df["sample_weight"],
    )

    booster_full = xgb.train(
        params_local,
        dall,
        num_boost_round=best_iter + 1,  # best_iteration is 0-based
        evals=[(dall, "all")],
        verbose_eval=50,
    )

    booster_full.set_attr(
        best_iteration=str(best_iter),
        best_score=str(booster_es.best_score),
    )
    return booster_full


# predict probabilities + derive scores 
def predict_scores(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: List[str],
    suffix: str,
) -> pd.DataFrame:
    """Return a DF with p_*_<suffix> and net_score_<suffix>."""
    probs = model.predict(xgb.DMatrix(df[feature_cols])) # ndarray (n,3)
    base_cols = ["p_bottom", "p_middle", "p_top"]
    tmp = pd.DataFrame(
        probs,
        index=df.index,
        columns=[f"{c}_{suffix}" for c in base_cols], # add suffix
    )
    tmp[f"net_score_{suffix}"] = tmp[f"p_top_{suffix}"] - tmp[f"p_bottom_{suffix}"]
    return df[["date", "permno"]].join(tmp)

def merge_preds_into_backtest(
    backtest_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    id_cols = ("date", "permno"),
) -> pd.DataFrame:
    bt = backtest_df.copy()

    value_cols = [c for c in preds_df.columns if c not in id_cols]
    for col in value_cols:
        if col not in bt.columns:
            bt[col] = np.nan

    bt_idxed   = bt.set_index(list(id_cols))
    preds_idxed = preds_df.set_index(list(id_cols))

    bt_idxed.update(preds_idxed[value_cols])

    return bt_idxed.reset_index()

def add_hmm_features_inplace(cols_to_add, dfs_to_receive_cols, hmm_dfs, window_id):
    for c in cols_to_add:
        for df in dfs_to_receive_cols:
            df[c] = df['date'].map(hmm_dfs[window_id].set_index('date')[c])

train_test_lsts = [
    [
        ("1999-01-01", "2004-11-25"),
        ("2004-12-01", "2009-11-30"),
    ],
    [
        ("1999-01-01", "2009-11-25"),
        ("2009-12-01", "2014-11-30"),
    ],
    [
        ("1999-01-01", "2014-11-25"),
        ("2014-12-01", "2019-11-30"),
    ],
    [
        ("1999-01-01", "2019-11-25"),
        ("2019-12-01", "2024-12-31"),
    ],
]

SIGMA_FEAT = "sigma_lag_span_63_z"

feature_cols = [
    'sigma_lag_span_63_z',
    
    # _z refers to winsorizing, then z-scoring (subtract mean return and divide by std of that day)
    # short & medium momentum
    '21_day_raw_ret_z',
    '63_day_raw_ret_z',

    # long momentum
    '252_day_raw_ret_z',

    # vol-and-time-adj the above
    '21_day_vol_time_scaled_ret_z',
    '63_day_vol_time_scaled_ret_z',
    '252_day_vol_time_scaled_ret_z',

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
    # # Baz composite MACD score as per Daniel's papers
    # # 'baz_comp_z',

    # Volume average
    'avg_turnover_of_past_21_days_z',
    'avg_turnover_of_past_63_days_z',
    'avg_turnover_of_past_126_days_z',
    'avg_turnover_of_past_252_days_z',

    # Volume betas and R2
    'beta_turnover_wrt_day_21_days_z',
    'beta_turnover_wrt_day_63_days_z',
    'beta_turnover_wrt_day_126_days_z',
    'beta_turnover_wrt_day_252_days_z',

    'r2_turnover_wrt_day_63_days_z',
    'r2_turnover_wrt_day_21_days_z',
    'r2_turnover_wrt_day_126_days_z',
    'r2_turnover_wrt_day_252_days_z',

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


print(feature_cols)

ltr_df = make_class3(ltr_df, ret_col='fwd_ret_div_sigma_lag_63', frac=TAIL_FRAC)
backtest_df = ltr_df[["date", "permno", "fwd_ret_div_sigma_lag_63", "decile_fwd_ret_div_sigma_lag_63", 'class3']].copy()
    
for window_id, two_tuple in enumerate(train_test_lsts):
    (trn_start, trn_end), (tst_start, tst_end) = two_tuple

    mask_trn = (ltr_df["date"] >= trn_start) & (ltr_df["date"] <= trn_end)
    df_trn_win = ltr_df[mask_trn].copy()

    mask_test = (ltr_df["date"] >= tst_start) & (ltr_df["date"] <= tst_end)
    df_test_win = ltr_df[mask_test].copy()

    if df_trn_win.empty:
        print(f"Window {window_id}: no train rows → skipped.")
        continue

    best_booster = None
    best_score = float("inf")
    best_id = None
    best_iter = None

    for cfg_id, cfg in PARAM_GRID.items():
        booster = train_one_window(
            df_trn_win,
            params=cfg,
            feature_cols=feature_cols,
            use_class_weights=USE_CLASS_WEIGHTS,
        )
        iters = booster.best_iteration + 1
        score = float(booster.best_score)

        print(f"Param results: ({cfg_id!r}, {iters}, {score:.5f})")

        if score < best_score:
            best_score = score
            best_booster = booster
            best_id = cfg_id
            best_iter = iters

    print(
        f"Window {window_id}: picked config {best_id} "
        f"(best_iter={best_iter}, valid_mlogloss={best_score:.5f})"
    )

    pred_trn = predict_scores(best_booster, df_trn_win, feature_cols, suffix="trn")
    backtest_df = merge_preds_into_backtest(backtest_df, pred_trn)

    if not df_test_win.empty:
        pred_test = predict_scores(best_booster, df_test_win, feature_cols, suffix="test")
        backtest_df = merge_preds_into_backtest(backtest_df, pred_test)

    print(
        f"Window {window_id}: "
        f"fit rows {len(df_trn_win):,} | "
        f"test rows {len(df_test_win):,}"
    )


# 3-day weighted-average signal
WEIGHTS_3D = np.array([0.5, 0.3, 0.2])
backtest_df = backtest_df.sort_values(["permno", "date"])
g = backtest_df.groupby("permno")["net_score_test"]
backtest_df["net_score_test_3d_wa"] = (
    WEIGHTS_3D[0] * g.shift(0).fillna(0)
    + WEIGHTS_3D[1] * g.shift(1).fillna(0)
    + WEIGHTS_3D[2] * g.shift(2).fillna(0)
)

try:
    name = f'backtest_df_{model_description}'  # key == filename
    backtest_df.to_hdf(f'results/{name}.h5', key=name, mode='w',
                    format='fixed', complevel=9, complib='zlib')
except Exception as e:
    print(f"Error saving backtest DataFrame: {e}")