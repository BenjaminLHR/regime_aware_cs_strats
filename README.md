# Regime-Aware Cross-Sectional Equity Selection — Per-Stock Bayesian Online Changepoint Features Improve Momentum, Low-Minus-High Volatility and Volume Strategies
_Per‑stock Bayesian Online Changepoint Detection (BOCD), Momentum, Volatility, Volume features fed to an XGBoost model; monthly long–short backtest with realistic costs/hedging._

This repository contains the code and notebooks accompanying the dissertation:

> **Regime‑Aware Cross‑Sectional Equity Selection — Per‑Stock Bayesian Online Changepoint Features Improve Momentum, Low‑Minus‑High Volatility and Volume Strategies** (Oxford, 2025).

At a high level, we engineer **online regime features** per stock using **Bayesian Online Changepoint Detection (BOCD)** and feed them—together with standard momentum/volume/volatility blocks—into an **XGBoost** learner that predicts a 3‑class cross‑sectional target. We then form monthly long–short portfolios (vol‑targeted), evaluate pre/post costs and optional beta‑hedging, and probe model behaviour via unconditional correlation tests, **TreeSHAP**, grouped permutation, and slice tests.

---

## Contents

```
├─ data_collection_and_feature_eng.ipynb        # Stage 1.1 — CRSP ingest + core features → base tables
├─ generate_bocd_feats.py                        # Stage 1.2 — per‑stock BOCD features (Parallelized on CPU and resumable)
├─ load_bocd_features_into_ltr_df.ipynb          # Stage 1.3 — merge BOCD into learning table (ltr_df)
├─ xgb_model_trng.py                             # Stage 2   — train XGBoost (GPU recommended/used)
├─ flexi_p_scaling_and_eval_metrics.ipynb        # Stage 3   — portfolio, backtests, metrics
├─ shap_mom_bocd_for_slurm.py                    # Stage 4   — SHAP + grouped permutation (GPU)
├─ shap_mom_bocd.ipynb                           # Stage 4   — SHAP analysis notebook
├─ slice_test_mom_bocd.ipynb                     # Stage 4   — slice tests (e.g., momentum×BOCD gates)
```

> **CRSP data notice** — This project requires access to CRSP via your own institutional license. **No CRSP data—raw or processed—is distributed here.** See _Data & Licensing_ below before running notebooks.

---

## Quick start (end-to-end)

### 0) Environment

- **Python** 3.11 (pinned in `regime_cs.yml`)
- Optional but STRONGLY recommended: an NVIDIA GPU with recent CUDA drivers for fast XGBoost training/SHAP.

Create and activate the environment from the YAML:

```bash
# from the repo root
conda env create --file regime_cs.yml --name regime_cs
conda activate regime_cs
```

> If your CUDA stack is non‑standard, consult the XGBoost docs for GPU wheels or build from source.

---

## Stage 1 — Build the learning table `[features, target]`

### 1.1  Data collection & base features
Open **`data_collection_and_feature_eng.ipynb`** and run cells top‑to‑bottom. This notebook:
- Pulls **CRSP** daily data (RET, PRC, VOL, SHROUT, etc.) and point‑in‑time descriptors.
- Applies investability filters (common shares on NYSE/AMEX/NASDAQ).
- Builds core price/return constructs (split‑consistent prices, forward 21‑day returns, EWMA σ with span=63) and momentum/volume blocks.
- Produces base tables used downstream.  
**End goal**: an intermediate table ready to accept BOCD features and generate **`ltr_df`** (“learning‑to‑rank” table. Variable name is outdated but retained as initially we wanted to adopt learning-to-rank algorithms instead of our 3-class classification).

> ⚠️ **CRSP licensing**: do not export or redistribute any panel constructed here. Keep all outputs on licensed systems only.

### 1.2  Per‑stock BOCD features (Parallelized on CPU and resumable)
Run **`generate_bocd_feats.py`** from the project root. It detects cores and uses **N‑1** threads, processes each PERMNO independently, and writes per‑stock HDFs plus a consolidated master. Default inputs/outputs are defined in the script header.

```bash
python generate_bocd_feats.py
```

Key behaviours:
- Online BOCD with constant hazard (default `h=0.005`), Student‑t emission with monthly shrinkage of scale/tails to an as‑of **top‑1000** market‑cap anchor (mean left free).
- First valid features appear after 252 trading days to align with other windows.
- Outputs include: instantaneous break risk `p_t`, expected/median run‑length (`E_rt`, `r_med`), multi‑horizon cumulative break risk `Pr_le_k` for `k∈{5,10,14,21,42,63,126,252}`, posterior entropy/variance, plus predictive and 21‑day hazard‑aware proxies. Z‑scoring happens downstream.
- Paths to adjust in the script:
  - `INPUT_HDF`, `KEY`: returns matrix source
  - `TOPK_MASK_HDF/KEY`: as‑of top‑1000 mask for shrinkage
  - `OUTPUT_DIR`: per‑stock feature dumps
  - `MASTER_OUT_HDF/KEY`: consolidated feature store

### 1.3  Merge BOCD into the learning table
Open **`load_bocd_features_into_ltr_df.ipynb`** and run. It merges the consolidated BOCD features into the base table from 1.1 and materialises the final **`ltr_df`** HDF:

- **File**: `features/ltr_df.h5`
- **Key**: `ltr_df`

---

## Stage 2 — Train XGBoost (GPU recommended)

We provide a script **`xgb_model_trng.py`** with multiple parameter sets (A–O). It is configured to use **`tree_method="gpu_hist"`** and **`predictor="gpu_predictor"`** by default. You can start with the **Momentum+Volatility+BOCD** specification (make changes to feature_cols to get other specifications).

```bash
python xgb_model_trng.py
```

Tips:
- Ensure your `ltr_df` paths in the script/notebooks match your environment.
- Stick to conservative depth (2–4) to control variance; multi‑class objective is `multi:softprob` with 3 classes.

---

## Stage 3 — Portfolio, backtests, and metrics

Open **`flexi_p_scaling_and_eval_metrics.ipynb`**:

- Forms monthly long–short portfolios from the model’s net score (P[Long]−P[Short]).
- Supports equal‑risk and unequal‑risk allocations within legs.
- Computes Sharpe pre/post **transaction costs** and after an optional **beta‑hedge**.
- Produces summary tables/plots used in the dissertation (e.g., pre/post‑cost Sharpe, ∆Sharpe from BOCD, P&L curves).

---

## Stage 4 — Interpretability Tests Demonstrated on Mom+BOCD Models (easily modifiable for other specifications)

### Script (GPU): `shap_mom_bocd_for_slurm.py`
Runnable on a single machine or a SLURM array (`--window`/`SLURM_ARRAY_TASK_ID` to pick OOS window). It:

- Loads `ltr_df`, rebuilds 3‑class labels (top/bottom deciles per date), and reuses the trained booster (or trains if missing).
- Computes **TreeSHAP** contributions and **grouped within‑date permutation** deltas (weighted mlogloss, date‑balanced rank‑IC, Top–Bottom spread), saving CSVs/plots under `OUT_DIR`.
- Environment variables you can set: `OUT_DIR`, `MODEL_DIR`, `TAIL_FRAC`, `PY_SEED`, `N_PERM`, `BATCH`, `NTHREADS`, `EVAL_WEIGHT_SOURCE`.
- Predefined OOS windows: four rolling splits across 2004–2024 (see script constants).

**Reason to use GPU** Large, date‑balanced cross‑sections (many stocks × many months) make SHAP computation expensive. Using XGBoost’s **GPUTreeShap** via `predict(pred_contribs=True)` (enabled when the model uses `gpu_hist`/`gpu_predictor`) accelerates SHAP attribution **and** batch probability scoring by orders of magnitude, keeping wall‑clock and memory within practical limits on modern GPUs.

### Notebooks
- **`shap_mom_bocd.ipynb`**: interactive exploration of SHAP summaries (family‑level and feature‑level), signed contributions, and consistency checks.
- **`slice_test_mom_bocd.ipynb`**: conditional analyses (e.g., Momentum×BOCD gates; within‑bucket paired differences).

---

## Data & licensing

- **Data source:** CRSP (via WRDS). You must have your **own** license and credentials.
- **Redistribution:** _Prohibited._ Do **not** share any raw or derived CRSP panels, even if transformed. Keep all outputs on licensed systems.
- **Universe:** U.S. common shares (SHRCD 10/11) on NYSE/AMEX/NASDAQ; investable top‑~1000 by market‑cap each day; monthly rebalancing; 21‑day forward, volatility‑scaled target.

---

## Configuration: paths & keys

Adjust these in scripts as needed:

- **BOCD features (`generate_bocd_feats.py`)**
  - `INPUT_HDF = "data/returns_matrix_core_stocks.h5"` (key: `returns_matrix_core_stocks`)
  - `TOPK_MASK_HDF = "features/top_1000_for_bocd_regularization.h5"` (key: file‑name‑based)
  - `OUTPUT_DIR = "features/bocd_per_stock_feats"`
  - `MASTER_OUT_HDF = "features/bocd_per_stock_feats.h5"` (key: file‑name‑based)
  - Auto parallelism: uses `CPU_COUNT-1` workers.

- **SHAP/permutation (`shap_mom_bocd_for_slurm.py`)**
  - `LTR_HDF_PATH = "features/ltr_df.h5"` (key: `ltr_df`)
  - Pre‑selected **feature lists**, **groups**, **windows**, and **rounds** are defined at the top of the script.

- **XGBoost training (`xgb_model_trng.py`)**
  - Parameter grids **A–O** with GPU settings; ensure file paths (returns, σ, BOCD features) point to your local stores.

---

## Reproducibility notes

- **Rolling splits:** four out‑of‑sample windows spanning ~2004–2024 are pre‑defined. Use the same windows for training, scoring, and SHAP/permutation for comparability.
- **Weights:** inverse‑frequency class weights and **date‑balanced** evaluation (weighted mlogloss) are used consistently between training and evaluation.
- **Randomness:** set `PY_SEED` for SHAP/permutation reproducibility; XGBoost seeds are defined in the param grids.

---

## Troubleshooting

- **CRSP paths/keys not found** → Verify HDF file names and keys. The BOCD script expects a dense returns matrix and a top‑1000 mask HDF.
- **Out-of-memory (OOM) during feature engineering** →  Close other existing programs (restarting machine helps too), delete large dataframes in current notebook environment.
- **CPU thrash during BOCD** → The script already sets `OMP_NUM_THREADS=1` etc. If running inside BLAS‑heavy environments, double‑check these env vars.
- **GPU not used** → Ensure `xgboost` is a GPU‑enabled build and that the model params include `tree_method="gpu_hist"` and `predictor="gpu_predictor"`.
- **SHAP too slow / OOM** → Reduce `BATCH`, lower `N_PERM`, or run per‑window on separate GPUs. Ensure GPUTreeShap is engaged (same GPU settings as training).

---

## Citing

To be completed at a later date if submitted for publication.

---
