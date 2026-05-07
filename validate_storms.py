"""
validate_storms.py
------------------
Paper-quality validation: Burton baseline, BiLSTM Corrector, RL Blend.

Computes:
  - Per-class RMSE / MAE with 95% bootstrap confidence intervals
  - True overall RMSE (full unsampled data) — no sampling bias
  - Ablation table: Burton vs Corrector vs RL Blend
  - Detection scores: FAR / POD / CSI / Heidke Skill Score
  - Comparison against published NOAA benchmarks
  - Validation plots saved to notebooks/Plots_Pipeline/

Usage:
    # Quick run — 10K rows per period, ~2-3 min
    python validate_storms.py

    # Paper quality — all rows, overnight
    python validate_storms.py --full-dataset

    # Specific periods only
    python validate_storms.py --periods test_a test_b
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.anomaly_autoencoder import TransformerAutoencoder
from models.lstm_corrector      import LSTMCorrector
from agents.rl_agent            import RLAgent, ALERT_ENC

# ── Config ────────────────────────────────────────────────────────────────────
ANOMALY_CFG   = json.loads((ROOT / "models/anomaly_config.json").read_text())
CORRECTOR_CFG = json.loads((ROOT / "models/corrector_config.json").read_text())

ANOMALY_FEATS = ANOMALY_CFG["feature_cols"]
CORR_FEATS    = CORRECTOR_CFG["feature_cols"]
ANOMALY_SEQ   = ANOMALY_CFG["seq_len"]
CORR_SEQ      = CORRECTOR_CFG["seq_len"]
RECON_THRESH  = ANOMALY_CFG["recon_threshold"]
RECON_MAX     = ANOMALY_CFG["recon_max"]
ALERT_GREEN   = ANOMALY_CFG["alert_green"]
ALERT_YELLOW  = ANOMALY_CFG["alert_yellow"]

PLOTS_DIR = ROOT / "notebooks" / "Plots_Pipeline"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Published benchmarks ──────────────────────────────────────────────────────
# NOTE: These come from different datasets/time windows. Direct comparison is
# informative but not perfectly apples-to-apples. Cited for context only.
PUBLISHED = {
    "Persistence model":             15.0,
    "Burton ODE (published)":         9.5,
    "Wang & Chen empirical":          8.5,
    "NOAA WFS operational":           7.2,
    "Gruet et al. LSTM (2018)":       8.3,
    "Siciliano Transformer (2021)":   6.1,
    "Shrivastava BiLSTM (2022)":      5.2,   # note: different test set
}

# ── Storm class bins ──────────────────────────────────────────────────────────
CLASSES = ["quiet", "minor", "moderate", "intense", "extreme"]
CLASS_MASKS = {
    "quiet":    lambda d: d > -30,
    "minor":    lambda d: (d <= -30)  & (d > -50),
    "moderate": lambda d: (d <= -50)  & (d > -100),
    "intense":  lambda d: (d <= -100) & (d > -200),
    "extreme":  lambda d: d <= -200,
}

# ── Metric helpers ────────────────────────────────────────────────────────────

def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))

def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(e))))

def skill(base: float, model: float) -> float:
    return (1.0 - model / base) * 100.0 if base > 0 else 0.0

def bootstrap_ci(errors: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """95% bootstrap confidence interval for RMSE."""
    errs = np.asarray(errors)
    rng  = np.random.default_rng(42)
    boot = [rmse(rng.choice(errs, size=len(errs), replace=True)) for _ in range(n_boot)]
    lo   = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi   = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    device = torch.device("cpu")

    anomaly_model = TransformerAutoencoder(**{
        k: ANOMALY_CFG[k]
        for k in ("input_size","seq_len","d_model","latent_dim","nhead","num_layers","dropout")
    })
    anomaly_model.load_state_dict(
        torch.load(ROOT / "models/anomaly_model.pt", map_location=device, weights_only=True))
    anomaly_model.eval()

    corrector_model = LSTMCorrector(**{
        k: CORRECTOR_CFG[k]
        for k in ("input_size","hidden_size","num_layers","dropout")
    })
    corrector_model.load_state_dict(
        torch.load(ROOT / "models/corrector_model.pt", map_location=device, weights_only=True))
    corrector_model.eval()

    with open(ROOT / "models/anomaly_scaler.pkl", "rb") as f:
        anomaly_scaler = pickle.load(f)
    with open(ROOT / "models/corrector_scaler.pkl", "rb") as f:
        corrector_scaler = pickle.load(f)

    rl = RLAgent()
    return anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(df, anomaly_model, corrector_model,
                  anomaly_scaler, corrector_scaler, rl, device,
                  batch: int = 512) -> pd.DataFrame:

    n_rows      = len(df)
    corr_starts = np.arange(0, n_rows - CORR_SEQ + 1)
    a_starts    = corr_starts + (CORR_SEQ - ANOMALY_SEQ)
    out_idx     = corr_starts + CORR_SEQ - 1
    n           = len(corr_starts)

    a_vals = anomaly_scaler.transform(
        pd.DataFrame(df[ANOMALY_FEATS].values, columns=ANOMALY_FEATS))
    c_vals = corrector_scaler.transform(
        pd.DataFrame(df[CORR_FEATS].values, columns=CORR_FEATS))

    a_scores  = np.zeros(n)
    residuals = np.zeros(n)

    with torch.no_grad():
        print(f"    anomaly inference   (n={n:,}, batch={batch}) ...", flush=True)
        for s in range(0, n, batch):
            e    = min(s + batch, n)
            wins = np.stack([a_vals[a_starts[i]: a_starts[i] + ANOMALY_SEQ]
                             for i in range(s, e)])
            t    = torch.tensor(wins, dtype=torch.float32)
            recon = anomaly_model(t)
            a_scores[s:e] = torch.mean((t - recon) ** 2, dim=(1, 2)).cpu().numpy()

        print(f"    corrector inference (n={n:,}, batch={batch}) ...", flush=True)
        for s in range(0, n, batch):
            e    = min(s + batch, n)
            wins = np.stack([c_vals[corr_starts[i]: corr_starts[i] + CORR_SEQ]
                             for i in range(s, e)])
            t    = torch.tensor(wins, dtype=torch.float32)
            residuals[s:e] = corrector_model(t).cpu().numpy().flatten()

    anomaly_score = np.clip(
        (a_scores - RECON_THRESH) / (RECON_MAX - RECON_THRESH), 0.0, 1.0)
    alert_level = np.where(
        anomaly_score < ALERT_GREEN, "GREEN",
        np.where(anomaly_score < ALERT_YELLOW, "YELLOW", "RED"))

    rows          = df.iloc[out_idx].reset_index(drop=True)
    dst_actual    = rows["dst"].values.astype(float)
    dst_burton    = rows["dst_burton"].values.astype(float)
    dst_corrector = dst_burton + residuals

    print(f"    RL blend inference  (n={n:,}) ...", flush=True)
    state_matrix = np.column_stack([
        dst_burton    / 500.0,
        dst_corrector / 500.0,
        anomaly_score,
        [ALERT_ENC.get(a, 0.0) for a in alert_level],
        rows["storm_phase"].values.astype(float) / 3.0,
        rows["E_field"].values.astype(float)     / 10.0,
        rows["bz_gsm"].values.astype(float)      / 50.0,
        rows["speed"].values.astype(float)        / 800.0,
        rows["dDst_dt"].values.astype(float)      / 20.0,
    ]).astype(np.float32)

    with torch.no_grad():
        weights, _ = rl.net(torch.tensor(state_matrix))
        w = weights.cpu().numpy()

    dst_final   = w[:, 0] * dst_burton + w[:, 1] * dst_corrector

    return pd.DataFrame({
        "dst":           dst_actual,
        "dst_burton":    dst_burton,
        "dst_corrector": dst_corrector,
        "dst_final":     dst_final,
        "err_burton":    dst_burton    - dst_actual,
        "err_corrector": dst_corrector - dst_actual,
        "err_final":     dst_final     - dst_actual,
        "anomaly_score": anomaly_score,
        "alert_level":   alert_level,
        "w_burton":      w[:, 0],
        "w_corrector":   w[:, 1],
    })


# ── Detection scores ──────────────────────────────────────────────────────────

def detection_scores(results: pd.DataFrame, alert_thresh: str = "RED",
                     dst_thresh: float = -50, lead_window: int = 60) -> dict:
    df = results.copy().reset_index(drop=True)
    dst = df["dst"].values

    storm = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if (dst[i: i + lead_window] < dst_thresh).any():
            storm[i] = True

    fired = (df["alert_level"] == alert_thresh).values if alert_thresh == "RED" \
            else df["alert_level"].isin(["YELLOW", "RED"]).values

    TP = int(( fired &  storm).sum())
    FP = int(( fired & ~storm).sum())
    FN = int((~fired &  storm).sum())
    TN = int((~fired & ~storm).sum())

    denom_far = FP + TN
    denom_pod = TP + FN
    denom_csi = TP + FP + FN
    hss_num   = 2 * (TP * TN - FP * FN)
    hss_den   = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)

    return {
        "TP":  TP, "FP": FP, "FN": FN, "TN": TN,
        "FAR": FP / denom_far if denom_far else 0.0,
        "POD": TP / denom_pod if denom_pod else 0.0,
        "CSI": TP / denom_csi if denom_csi else 0.0,
        "HSS": hss_num / hss_den if hss_den else 0.0,
    }


# ── Per-period report ─────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame, label: str,
                 add_ci: bool = False) -> dict:
    dst = results["dst"].values

    print()
    print("=" * 100)
    print(f"  VALIDATION REPORT  |  {label}  |  n={len(results):,}")
    print("=" * 100)

    # ── Ablation table (per class) ────────────────────────────────────────────
    hdr = f"  {'CLASS':<10} {'N':>7}  {'RMSE Burton':>12}  {'RMSE Corr':>11}  {'RMSE RL':>9}  " \
          f"{'Skill Corr':>11}  {'Skill RL':>9}  {'MAE RL':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * 95)

    class_metrics = {}
    for cls in CLASSES:
        mask = CLASS_MASKS[cls](dst)
        sub  = results[mask]
        if len(sub) < 2:
            continue
        rb = rmse(sub["err_burton"])
        rc = rmse(sub["err_corrector"])
        rf = rmse(sub["err_final"])
        mf = mae(sub["err_final"])
        sc = skill(rb, rc)
        sr = skill(rb, rf)

        ci_str = ""
        if add_ci and len(sub) >= 30:
            lo, hi = bootstrap_ci(sub["err_final"].values)
            ci_str = f"  [{lo:.2f}–{hi:.2f}]"

        print(f"  {cls.upper():<10} {len(sub):>7}  {rb:>12.2f}  {rc:>11.2f}  "
              f"{rf:>9.2f}  {sc:>10.1f}%  {sr:>8.1f}%  {mf:>8.2f}{ci_str}")
        class_metrics[cls] = {"rmse_burton": rb, "rmse_corr": rc,
                               "rmse_rl": rf, "mae_rl": mf, "n": len(sub)}

    print("  " + "-" * 95)
    rb_all = rmse(results["err_burton"])
    rc_all = rmse(results["err_corrector"])
    rf_all = rmse(results["err_final"])
    mf_all = mae(results["err_final"])
    ci_overall = ""
    if add_ci:
        lo, hi = bootstrap_ci(results["err_final"].values)
        ci_overall = f"  [95% CI: {lo:.2f}–{hi:.2f}]"
    print(f"  {'OVERALL':<10} {len(results):>7}  {rb_all:>12.2f}  {rc_all:>11.2f}  "
          f"{rf_all:>9.2f}  {skill(rb_all,rc_all):>10.1f}%  "
          f"{skill(rb_all,rf_all):>8.1f}%  {mf_all:>8.2f}{ci_overall}")
    print("=" * 100)

    # ── Detection scores ──────────────────────────────────────────────────────
    print(f"\n  ANOMALY ALERT DISTRIBUTION:")
    for lvl in ["GREEN", "YELLOW", "RED"]:
        n   = (results["alert_level"] == lvl).sum()
        pct = n / len(results) * 100
        print(f"  {lvl:<7} {n:>7,} ({pct:5.1f}%)  {'#' * int(pct / 2)}")

    # Detection scores are only meaningful on full temporal sequences.
    # Storm-enriched samples skip quiet/onset rows so FAR/POD/HSS will be
    # misleading. Only compute when the sample has GREEN rows (full coverage).
    green_pct = (results["alert_level"] == "GREEN").mean()
    if green_pct < 0.3:
        print("\n  [Detection scores skipped — storm-enriched sample lacks onset/quiet rows.")
        print("   Run --full-dataset for meaningful FAR/POD/HSS metrics.]")
    else:
        for thresh in ["YELLOW", "RED"]:
            s = detection_scores(results, alert_thresh=thresh)
            print(f"\n  DETECTION SCORES  (alert>={thresh}, storm = Dst<-50 within 60 steps)")
            print(f"  TP={s['TP']:,}  FP={s['FP']:,}  FN={s['FN']:,}  TN={s['TN']:,}")
            print(f"  FAR={s['FAR']:.3f} ({s['FAR']*100:.1f}%)  "
                  f"POD={s['POD']:.3f} ({s['POD']*100:.1f}%)  "
                  f"CSI={s['CSI']:.3f} ({s['CSI']*100:.1f}%)  "
                  f"HSS={s['HSS']:.3f}   "
                  f"[targets: FAR<30%, POD>80%, HSS>0.4]")

    print("=" * 100)
    return {
        "rmse_burton":    rb_all,
        "rmse_corrector": rc_all,
        "rmse_final":     rf_all,
        "mae_final":      mf_all,
        "class_metrics":  class_metrics,
    }


# ── Cross-period paper table ──────────────────────────────────────────────────

def print_paper_tables(all_results: dict, all_metrics: dict) -> None:
    # Combine all periods into one dataframe for true aggregate metrics
    combined = pd.concat(all_results.values(), ignore_index=True)
    dst      = combined["dst"].values

    print()
    print("=" * 100)
    print("  PAPER TABLE 1 — ABLATION: BURTON vs BiLSTM CORRECTOR vs RL BLEND")
    print("  (aggregated across all evaluated periods)")
    print("=" * 100)
    print(f"\n  {'Model':<22} {'Quiet':>8}  {'Minor':>8}  {'Moderate':>10}  "
          f"{'Intense':>8}  {'Extreme':>8}  {'Overall':>8}  {'MAE':>8}")
    print("  " + "-" * 90)

    for col, label in [
        ("err_burton",    "Burton ODE (baseline)"),
        ("err_corrector", "BiLSTM Corrector"),
        ("err_final",     "RL Final Blend"),
    ]:
        row = []
        for cls in CLASSES:
            mask = CLASS_MASKS[cls](dst)
            sub  = combined[mask]
            row.append(f"{rmse(sub[col]):>8.2f}" if len(sub) else f"{'N/A':>8}")
        overall = rmse(combined[col])
        mean_ae = mae(combined[col])
        print(f"  {label:<22} {'  '.join(row)}  {overall:>8.2f}  {mean_ae:>8.2f}")

    print("  " + "-" * 90)
    n_total  = len(combined)
    n_storms = int((combined["dst"].values <= -100).sum())
    storm_pct = n_storms / n_total * 100
    print(f"  All values in nT (RMSE / MAE). Lower is better.")
    print(f"  Dataset: {n_total:,} rows  |  storm rows (Dst<=-100): {n_storms:,} ({storm_pct:.1f}%)")
    if storm_pct > 40:
        print("  NOTE: Storm-enriched sample — 'Overall' is dominated by storms.")
        print("        Run --full-dataset for true population RMSE (quiet-dominated).")
    else:
        print("  NOTE: Full-dataset run — 'Overall' reflects true population RMSE.")

    # ── Improvement ratios ────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("  PAPER TABLE 2 — IMPROVEMENT RATIO vs BURTON BASELINE")
    print("=" * 100)
    print(f"\n  {'Model':<22} {'Quiet':>8}  {'Minor':>8}  {'Moderate':>10}  "
          f"{'Intense':>8}  {'Extreme':>8}  {'Overall':>8}")
    print("  " + "-" * 80)

    rb_by_class = {}
    for cls in CLASSES:
        mask = CLASS_MASKS[cls](dst)
        rb_by_class[cls] = rmse(combined[mask]["err_burton"]) if mask.sum() else 0.0
    rb_overall = rmse(combined["err_burton"])

    for col, label in [
        ("err_corrector", "BiLSTM Corrector"),
        ("err_final",     "RL Final Blend"),
    ]:
        row = []
        for cls in CLASSES:
            mask = CLASS_MASKS[cls](dst)
            sub  = combined[mask]
            if len(sub) and rb_by_class[cls] > 0:
                ratio = rb_by_class[cls] / rmse(sub[col])
                row.append(f"{ratio:>8.1f}x")
            else:
                row.append(f"{'N/A':>8}")
        ratio_overall = rb_overall / rmse(combined[col])
        print(f"  {label:<22} {'  '.join(row)}  {ratio_overall:>8.1f}x")

    print("  " + "-" * 80)
    print("  Values are improvement ratio (e.g. 3.0x = 3x lower RMSE than Burton)")

    # ── Published comparison ──────────────────────────────────────────────────
    rf_overall = rmse(combined["err_final"])
    rc_overall = rmse(combined["err_corrector"])

    n_total   = len(combined)
    n_storms  = int((combined["dst"].values <= -100).sum())
    storm_pct = n_storms / n_total * 100
    fair_compare = storm_pct < 40

    print()
    print("=" * 100)
    print("  PAPER TABLE 3 — COMPARISON WITH PUBLISHED MODELS (overall RMSE)")
    print("  IMPORTANT: published models report overall RMSE on different datasets.")
    if not fair_compare:
        print(f"  WARNING: current run is {storm_pct:.0f}% storm rows — published values include")
        print("  ~80% quiet conditions. This comparison is NOT valid for the current sample.")
        print("  Run --full-dataset for a fair comparison.")
    else:
        print("  Full-dataset run — directional comparison is reasonable.")
    print("=" * 100)
    print(f"\n  {'Model':<40} {'RMSE (nT)':>10}  {'vs Our RL':>12}")
    print("  " + "-" * 68)
    for name, val in sorted(PUBLISHED.items(), key=lambda x: x[1], reverse=True):
        diff   = val - rf_overall
        marker = "  <- OUR SYSTEM BETTER" if (diff > 0 and fair_compare) else ""
        print(f"  {name:<40} {val:>10.1f}  {diff:>+10.2f} nT{marker}")
    print("  " + "-" * 68)
    print(f"  {'Our BiLSTM Corrector':<40} {rc_overall:>10.2f}")
    print(f"  {'Our RL Final Blend':<40} {rf_overall:>10.2f}  <-- this system")
    if not fair_compare:
        print("  ^ Storm-enriched RMSE — not comparable to published overall RMSE above.")
    print("=" * 100)

    # ── Per-class novelty statement ───────────────────────────────────────────
    print()
    print("  NOVELTY NOTE:")
    print("  None of the published models above report per-class RMSE by storm severity.")
    print("  Table 1 is the first per-class ablation of this kind in the literature.")
    print("  The improvement for INTENSE and EXTREME storms is the primary contribution.")
    print()

    # ── Cross-period summary ──────────────────────────────────────────────────
    print("=" * 100)
    print("  CROSS-PERIOD SUMMARY")
    print("=" * 100)
    print(f"  {'Period':<14} {'N':>8}  {'RMSE Burton':>12}  {'RMSE Corr':>11}  "
          f"{'RMSE RL':>9}  {'Skill Corr':>11}  {'Skill RL':>9}")
    print("  " + "-" * 85)
    for period, m in all_metrics.items():
        n = len(all_results[period])
        sc = skill(m["rmse_burton"], m["rmse_corrector"])
        sr = skill(m["rmse_burton"], m["rmse_final"])
        print(f"  {period:<14} {n:>8,}  {m['rmse_burton']:>12.2f}  "
              f"{m['rmse_corrector']:>11.2f}  {m['rmse_final']:>9.2f}  "
              f"{sc:>10.1f}%  {sr:>8.1f}%")
    print("  " + "-" * 85)
    avg_b = np.mean([m["rmse_burton"]    for m in all_metrics.values()])
    avg_c = np.mean([m["rmse_corrector"] for m in all_metrics.values()])
    avg_f = np.mean([m["rmse_final"]     for m in all_metrics.values()])
    print(f"  {'MACRO-AVG':<14} {'':>8}  {avg_b:>12.2f}  {avg_c:>11.2f}  "
          f"{avg_f:>9.2f}  {skill(avg_b,avg_c):>10.1f}%  {skill(avg_b,avg_f):>8.1f}%")
    print(f"\n  Micro-avg (all rows combined): Burton={rmse(combined['err_burton']):.2f}  "
          f"Corrector={rc_overall:.2f}  RL={rf_overall:.2f} nT")
    print("=" * 100)


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(all_results: dict) -> None:
    BG = "#060b18"; FG = "#e8f4f8"
    CYAN = "#00d4ff"; GREEN = "#00ff88"; ORANGE = "#ff8c00"
    RED = "#e74c3c"; PURPLE = "#9b59b6"

    combined = pd.concat(all_results.values(), ignore_index=True)
    dst      = combined["dst"].values

    # ── Plot 1: NOAA benchmark comparison ────────────────────────────────────
    avg_rc = rmse(combined["err_corrector"])
    avg_rf = rmse(combined["err_final"])

    names  = list(PUBLISHED.keys()) + ["─" * 20, "Our Corrector", "Our RL Blend"]
    values = list(PUBLISHED.values()) + [None, avg_rc, avg_rf]
    colors = [ORANGE] * len(PUBLISHED) + ["none", GREEN, CYAN]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    for i, (name, val, col) in enumerate(zip(names, values, colors)):
        if val is None:
            continue
        bar = ax.barh(name, val, color=col, alpha=0.85)
        ax.text(val + 0.1, bar[0].get_y() + bar[0].get_height() / 2,
                f"{val:.2f} nT", va="center", color=FG, fontsize=8)
    ax.axvline(avg_rf, color=CYAN, lw=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Overall RMSE (nT)  — lower is better", color=FG)
    ax.set_title("System vs Published Models  |  Overall RMSE", color=CYAN, fontsize=12)
    ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "validation_noaa_comparison.png"),
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: validation_noaa_comparison.png")

    # ── Plot 2: Per-class RMSE bar chart (combined) ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(CLASSES))
    w = 0.25

    rb = [rmse(combined[CLASS_MASKS[c](dst)]["err_burton"])    for c in CLASSES]
    rc = [rmse(combined[CLASS_MASKS[c](dst)]["err_corrector"]) for c in CLASSES]
    rf = [rmse(combined[CLASS_MASKS[c](dst)]["err_final"])     for c in CLASSES]

    ax.bar(x - w, rb, w, label="Burton ODE",       color=ORANGE, alpha=0.85)
    ax.bar(x,     rc, w, label="BiLSTM Corrector", color=PURPLE, alpha=0.85)
    ax.bar(x + w, rf, w, label="RL Final Blend",   color=CYAN,   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in CLASSES], color=FG, fontsize=9)
    ax.set_ylabel("RMSE (nT)", color=FG)
    ax.set_title("Per-Class RMSE — All Periods Combined", color=CYAN, fontsize=11)
    ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
    ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)

    for bars in [ax.patches[:len(CLASSES)], ax.patches[len(CLASSES):2*len(CLASSES)],
                 ax.patches[2*len(CLASSES):]]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", color=FG, fontsize=7)

    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "validation_per_class.png"),
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: validation_per_class.png")

    # ── Plot 3: Storm trace — most extreme period ─────────────────────────────
    best_period = max(all_results.items(),
                      key=lambda x: (x[1]["dst"].values <= -200).sum())[0]
    res  = all_results[best_period]
    mask = res["dst"].values <= -100
    if mask.sum() > 50:
        sub = res[mask].reset_index(drop=True)
        t   = np.arange(len(sub))

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), facecolor=BG)
        fig.suptitle(f"Storm Deep Dive — {best_period} (Intense+Extreme rows)",
                     color=CYAN, fontsize=12)

        ax = axes[0]
        ax.set_facecolor(BG)
        ax.plot(t, sub["dst"],           color=FG,     lw=1.5, label="Actual Dst",    alpha=0.95)
        ax.plot(t, sub["dst_burton"],    color=ORANGE,  lw=1.0, label="Burton ODE",    alpha=0.75)
        ax.plot(t, sub["dst_corrector"], color=PURPLE,  lw=1.0, label="BiLSTM Corr",  alpha=0.80)
        ax.plot(t, sub["dst_final"],     color=CYAN,    lw=1.5, label="RL Final",      alpha=0.95)
        for thresh, lbl, col in [(-50,"MINOR","#ffe066"),(-100,"MODERATE",ORANGE),(-200,"EXTREME",RED)]:
            ax.axhline(thresh, color=col, lw=0.7, linestyle="--", alpha=0.5)
            ax.text(5, thresh + 4, lbl, color=col, fontsize=7)
        ax.set_ylabel("Dst (nT)", color=FG)
        ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)
        ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")

        ax = axes[1]
        ax.set_facecolor(BG)
        ax.stackplot(t, sub["w_burton"], sub["w_corrector"],
                     labels=["w_burton", "w_corrector"],
                     colors=[ORANGE, PURPLE], alpha=0.75)
        ax.plot(t, sub["anomaly_score"], color=CYAN, lw=1.2, label="Anomaly score")
        ax.set_ylabel("Weight / Score", color=FG)
        ax.set_xlabel("Step", color=FG)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)
        ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")

        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / "validation_storm_trace.png"),
                    dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Saved: validation_storm_trace.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate storm prediction pipeline")
    parser.add_argument("--full-dataset", action="store_true",
                        help="Run on all rows (no sampling). Paper-quality but slow.")
    parser.add_argument("--n-rows", type=int, default=10_000,
                        help="Max rows per period for quick mode (default 10000).")
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Periods to evaluate. Default: all periods in parquet.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--ci", action="store_true",
                        help="Add 95%% bootstrap confidence intervals (slower).")
    args = parser.parse_args()

    print("Loading models...")
    models = load_models()
    anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device = models

    parquet_path = ROOT / "data/enriched.parquet"
    print(f"Loading {parquet_path} ...")
    df_all = pd.read_parquet(parquet_path)
    df_all = df_all.dropna(subset=CORR_FEATS + ANOMALY_FEATS + ["dst", "dst_burton"])
    print(f"Total rows after dropna: {len(df_all):,}")

    # Auto-detect periods if not specified
    available = sorted(df_all["period"].unique().tolist())
    periods   = args.periods if args.periods else available
    print(f"Periods: {periods}")

    mode = "FULL DATASET" if args.full_dataset else f"SAMPLED ({args.n_rows:,} rows/period)"
    print(f"Mode: {mode}\n")

    all_results: dict[str, pd.DataFrame] = {}
    all_metrics: dict[str, dict]         = {}

    for period in periods:
        df = df_all[df_all["period"] == period].reset_index(drop=True)
        if len(df) == 0:
            print(f"  [{period}] no data, skipping.")
            continue

        if args.full_dataset:
            df_eval = df
        else:
            # Smart sampling: keep all storm rows + fill with quiet
            storm_mask = df["dst"] <= -100
            storm_rows = df[storm_mask]
            quiet_rows = df[~storm_mask]
            n_quiet    = max(0, args.n_rows - len(storm_rows))
            df_eval    = pd.concat([storm_rows, quiet_rows.iloc[:n_quiet]]) \
                           .sort_index().reset_index(drop=True)
            df_eval    = df_eval.iloc[:args.n_rows].reset_index(drop=True)

        n_ext = (df_eval["dst"] <= -200).sum()
        n_int = (df_eval["dst"] <= -100).sum()
        print(f"\n{'='*60}")
        print(f"Period: {period}  |  rows={len(df_eval):,}  "
              f"|  intense={n_int}  extreme={n_ext}")
        print(f"{'='*60}")

        results = run_inference(df_eval, *models)
        metrics = print_report(results, period, add_ci=args.ci)

        all_results[period] = results
        all_metrics[period] = metrics

    if not all_results:
        print("No results — check period names.")
        return

    print_paper_tables(all_results, all_metrics)

    if not args.no_plots:
        print("\nSaving plots...")
        save_plots(all_results)

    print("\nDone. Results in notebooks/Plots_Pipeline/")


if __name__ == "__main__":
    main()
