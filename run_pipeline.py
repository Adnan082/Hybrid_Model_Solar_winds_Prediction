"""
run_pipeline.py
---------------
Offline batch evaluation of the full hybrid pipeline.
Shows stacked model decisions, RL blending behaviour, and per-storm-class metrics.

Usage:
    python run_pipeline.py                  # all periods, first 5000 rows
    python run_pipeline.py --period train_a
    python run_pipeline.py --n-rows 2000
    python run_pipeline.py --no-plots
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
from models.lstm_corrector import LSTMCorrector
from agents.rl_agent import RLAgent
from training.mlflow_logger import (
    start_pipeline_run, log_pipeline_metrics,
    log_pipeline_artifacts, log_rl_checkpoint,
)

# ── Config ────────────────────────────────────────────────────────────────────
ANOMALY_CFG  = json.loads((ROOT / "models/anomaly_config.json").read_text())
CORRECTOR_CFG= json.loads((ROOT / "models/corrector_config.json").read_text())

ANOMALY_FEATS = ANOMALY_CFG["feature_cols"]          # 8 features
CORR_FEATS    = CORRECTOR_CFG["feature_cols"]         # 17 features
ANOMALY_SEQ   = ANOMALY_CFG["seq_len"]               # 60
CORR_SEQ      = CORRECTOR_CFG["seq_len"]             # 120
RECON_THRESH  = ANOMALY_CFG["recon_threshold"]
RECON_MAX     = ANOMALY_CFG["recon_max"]
ALERT_GREEN   = ANOMALY_CFG["alert_green"]
ALERT_YELLOW  = ANOMALY_CFG["alert_yellow"]

PLOTS_DIR = ROOT / "notebooks" / "Plots_Pipeline"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STORM_BINS = {
    "quiet":    lambda d: d > -30,
    "minor":    lambda d: (d <= -30) & (d > -50),
    "moderate": lambda d: (d <= -50) & (d > -100),
    "intense":  lambda d: (d <= -100) & (d > -200),
    "extreme":  lambda d: d <= -200,
}


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    device = torch.device("cpu")

    anomaly_model = TransformerAutoencoder(**{
        k: ANOMALY_CFG[k]
        for k in ("input_size", "seq_len", "d_model", "latent_dim", "nhead", "num_layers", "dropout")
    }).to(device)
    anomaly_model.load_state_dict(
        torch.load(ROOT / "models/anomaly_model.pt", map_location=device, weights_only=True)
    )
    anomaly_model.eval()

    corrector_model = LSTMCorrector(**{
        k: CORRECTOR_CFG[k]
        for k in ("input_size", "hidden_size", "num_layers", "dropout")
    }).to(device)
    corrector_model.load_state_dict(
        torch.load(ROOT / "models/corrector_model.pt", map_location=device, weights_only=True)
    )
    corrector_model.eval()

    with open(ROOT / "models/anomaly_scaler.pkl", "rb") as f:
        anomaly_scaler = pickle.load(f)
    with open(ROOT / "models/corrector_scaler.pkl", "rb") as f:
        corrector_scaler = pickle.load(f)

    rl = RLAgent()

    print("Models loaded.")
    return anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, anomaly_model, corrector_model,
                  anomaly_scaler, corrector_scaler, rl: RLAgent, device):

    # Window alignment: corrector drives the index (longer window)
    corr_starts = np.arange(0, len(df) - CORR_SEQ + 1)
    a_starts    = corr_starts + (CORR_SEQ - ANOMALY_SEQ)   # aligned end
    out_idx     = corr_starts + CORR_SEQ - 1               # last row of window

    n = len(corr_starts)
    print(f"Running inference on {n} windows...")

    dst_burton    = np.zeros(n)
    dst_corrector = np.zeros(n)
    dst_final     = np.zeros(n)
    dst_actual    = np.zeros(n)
    anomaly_score = np.zeros(n)
    recon_error   = np.zeros(n)
    alert_level   = np.empty(n, dtype=object)
    w_burton_arr  = np.zeros(n)
    w_corr_arr    = np.zeros(n)
    blend_cert    = np.zeros(n)
    storm_phase_arr = np.zeros(n, dtype=int)

    # Scaled arrays
    a_vals = anomaly_scaler.transform(df[ANOMALY_FEATS].values)
    c_vals = corrector_scaler.transform(df[CORR_FEATS].values)

    with torch.no_grad():
        for i in range(n):
            # ── Anomaly window ────────────────────────────────────────────────
            a_win = a_vals[a_starts[i]: a_starts[i] + ANOMALY_SEQ]
            a_t   = torch.tensor(a_win, dtype=torch.float32).unsqueeze(0).to(device)
            recon = anomaly_model(a_t)
            err   = float(torch.mean((a_t - recon) ** 2).item())
            score = float(np.clip((err - RECON_THRESH) / (RECON_MAX - RECON_THRESH), 0, 1))
            alert = "GREEN" if score < ALERT_GREEN else ("YELLOW" if score < ALERT_YELLOW else "RED")
            recon_error[i]   = err
            anomaly_score[i] = score
            alert_level[i]   = alert

            # ── Corrector window ──────────────────────────────────────────────
            c_win   = c_vals[corr_starts[i]: corr_starts[i] + CORR_SEQ]
            c_t     = torch.tensor(c_win, dtype=torch.float32).unsqueeze(0).to(device)
            residual= float(corrector_model(c_t).item())

            row = df.iloc[out_idx[i]]
            d_burton    = float(row["dst_burton"])
            d_corrector = d_burton + residual
            d_actual    = float(row["dst"])
            sp          = int(row["storm_phase"])
            ef          = float(row["E_field"])
            bz          = float(row["bz_gsm"])
            spd         = float(row["speed"])
            ddt         = float(row["dDst_dt"])

            dst_burton[i]    = d_burton
            dst_corrector[i] = d_corrector
            dst_actual[i]    = d_actual
            storm_phase_arr[i] = sp

            # ── RL blend ──────────────────────────────────────────────────────
            state = {
                "dst_burton":   d_burton,
                "dst_corrector":d_corrector,
                "anomaly_score":score,
                "alert_level":  alert,
                "storm_phase":  sp,
                "E_field":      ef,
                "bz_gsm":       bz,
                "speed":        spd,
                "dDst_dt":      ddt,
            }
            weights = rl.act(state)          # returns np.ndarray [w_burton, w_corrector]
            wb, wc  = float(weights[0]), float(weights[1])
            blend   = float(rl.blend(state, weights))
            cert    = float(abs(wb - wc))

            dst_final[i]  = blend
            w_burton_arr[i] = wb
            w_corr_arr[i]   = wc
            blend_cert[i]   = cert

            # RL online reward every step
            rl.give_reward(d_actual)

            # Print every 200 rows
            if i % 200 == 0 or i < 5:
                _print_row(i, n, d_actual, d_burton, d_corrector, blend,
                           score, alert, wb, wc, cert, sp)

    results = pd.DataFrame({
        "dst":          dst_actual,
        "dst_burton":   dst_burton,
        "dst_corrector":dst_corrector,
        "dst_final":    dst_final,
        "err_burton":   dst_burton    - dst_actual,
        "err_corrector":dst_corrector - dst_actual,
        "err_final":    dst_final     - dst_actual,
        "anomaly_score":anomaly_score,
        "recon_error":  recon_error,
        "alert_level":  alert_level,
        "w_burton":     w_burton_arr,
        "w_corrector":  w_corr_arr,
        "blend_certainty": blend_cert,
        "storm_phase":  storm_phase_arr,
    })
    return results


def _print_row(i, n, actual, burton, corrector, final,
               score, alert, wb, wc, cert, phase):
    pct = i / max(n - 1, 1) * 100
    print(
        f"  [{i:5d}/{n}  {pct:5.1f}%]"
        f"  Dst actual={actual:7.1f}  burton={burton:7.1f}"
        f"  corrector={corrector:7.1f}  final={final:7.1f}"
        f"  | anomaly={score:.3f} [{alert:6s}]"
        f"  | w_burton={wb:.2f} w_corr={wc:.2f}  cert={cert:.2f}"
        f"  phase={phase}"
    )


# ── Metrics table ─────────────────────────────────────────────────────────────

def print_metrics(results: pd.DataFrame, period: str):
    def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
    def mae(e):  return float(np.mean(np.abs(e)))
    def skill(base, model): return (1 - model / base) * 100 if base > 0 else 0.0

    print()
    print("=" * 90)
    print(f"  PIPELINE RESULTS  |  period={period}  |  n={len(results)}")
    print("=" * 90)
    print(f"  {'CLASS':<10} {'N':>6}  {'RMSE Burton':>12}  {'RMSE Corr':>10}  {'RMSE Final':>10}  {'MAE Final':>9}  {'Skill%':>7}")
    print("-" * 90)

    for cls, mask_fn in STORM_BINS.items():
        mask = mask_fn(results["dst"].values)
        sub  = results[mask]
        if len(sub) == 0:
            continue
        rb = rmse(sub["err_burton"].values)
        rc = rmse(sub["err_corrector"].values)
        rf = rmse(sub["err_final"].values)
        mf = mae(sub["err_final"].values)
        sk = skill(rb, rf)
        print(f"  {cls.upper():<10} {len(sub):>6}  {rb:>12.2f}  {rc:>10.2f}  {rf:>10.2f}  {mf:>9.2f}  {sk:>6.1f}%")

    print("-" * 90)
    rb = rmse(results["err_burton"].values)
    rc = rmse(results["err_corrector"].values)
    rf = rmse(results["err_final"].values)
    mf = mae(results["err_final"].values)
    sk = skill(rb, rf)
    print(f"  {'OVERALL':<10} {len(results):>6}  {rb:>12.2f}  {rc:>10.2f}  {rf:>10.2f}  {mf:>9.2f}  {sk:>6.1f}%")
    print("=" * 90)

    print()
    print("  RL BLEND WEIGHTS")
    print(f"  mean w_burton    = {results['w_burton'].mean():.3f}")
    print(f"  mean w_corrector = {results['w_corrector'].mean():.3f}")
    print(f"  mean certainty   = {results['blend_certainty'].mean():.3f}")
    print(f"  RL reward steps  = {len(results)}")

    print()
    print("  ALERT DISTRIBUTION")
    for lvl in ["GREEN", "YELLOW", "RED"]:
        n = (results["alert_level"] == lvl).sum()
        pct = n / len(results) * 100
        bar = "#" * int(pct / 2)
        print(f"  {lvl:<7} {n:>6}  ({pct:5.1f}%)  {bar}")
    print("=" * 90)


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(results: pd.DataFrame, period: str):
    BG = "#060b18"; FG = "#e8f4f8"; CYAN = "#00d4ff"
    GREEN = "#00ff88"; ORANGE = "#ff8c00"; RED = "#e74c3c"; PURPLE = "#9b59b6"

    t = np.arange(len(results))

    # Plot 1: Dst predictions vs actual
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor=BG)
    fig.suptitle(f"Solar Wind Dst Pipeline — {period}", color=CYAN, fontsize=13)

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(t, results["dst"],          color=FG,     lw=1.2, label="Actual Dst", alpha=0.9)
    ax.plot(t, results["dst_burton"],   color=ORANGE,  lw=1,   label="Burton ODE", alpha=0.8)
    ax.plot(t, results["dst_corrector"],color=GREEN,   lw=1,   label="BiLSTM Corrector", alpha=0.8)
    ax.plot(t, results["dst_final"],    color=CYAN,    lw=1.5, label="RL Final", alpha=0.95)
    for thresh, lbl, col in [(-50,"MINOR","#ffe066"),(-100,"MODERATE","#ff8c00"),(-200,"EXTREME",RED)]:
        ax.axhline(thresh, color=col, lw=0.6, linestyle="--", alpha=0.5)
        ax.text(len(t)*0.01, thresh+3, lbl, color=col, fontsize=7)
    ax.set_ylabel("Dst (nT)", color=FG); ax.tick_params(colors=FG)
    ax.spines[:].set_color("#1a3a5c"); ax.legend(fontsize=8, facecolor=BG, labelcolor=FG)
    ax.set_title("Dst Predictions", color=FG, fontsize=10)

    # Plot 2: RL blend weights
    ax = axes[1]
    ax.set_facecolor(BG)
    ax.stackplot(t, results["w_burton"], results["w_corrector"],
                 labels=["w_burton", "w_corrector"],
                 colors=[ORANGE, PURPLE], alpha=0.75)
    ax.plot(t, results["blend_certainty"], color=CYAN, lw=1, label="Certainty", alpha=0.8)
    ax.set_ylabel("Weight / Certainty", color=FG); ax.tick_params(colors=FG)
    ax.spines[:].set_color("#1a3a5c"); ax.legend(fontsize=8, facecolor=BG, labelcolor=FG)
    ax.set_title("RL Blend Weights (how the agent decides)", color=FG, fontsize=10)
    ax.set_ylim(0, 1.05)

    # Plot 3: Anomaly score + alert
    ax = axes[2]
    ax.set_facecolor(BG)
    color_map = {"GREEN": GREEN, "YELLOW": "#ffe066", "RED": RED}
    colors = [color_map[a] for a in results["alert_level"]]
    ax.scatter(t, results["anomaly_score"], c=colors, s=1.5, alpha=0.6)
    ax.axhline(0.4, color="#ffe066", lw=0.8, linestyle="--", alpha=0.6, label="YELLOW threshold")
    ax.axhline(0.8, color=RED,       lw=0.8, linestyle="--", alpha=0.6, label="RED threshold")
    ax.set_ylabel("Anomaly Score", color=FG); ax.set_xlabel("Time step (min)", color=FG)
    ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
    ax.legend(fontsize=8, facecolor=BG, labelcolor=FG)
    ax.set_title("Anomaly Detector (Transformer AE)", color=FG, fontsize=10)

    plt.tight_layout()
    out1 = PLOTS_DIR / f"pipeline_{period}_dst.png"
    plt.savefig(str(out1), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out1}")

    # Plot 2: Errors per storm class
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    classes = list(STORM_BINS.keys())
    x       = np.arange(len(classes))
    width   = 0.25

    def rmse(e): return float(np.sqrt(np.mean(e ** 2))) if len(e) else 0.0

    rb_vals = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_burton"].values)  for c in classes]
    rc_vals = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_corrector"].values) for c in classes]
    rf_vals = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_final"].values)   for c in classes]

    ax.bar(x - width, rb_vals, width, label="Burton",    color=ORANGE, alpha=0.8)
    ax.bar(x,         rc_vals, width, label="Corrector", color=PURPLE, alpha=0.8)
    ax.bar(x + width, rf_vals, width, label="RL Final",  color=CYAN,   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in classes], color=FG)
    ax.set_ylabel("RMSE (nT)", color=FG); ax.tick_params(colors=FG)
    ax.spines[:].set_color("#1a3a5c")
    ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)
    ax.set_title(f"RMSE by Storm Class — {period}", color=CYAN, fontsize=12)
    plt.tight_layout()
    out2 = PLOTS_DIR / f"pipeline_{period}_rmse.png"
    plt.savefig(str(out2), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out2}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline pipeline evaluation")
    parser.add_argument("--period",  default=None, help="Data period (e.g. train_a)")
    parser.add_argument("--n-rows",  type=int, default=5000, help="Max rows to evaluate")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df_all = pd.read_parquet(ROOT / "data/enriched.parquet")
    if args.period:
        df = df_all[df_all["period"] == args.period].copy()
        if len(df) == 0:
            print(f"ERROR: period '{args.period}' not found. Available: {df_all['period'].unique().tolist()}")
            sys.exit(1)
    else:
        df = df_all.copy()

    df = df.dropna(subset=CORR_FEATS + ANOMALY_FEATS + ["dst", "dst_burton"]).reset_index(drop=True)
    if args.n_rows and len(df) > args.n_rows:
        df = df.iloc[:args.n_rows].reset_index(drop=True)

    period_label = args.period or "all"
    print(f"Period: {period_label}  |  Rows: {len(df)}")

    # ── Load models ───────────────────────────────────────────────────────────
    anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device = load_models()

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow_run = None
    if not args.no_mlflow:
        try:
            mlflow_run = start_pipeline_run(period=period_label)
        except Exception as e:
            print(f"[MLflow] skipped: {e}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print()
    print("Step-by-step decisions (every 200 rows):")
    print("-" * 90)
    results = run_inference(df, anomaly_model, corrector_model,
                            anomaly_scaler, corrector_scaler, rl, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print_metrics(results, period_label)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = PLOTS_DIR / "pipeline_output.csv"
    results.to_csv(csv_out, index=False)
    print(f"\n  CSV saved: {csv_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nSaving plots...")
        save_plots(results, period_label)

    # ── MLflow logging ────────────────────────────────────────────────────────
    if mlflow_run and not args.no_mlflow:
        try:
            import mlflow
            log_pipeline_metrics(results, period=period_label)
            log_pipeline_artifacts(PLOTS_DIR)
            log_rl_checkpoint(rl_reward_steps=len(results))
            mlflow.end_run()
            print("\n  MLflow run logged.")
        except Exception as e:
            print(f"[MLflow] logging failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
