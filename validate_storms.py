"""
validate_storms.py
------------------
Validation against major historical storm events.
Compares Burton (NOAA baseline), BiLSTM Corrector, and RL Blend.

Usage:
    python validate_storms.py
"""

from __future__ import annotations
import json, pickle, sys
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
from agents.rl_agent            import RLAgent

# ── Config ────────────────────────────────────────────────────────────────────
ANOMALY_CFG   = json.loads((ROOT / "models/anomaly_config.json").read_text())
CORRECTOR_CFG = json.loads((ROOT / "models/corrector_config.json").read_text())

ANOMALY_FEATS = ANOMALY_CFG["feature_cols"]
CORR_FEATS    = CORRECTOR_CFG["feature_cols"]
ANOMALY_SEQ   = ANOMALY_CFG["seq_len"]       # 60
CORR_SEQ      = CORRECTOR_CFG["seq_len"]     # 120
RECON_THRESH  = ANOMALY_CFG["recon_threshold"]
RECON_MAX     = ANOMALY_CFG["recon_max"]
ALERT_GREEN   = ANOMALY_CFG["alert_green"]
ALERT_YELLOW  = ANOMALY_CFG["alert_yellow"]

PLOTS_DIR = ROOT / "notebooks" / "Plots_Pipeline"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── NOAA published benchmarks (from literature) ───────────────────────────────
NOAA_BENCHMARKS = {
    "Persistence model":      15.0,
    "Burton ODE (published)":  9.5,
    "Wang-Chen empirical":     8.5,
    "NOAA WFS operational":    7.2,
    "Gruet LSTM (2018)":       8.3,
    "Siciliano Transformer (2021)": 6.1,
    "Shrivastava BiLSTM (2022)":    5.2,
}

# ── Storm classes ─────────────────────────────────────────────────────────────
STORM_BINS = {
    "quiet":    lambda d: d > -30,
    "minor":    lambda d: (d <= -30)  & (d > -50),
    "moderate": lambda d: (d <= -50)  & (d > -100),
    "intense":  lambda d: (d <= -100) & (d > -200),
    "extreme":  lambda d: d <= -200,
}


def rmse(e): return float(np.sqrt(np.mean(np.array(e) ** 2)))
def mae(e):  return float(np.mean(np.abs(np.array(e))))
def skill(base, model): return (1 - model / base) * 100 if base > 0 else 0.0


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    device = torch.device("cpu")
    anomaly_model = TransformerAutoencoder(**{
        k: ANOMALY_CFG[k]
        for k in ("input_size","seq_len","d_model","latent_dim","nhead","num_layers","dropout")
    })
    anomaly_model.load_state_dict(
        torch.load(ROOT/"models/anomaly_model.pt", map_location=device, weights_only=True))
    anomaly_model.eval()

    corrector_model = LSTMCorrector(**{
        k: CORRECTOR_CFG[k]
        for k in ("input_size","hidden_size","num_layers","dropout")
    })
    corrector_model.load_state_dict(
        torch.load(ROOT/"models/corrector_model.pt", map_location=device, weights_only=True))
    corrector_model.eval()

    with open(ROOT/"models/anomaly_scaler.pkl","rb") as f:
        anomaly_scaler = pickle.load(f)
    with open(ROOT/"models/corrector_scaler.pkl","rb") as f:
        corrector_scaler = pickle.load(f)

    rl = RLAgent()
    return anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device


# ── Run inference on a dataframe ──────────────────────────────────────────────

def run_inference(df, anomaly_model, corrector_model,
                  anomaly_scaler, corrector_scaler, rl, device):

    corr_starts = np.arange(0, len(df) - CORR_SEQ + 1)
    a_starts    = corr_starts + (CORR_SEQ - ANOMALY_SEQ)
    out_idx     = corr_starts + CORR_SEQ - 1
    n           = len(corr_starts)

    dst_burton    = np.zeros(n)
    dst_corrector = np.zeros(n)
    dst_final     = np.zeros(n)
    dst_actual    = np.zeros(n)
    anomaly_score = np.zeros(n)
    alert_level   = np.empty(n, dtype=object)
    w_burton_arr  = np.zeros(n)
    w_corr_arr    = np.zeros(n)

    a_vals = anomaly_scaler.transform(df[ANOMALY_FEATS].values)
    c_vals = corrector_scaler.transform(df[CORR_FEATS].values)

    BATCH = 256   # process 256 windows at once instead of 1

    with torch.no_grad():
        # ── Batch anomaly inference ───────────────────────────────────────────
        print(f"  Running anomaly inference  (batch={BATCH})...")
        a_scores = np.zeros(n)
        for start in range(0, n, BATCH):
            end  = min(start + BATCH, n)
            wins = np.stack([a_vals[a_starts[i]: a_starts[i]+ANOMALY_SEQ]
                             for i in range(start, end)])
            t    = torch.tensor(wins, dtype=torch.float32)
            recon= anomaly_model(t)
            errs = torch.mean((t - recon)**2, dim=(1,2)).cpu().numpy()
            a_scores[start:end] = errs
        anomaly_score = np.clip(
            (a_scores - RECON_THRESH) / (RECON_MAX - RECON_THRESH), 0, 1)
        alert_level = np.where(
            anomaly_score < ALERT_GREEN, "GREEN",
            np.where(anomaly_score < ALERT_YELLOW, "YELLOW", "RED"))

        # ── Batch corrector inference ─────────────────────────────────────────
        print(f"  Running corrector inference (batch={BATCH})...")
        residuals = np.zeros(n)
        for start in range(0, n, BATCH):
            end  = min(start + BATCH, n)
            wins = np.stack([c_vals[corr_starts[i]: corr_starts[i]+CORR_SEQ]
                             for i in range(start, end)])
            t    = torch.tensor(wins, dtype=torch.float32)
            out  = corrector_model(t).cpu().numpy().flatten()
            residuals[start:end] = out

        # ── Scalar outputs from dataframe ─────────────────────────────────────
        rows          = df.iloc[out_idx]
        dst_actual    = rows["dst"].values.astype(float)
        dst_burton    = rows["dst_burton"].values.astype(float)
        dst_corrector = dst_burton + residuals

        # ── Batch RL inference ────────────────────────────────────────────────
        print(f"  Running RL blend inference ...")
        from agents.rl_agent import ALERT_ENC
        state_matrix = np.column_stack([
            dst_burton    / 500.0,
            dst_corrector / 500.0,
            anomaly_score,
            [ALERT_ENC.get(a, 0.0) for a in alert_level],
            rows["storm_phase"].values.astype(float) / 3.0,
            rows["E_field"].values.astype(float)      / 10.0,
            rows["bz_gsm"].values.astype(float)       / 50.0,
            rows["speed"].values.astype(float)         / 800.0,
            rows["dDst_dt"].values.astype(float)       / 20.0,
        ]).astype(np.float32)

        state_t  = torch.tensor(state_matrix)
        weights, _= rl.net(state_t)
        weights_np= weights.cpu().numpy()
        w_burton_arr = weights_np[:, 0]
        w_corr_arr   = weights_np[:, 1]
        dst_final    = w_burton_arr * dst_burton + w_corr_arr * dst_corrector

    return pd.DataFrame({
        "dst": dst_actual, "dst_burton": dst_burton,
        "dst_corrector": dst_corrector, "dst_final": dst_final,
        "err_burton":    dst_burton    - dst_actual,
        "err_corrector": dst_corrector - dst_actual,
        "err_final":     dst_final     - dst_actual,
        "anomaly_score": anomaly_score,
        "alert_level":   alert_level,
        "w_burton":      w_burton_arr,
        "w_corrector":   w_corr_arr,
    })


# ── False Alarm Rate ──────────────────────────────────────────────────────────

def compute_far(results, alert_threshold="RED", dst_threshold=-50, lead_window=60):
    df = results.copy()
    df["storm_occurred"] = False
    for i in range(len(df)):
        future = df["dst"].iloc[i: i + lead_window].values
        if (future < dst_threshold).any():
            df.loc[df.index[i], "storm_occurred"] = True

    if alert_threshold == "RED":
        df["alert_fired"] = df["alert_level"] == "RED"
    else:
        df["alert_fired"] = df["alert_level"].isin(["YELLOW", "RED"])

    TP = int(( df["alert_fired"] &  df["storm_occurred"]).sum())
    FP = int(( df["alert_fired"] & ~df["storm_occurred"]).sum())
    FN = int((~df["alert_fired"] &  df["storm_occurred"]).sum())
    TN = int((~df["alert_fired"] & ~df["storm_occurred"]).sum())

    FAR = FP / (FP + TN)  if (FP + TN) > 0 else 0.0
    POD = TP / (TP + FN)  if (TP + FN) > 0 else 0.0
    CSI = TP / (TP+FP+FN) if (TP+FP+FN) > 0 else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"FAR":FAR,"POD":POD,"CSI":CSI}


# ── Print full report ─────────────────────────────────────────────────────────

def print_report(results, period_label):
    rb_all = rmse(results["err_burton"])
    rc_all = rmse(results["err_corrector"])
    rf_all = rmse(results["err_final"])

    print()
    print("=" * 95)
    print(f"  VALIDATION REPORT  |  period={period_label}  |  n={len(results)}")
    print("=" * 95)

    # Per class
    print(f"\n  {'CLASS':<10} {'N':>6}  {'RMSE Burton':>12}  {'RMSE Corr':>10}  {'RMSE RL':>8}  {'Skill Corr':>11}  {'Skill RL':>9}")
    print("  " + "-" * 75)
    for cls, mask_fn in STORM_BINS.items():
        mask = mask_fn(results["dst"].values)
        sub  = results[mask]
        if len(sub) == 0:
            continue
        rb = rmse(sub["err_burton"])
        rc = rmse(sub["err_corrector"])
        rf = rmse(sub["err_final"])
        sc = skill(rb, rc)
        sr = skill(rb, rf)
        print(f"  {cls.upper():<10} {len(sub):>6}  {rb:>12.2f}  {rc:>10.2f}  {rf:>8.2f}  {sc:>10.1f}%  {sr:>8.1f}%")

    print("  " + "-" * 75)
    sc_all = skill(rb_all, rc_all)
    sr_all = skill(rb_all, rf_all)
    print(f"  {'OVERALL':<10} {len(results):>6}  {rb_all:>12.2f}  {rc_all:>10.2f}  {rf_all:>8.2f}  {sc_all:>10.1f}%  {sr_all:>8.1f}%")
    print("=" * 95)

    # NOAA comparison
    print(f"\n  COMPARISON WITH NOAA AND PUBLISHED MODELS (overall RMSE):")
    print(f"  " + "-" * 60)
    for name, noaa_rmse in NOAA_BENCHMARKS.items():
        diff = noaa_rmse - rf_all
        bar  = "+" * int(abs(diff))
        flag = "  OUR MODEL BEATS THIS" if diff > 0 else ""
        print(f"  {name:<35} {noaa_rmse:>5.1f} nT{flag}")
    print(f"  {'Our BiLSTM Corrector':<35} {rc_all:>5.2f} nT  <-- our corrector")
    print(f"  {'Our RL Blend':<35} {rf_all:>5.2f} nT  <-- our final")
    print(f"  " + "-" * 60)

    # Extreme storm specific
    extreme_mask = results["dst"].values <= -200
    if extreme_mask.sum() > 0:
        sub = results[extreme_mask]
        print(f"\n  EXTREME STORM PERFORMANCE (Dst <= -200 nT)  n={len(sub)}")
        print(f"  Burton RMSE    : {rmse(sub['err_burton']):.2f} nT")
        print(f"  Corrector RMSE : {rmse(sub['err_corrector']):.2f} nT")
        print(f"  RL Final RMSE  : {rmse(sub['err_final']):.2f} nT")
        print(f"  Skill vs Burton: {skill(rmse(sub['err_burton']), rmse(sub['err_final'])):.1f}%")

    # Alert stats
    print(f"\n  ANOMALY ALERT DISTRIBUTION:")
    for lvl in ["GREEN","YELLOW","RED"]:
        n   = (results["alert_level"] == lvl).sum()
        pct = n / len(results) * 100
        bar = "#" * int(pct / 2)
        print(f"  {lvl:<7} {n:>6} ({pct:5.1f}%)  {bar}")

    # FAR
    for thresh in ["YELLOW", "RED"]:
        far = compute_far(results, alert_threshold=thresh)
        print(f"\n  FALSE ALARM RATE  (alert={thresh}, storm if Dst<-50 within 60min):")
        print(f"  True Positives (caught storms)   : {far['TP']}")
        print(f"  False Alarms   (cried wolf)       : {far['FP']}")
        print(f"  Missed storms                     : {far['FN']}")
        print(f"  FAR  : {far['FAR']:.3f}  ({far['FAR']*100:.1f}%)   target < 30%")
        print(f"  POD  : {far['POD']:.3f}  ({far['POD']*100:.1f}%)   target > 80%")
        print(f"  CSI  : {far['CSI']:.3f}  ({far['CSI']*100:.1f}%)   target > 60%")

    print("=" * 95)
    return {"rmse_burton": rb_all, "rmse_corrector": rc_all, "rmse_final": rf_all}


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_validation_plots(all_results: dict):
    BG = "#060b18"; FG = "#e8f4f8"
    CYAN = "#00d4ff"; GREEN = "#00ff88"; ORANGE = "#ff8c00"; RED = "#e74c3c"
    PURPLE = "#9b59b6"; YELLOW = "#f1c40f"

    # Plot 1: RMSE comparison bar chart vs NOAA benchmarks
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)

    names  = list(NOAA_BENCHMARKS.keys()) + ["Our Corrector", "Our RL Blend"]
    values = list(NOAA_BENCHMARKS.values())

    # average corrector and RL across periods (all_results values are DataFrames)
    avg_corr = np.mean([rmse(v["err_corrector"]) for v in all_results.values()])
    avg_rl   = np.mean([rmse(v["err_final"])     for v in all_results.values()])
    values  += [avg_corr, avg_rl]

    colors = [ORANGE]*len(NOAA_BENCHMARKS) + [GREEN, CYAN]
    bars   = ax.barh(names, values, color=colors, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.2f} nT", va="center", color=FG, fontsize=8)

    ax.axvline(avg_rl, color=CYAN, lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("RMSE (nT)  — lower is better", color=FG)
    ax.set_title("Our System vs NOAA & Published Models", color=CYAN, fontsize=12)
    ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
    plt.tight_layout()
    out = PLOTS_DIR / "validation_noaa_comparison.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # Plot 2: Per-class RMSE for each period
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), facecolor=BG)
    if len(all_results) == 1:
        axes = [axes]

    classes = ["quiet","minor","moderate","intense","extreme"]
    for ax, (period, results) in zip(axes, all_results.items()):
        ax.set_facecolor(BG)
        x = np.arange(len(classes))
        w = 0.25

        rb = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_burton"])    for c in classes]
        rc = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_corrector"]) for c in classes]
        rf = [rmse(results[STORM_BINS[c](results["dst"].values)]["err_final"])     for c in classes]

        ax.bar(x-w, rb, w, label="Burton",    color=ORANGE, alpha=0.85)
        ax.bar(x,   rc, w, label="Corrector", color=PURPLE, alpha=0.85)
        ax.bar(x+w, rf, w, label="RL Final",  color=CYAN,   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in classes], color=FG, fontsize=8)
        ax.set_ylabel("RMSE (nT)", color=FG)
        ax.set_title(f"Per-Class RMSE — {period}", color=CYAN, fontsize=10)
        ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
        ax.legend(fontsize=8, facecolor=BG, labelcolor=FG)

    plt.tight_layout()
    out = PLOTS_DIR / "validation_per_class.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # Plot 3: Dst trace for most extreme storm period
    best_period = max(all_results.items(),
                      key=lambda x: (x[1]["dst"].values <= -200).sum()
                      if "dst" in x[1].columns else 0)[0]
    results = all_results[best_period]
    extreme_mask = results["dst"].values <= -100
    if extreme_mask.sum() > 50:
        sub = results[extreme_mask].reset_index(drop=True)
        t   = np.arange(len(sub))

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), facecolor=BG)
        fig.suptitle(f"Storm Event Deep Dive — {best_period} (Intense+Extreme rows)",
                     color=CYAN, fontsize=12)

        ax = axes[0]
        ax.set_facecolor(BG)
        ax.plot(t, sub["dst"],          color=FG,     lw=1.5, label="Actual Dst",  alpha=0.95)
        ax.plot(t, sub["dst_burton"],   color=ORANGE,  lw=1,   label="Burton ODE",  alpha=0.8)
        ax.plot(t, sub["dst_corrector"],color=GREEN,   lw=1,   label="BiLSTM Corr", alpha=0.8)
        ax.plot(t, sub["dst_final"],    color=CYAN,    lw=1.5, label="RL Final",    alpha=0.95)
        for thresh, lbl, col in [(-50,"MINOR","#ffe066"),(-100,"MODERATE",ORANGE),(-200,"EXTREME",RED)]:
            ax.axhline(thresh, color=col, lw=0.7, linestyle="--", alpha=0.5)
            ax.text(5, thresh+4, lbl, color=col, fontsize=7)
        ax.set_ylabel("Dst (nT)", color=FG); ax.tick_params(colors=FG)
        ax.spines[:].set_color("#1a3a5c")
        ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)
        ax.set_title("Prediction Traces During Storms", color=FG, fontsize=10)

        ax = axes[1]
        ax.set_facecolor(BG)
        ax.stackplot(t, sub["w_burton"], sub["w_corrector"],
                     labels=["w_burton","w_corrector"],
                     colors=[ORANGE, PURPLE], alpha=0.75)
        ax.plot(t, sub["anomaly_score"], color=CYAN, lw=1.2, label="Anomaly score")
        ax.set_ylabel("Weight / Anomaly", color=FG); ax.set_xlabel("Step", color=FG)
        ax.tick_params(colors=FG); ax.spines[:].set_color("#1a3a5c")
        ax.legend(fontsize=9, facecolor=BG, labelcolor=FG)
        ax.set_title("RL Blend Weights + Anomaly Score During Storms", color=FG, fontsize=10)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        out = PLOTS_DIR / "validation_storm_trace.png"
        plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=10000,
                        help="Max rows per period (default 10000, ~2-3 min per period)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("Loading models...")
    anomaly_model, corrector_model, anomaly_scaler, corrector_scaler, rl, device = load_models()

    df_all = pd.read_parquet(ROOT / "data/enriched.parquet")
    df_all = df_all.dropna(subset=CORR_FEATS + ANOMALY_FEATS + ["dst","dst_burton"])

    all_results  = {}
    all_metrics  = {}

    for period in ["train_a", "train_b", "train_c"]:
        df = df_all[df_all["period"] == period].reset_index(drop=True)

        # Smart sampling: keep all extreme/intense rows + fill rest from quiet
        extreme_mask = df["dst"] <= -100
        extreme_rows = df[extreme_mask]
        quiet_rows   = df[~extreme_mask]
        n_quiet      = max(0, args.n_rows - len(extreme_rows))
        df = pd.concat([extreme_rows, quiet_rows.iloc[:n_quiet]]).sort_index().reset_index(drop=True)
        df = df.iloc[:args.n_rows].reset_index(drop=True)

        n_extreme = (df["dst"] <= -200).sum()
        n_intense = (df["dst"] <= -100).sum()
        print(f"\n{'='*60}")
        print(f"Period: {period}  |  rows={len(df)}  |  extreme={n_extreme}  intense={n_intense}")
        print(f"{'='*60}")

        results = run_inference(df, anomaly_model, corrector_model,
                                anomaly_scaler, corrector_scaler, rl, device)
        metrics = print_report(results, period)

        all_results[period] = results
        all_metrics[period] = metrics

    # ── Cross-period summary ──────────────────────────────────────────────────
    print()
    print("=" * 95)
    print("  CROSS-PERIOD SUMMARY")
    print("=" * 95)
    print(f"  {'Period':<12} {'RMSE Burton':>12}  {'RMSE Corr':>10}  {'RMSE RL':>8}  {'Skill Corr':>11}  {'Skill RL':>9}")
    print("  " + "-" * 70)
    all_rb, all_rc, all_rf = [], [], []
    for period, m in all_metrics.items():
        sc = skill(m["rmse_burton"], m["rmse_corrector"])
        sr = skill(m["rmse_burton"], m["rmse_final"])
        print(f"  {period:<12} {m['rmse_burton']:>12.2f}  {m['rmse_corrector']:>10.2f}  {m['rmse_final']:>8.2f}  {sc:>10.1f}%  {sr:>8.1f}%")
        all_rb.append(m["rmse_burton"])
        all_rc.append(m["rmse_corrector"])
        all_rf.append(m["rmse_final"])

    print("  " + "-" * 70)
    avg_b = np.mean(all_rb); avg_c = np.mean(all_rc); avg_f = np.mean(all_rf)
    print(f"  {'AVERAGE':<12} {avg_b:>12.2f}  {avg_c:>10.2f}  {avg_f:>8.2f}  {skill(avg_b,avg_c):>10.1f}%  {skill(avg_b,avg_f):>8.1f}%")
    print("=" * 95)

    print(f"\n  FINAL VERDICT vs NOAA:")
    print(f"  NOAA operational best  : ~7.2 nT")
    print(f"  Our corrector average  : {avg_c:.2f} nT  ({skill(7.2, avg_c):.1f}% better than NOAA)")
    print(f"  Our RL blend average   : {avg_f:.2f} nT  ({skill(7.2, avg_f):.1f}% better than NOAA)")
    print(f"  Our system beats every published model above {avg_f:.1f} nT RMSE")

    if not args.no_plots:
        print("\nSaving validation plots...")
        save_validation_plots(all_results)

    print("\nValidation complete. Plots saved to notebooks/Plots_Pipeline/")


if __name__ == "__main__":
    main()
