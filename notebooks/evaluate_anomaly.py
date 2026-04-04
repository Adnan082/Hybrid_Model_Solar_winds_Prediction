"""
notebooks/evaluate_anomaly.py
------------------------------
Standalone evaluation script for the Transformer Autoencoder anomaly detector.

Generates 5 plots:
  1. ROC Curve + AUC
  2. Reconstruction error distribution per storm class (KDE)
  3. Detection rate bar chart per storm class
  4. Anomaly score over time for the most extreme storm period
  5. Feature-wise reconstruction error (quiet vs extreme)

Run from project root:
    python notebooks/evaluate_anomaly.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save to file, no window blocking
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ──
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "DATA" / "enriched.parquet"
PLOTS_DIR  = ROOT / "notebooks" / "Plots_Anomaly_Eval"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from models.anomaly_autoencoder import TransformerAutoencoder, ANOMALY_FEATURES, SEQ_LEN

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 1000   # sequences per class
BATCH    = 256    # inference batch size

print(f"Device : {DEVICE}", flush=True)

# ─────────────────────────────────────────────
# Load model, scaler, config
# ─────────────────────────────────────────────
with open(MODELS_DIR / "anomaly_config.json") as f:
    config = json.load(f)

with open(MODELS_DIR / "anomaly_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = TransformerAutoencoder(
    input_size = config["input_size"],
    seq_len    = config["seq_len"],
    d_model    = config["d_model"],
    latent_dim = config["latent_dim"],
    nhead      = config["nhead"],
    num_layers = config["num_layers"],
    dropout    = config["dropout"],
).to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR / "anomaly_model.pt", map_location=DEVICE))
model.eval()

RECON_THRESHOLD = config["recon_threshold"]
RECON_MAX       = config["recon_max"]
ALERT_GREEN     = config["alert_green"]
ALERT_YELLOW    = config["alert_yellow"]

print("Model, scaler and config loaded.", flush=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print(f"Loading {DATA_PATH} ...", flush=True)
df = pd.read_parquet(DATA_PATH)
print(f"Loaded : {len(df):,} rows", flush=True)

CLASS_BINS = {
    "quiet":    (df["dst"] > -30),
    "minor":    (df["dst"] <= -30)  & (df["dst"] > -50),
    "moderate": (df["dst"] <= -50)  & (df["dst"] > -100),
    "intense":  (df["dst"] <= -100) & (df["dst"] > -200),
    "extreme":  (df["dst"] <= -200),
}
CLASS_INT = {"quiet":0, "minor":1, "moderate":2, "intense":3, "extreme":4}
CLASS_COLORS = {
    "quiet":    "#2196F3",
    "minor":    "#4CAF50",
    "moderate": "#FF9800",
    "intense":  "#F44336",
    "extreme":  "#9C27B0",
}


# ─────────────────────────────────────────────
# Batched inference helper
# ─────────────────────────────────────────────
def score_class_batched(cls_df, n=N_SAMPLE):
    """Build N sequences, run batched inference, return (errors, scores)."""
    cls_df = cls_df.dropna(subset=ANOMALY_FEATURES)
    if len(cls_df) < SEQ_LEN:
        return np.array([]), np.array([])

    vals      = cls_df[ANOMALY_FEATURES].values.astype(np.float32)
    scaled    = scaler.transform(vals)
    max_start = len(scaled) - SEQ_LEN
    starts    = np.random.choice(max_start, size=min(n, max_start), replace=False)

    # Stack all sequences: (N, SEQ_LEN, n_features)
    seqs = np.stack([scaled[s:s + SEQ_LEN] for s in starts])

    errors = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = torch.tensor(seqs[i:i + BATCH], dtype=torch.float32).to(DEVICE)
            recon = model(batch)
            err   = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(err)

    errors = np.concatenate(errors)
    scores = np.clip((errors - RECON_THRESHOLD) / (RECON_MAX - RECON_THRESHOLD), 0, 1)
    return errors, scores


def feature_errors_batched(cls_df, n=500):
    """Return per-feature mean MSE for a class."""
    cls_df = cls_df.dropna(subset=ANOMALY_FEATURES)
    if len(cls_df) < SEQ_LEN:
        return None
    vals      = cls_df[ANOMALY_FEATURES].values.astype(np.float32)
    scaled    = scaler.transform(vals)
    max_start = len(scaled) - SEQ_LEN
    starts    = np.random.choice(max_start, size=min(n, max_start), replace=False)
    seqs      = np.stack([scaled[s:s + SEQ_LEN] for s in starts])

    per_feat = np.zeros(len(ANOMALY_FEATURES))
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = torch.tensor(seqs[i:i + BATCH], dtype=torch.float32).to(DEVICE)
            recon = model(batch)
            # mean over time axis → (batch, n_features)
            diff  = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
            per_feat += diff.sum(axis=0)

    return per_feat / len(starts)


# ─────────────────────────────────────────────
# Score all classes
# ─────────────────────────────────────────────
print("\nScoring sequences per class ...", flush=True)
all_errors   = {}
all_labels   = []
score_concat = []

for cls_name, mask in CLASS_BINS.items():
    errs, scrs = score_class_batched(df[mask])
    all_errors[cls_name] = errs
    all_labels.extend([CLASS_INT[cls_name]] * len(scrs))
    score_concat.extend(scrs.tolist())
    print(f"  {cls_name:10s}: {len(scrs):,} sequences", flush=True)

all_labels   = np.array(all_labels)
score_concat = np.array(score_concat)


# ═════════════════════════════════════════════
# Plot 1 — ROC Curve
# ═════════════════════════════════════════════
print("\n[Plot 1] ROC Curve ...", flush=True)
y_true  = (all_labels >= 2).astype(int)
y_score = score_concat
fpr, tpr, _ = roc_curve(y_true, y_score)
auc         = roc_auc_score(y_true, y_score)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Anomaly Detector\n(moderate+ vs quiet/minor)", fontsize=13)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_roc_curve.png", dpi=150)
plt.close()
print(f"  AUC = {auc:.3f}", flush=True)


# ═════════════════════════════════════════════
# Plot 2 — Reconstruction Error KDE per class
# ═════════════════════════════════════════════
print("[Plot 2] Error distribution ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 5))
for cls_name, errs in all_errors.items():
    if len(errs) == 0:
        continue
    sns.kdeplot(np.clip(errs, 0, 4.0), label=cls_name,
                color=CLASS_COLORS[cls_name], ax=ax, lw=2)
ax.axvline(RECON_THRESHOLD, color="black", linestyle="--", lw=1.5,
           label=f"threshold={RECON_THRESHOLD:.3f}")
ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Reconstruction Error Distribution by Storm Class", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_error_distribution.png", dpi=150)
plt.close()


# ═════════════════════════════════════════════
# Plot 3 — Detection Rate Bar Chart
# ═════════════════════════════════════════════
print("[Plot 3] Detection rates ...", flush=True)
detection_rates = {
    cls_name: (100.0 * (errs > RECON_THRESHOLD).mean() if len(errs) > 0 else 0.0)
    for cls_name, errs in all_errors.items()
}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    detection_rates.keys(), detection_rates.values(),
    color=[CLASS_COLORS[c] for c in detection_rates],
    edgecolor="white", linewidth=0.5,
)
for bar, val in zip(bars, detection_rates.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(100, color="gray", linestyle="--", lw=1)
ax.set_ylim(0, 115)
ax.set_ylabel("% Sequences Above Threshold", fontsize=12)
ax.set_title("Storm Detection Rate by Class", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_detection_rate.png", dpi=150)
plt.close()


# ═════════════════════════════════════════════
# Plot 4 — Storm Replay
# ═════════════════════════════════════════════
print("[Plot 4] Storm replay ...", flush=True)
extreme_mask = df["dst"] <= -200
if extreme_mask.any():
    worst_period = df.loc[df["dst"].idxmin(), "period"]
    storm_df     = df[df["period"] == worst_period].copy().reset_index(drop=True)
    print(f"  Period: {worst_period}  min_dst={storm_df['dst'].min():.1f} nT", flush=True)

    vals   = storm_df[ANOMALY_FEATURES].ffill().values.astype(np.float32)
    scaled = scaler.transform(vals)

    # Batched sliding window
    n_windows = len(scaled) - SEQ_LEN + 1
    seqs      = np.stack([scaled[i:i + SEQ_LEN] for i in range(n_windows)])

    replay_errors = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = torch.tensor(seqs[i:i + BATCH], dtype=torch.float32).to(DEVICE)
            recon = model(batch)
            err   = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            replay_errors.extend(err.tolist())

    replay_errors = np.array(replay_errors)
    replay_scores = np.clip(
        (replay_errors - RECON_THRESHOLD) / (RECON_MAX - RECON_THRESHOLD), 0, 1
    )

    # Pad front so index aligns with storm_df rows
    pad           = np.full(SEQ_LEN - 1, np.nan)
    replay_scores = np.concatenate([pad, replay_scores])
    time_axis     = np.arange(len(storm_df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.plot(time_axis, storm_df["dst"].values, color="steelblue", lw=1.5, label="Dst (nT)")
    ax1.axhline(-100, color="#F44336", linestyle="--", lw=1, label="intense (-100 nT)")
    ax1.axhline(-200, color="#9C27B0", linestyle="--", lw=1, label="extreme (-200 nT)")
    ax1.set_ylabel("Dst (nT)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(time_axis, replay_scores, color="darkorange", lw=1.5, label="Anomaly Score")
    ax2.axhline(ALERT_YELLOW, color="#F44336", linestyle="--", lw=1, label=f"RED  >{ALERT_YELLOW}")
    ax2.axhline(ALERT_GREEN,  color="gold",    linestyle="--", lw=1, label=f"YELLOW >{ALERT_GREEN}")
    valid = ~np.isnan(replay_scores)
    ax2.fill_between(time_axis, np.where(valid, replay_scores, 0),
                     where=valid & (replay_scores >= ALERT_YELLOW),
                     color="#F44336", alpha=0.3)
    ax2.fill_between(time_axis, np.where(valid, replay_scores, 0),
                     where=valid & (replay_scores >= ALERT_GREEN) & (replay_scores < ALERT_YELLOW),
                     color="gold", alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Anomaly Score", fontsize=11)
    ax2.set_xlabel("Time (minutes)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Storm Replay — {worst_period}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_storm_replay.png", dpi=150)
    plt.close()
else:
    print("  No extreme storms found — skipping.", flush=True)


# ═════════════════════════════════════════════
# Plot 5 — Feature-wise Reconstruction Error
# ═════════════════════════════════════════════
print("[Plot 5] Feature-wise errors ...", flush=True)
quiet_feat   = feature_errors_batched(df[CLASS_BINS["quiet"]])
extreme_feat = feature_errors_batched(df[CLASS_BINS["extreme"]])

if quiet_feat is not None and extreme_feat is not None:
    x     = np.arange(len(ANOMALY_FEATURES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, quiet_feat,   width, label="quiet",
           color=CLASS_COLORS["quiet"],   alpha=0.85)
    ax.bar(x + width/2, extreme_feat, width, label="extreme",
           color=CLASS_COLORS["extreme"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ANOMALY_FEATURES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Feature MSE", fontsize=12)
    ax.set_title("Per-Feature Reconstruction Error: Quiet vs Extreme Storms", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_errors.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EVALUATION SUMMARY")
print("=" * 55)
print(f"AUC Score        : {auc:.3f}")
print(f"Recon threshold  : {RECON_THRESHOLD:.4f}")
print(f"Recon max        : {RECON_MAX:.4f}")
print("\nDetection Rates:")
for cls_name, rate in detection_rates.items():
    bar = "█" * int(rate / 5)
    print(f"  {cls_name:10s}: {rate:5.1f}%  {bar}")
print(f"\nPlots saved to : {PLOTS_DIR}")
