"""
Generate architecture diagram PNG using matplotlib.
Run: python architecture/generate_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "architecture.png"

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#060b18"
PANEL     = "#0d1b2e"
BORDER    = "#1a3a5c"
CYAN      = "#00d4ff"
GREEN     = "#00ff88"
ORANGE    = "#ff8c00"
PURPLE    = "#9b59b6"
RED       = "#e74c3c"
YELLOW    = "#f1c40f"
TEAL      = "#1abc9c"
WHITE     = "#e8f4f8"
GREY      = "#4a6fa5"

fig, ax = plt.subplots(figsize=(20, 13))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")


def box(x, y, w, h, color, label, sublabel=None, fontsize=9.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.5, edgecolor=color,
                          facecolor=PANEL)
    ax.add_patch(rect)
    ty = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, ty, label,
            ha="center", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            fontfamily="monospace")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center",
                color=WHITE, fontsize=7.5, alpha=0.75,
                fontfamily="monospace")


def arrow(x1, y1, x2, y2, color=GREY, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, color=color,
                fontsize=7, alpha=0.85, fontfamily="monospace")


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(10, 12.5, "Solar Wind Dst Prediction — System Architecture",
        ha="center", va="center", color=CYAN,
        fontsize=15, fontweight="bold", fontfamily="monospace")
ax.text(10, 12.15, "5-Agent Hybrid ML Pipeline  |  Redis Event Bus  |  Real-Time Dashboard",
        ha="center", va="center", color=WHITE,
        fontsize=9, alpha=0.7, fontfamily="monospace")

# ── Data Source ───────────────────────────────────────────────────────────────
box(0.4, 10.5, 2.8, 0.9, YELLOW, "DATA SOURCE", "NASA OMNI / solar_wind.csv")
arrow(3.2, 10.95, 4.0, 10.95, YELLOW, "solar_wind.raw")

# ── Event Bus ─────────────────────────────────────────────────────────────────
box(4.0, 10.3, 3.5, 1.3, CYAN, "EVENT BUS", "Redis Pub/Sub")
ax.text(5.75, 10.0, "Topics: raw | burton | anomaly | ml | fusion | final | rl_reward",
        ha="center", color=CYAN, fontsize=7, alpha=0.7, fontfamily="monospace")

# ── Agent 1: Burton ────────────────────────────────────────────────────────────
box(0.4, 7.8, 2.8, 1.3, GREEN, "AGENT 1", "Burton ODE Solver")
ax.text(1.8, 8.25, "Physics: tau=7.7hr\nEuler integration",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Agent 2: Anomaly ───────────────────────────────────────────────────────────
box(4.1, 7.8, 2.8, 1.3, ORANGE, "AGENT 2", "Anomaly Detector")
ax.text(5.5, 8.25, "Transformer AE\nseq=60, features=8",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Agent 3: Corrector ────────────────────────────────────────────────────────
box(7.8, 7.8, 2.8, 1.3, PURPLE, "AGENT 3", "BiLSTM Corrector")
ax.text(9.2, 8.25, "Residual correction\nseq=120, features=17",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Agent 4: Fusion ───────────────────────────────────────────────────────────
box(4.1, 5.5, 2.8, 1.3, TEAL, "AGENT 4", "Fusion Agent")
ax.text(5.5, 5.95, "Merges: burton +\nanomaly + ml outputs",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Agent 5: RL ───────────────────────────────────────────────────────────────
box(4.1, 3.2, 2.8, 1.8, RED, "AGENT 5", "RL Blend Agent")
ax.text(5.5, 4.0, "Actor-Critic PolicyNet\nState: 9-dim\nAction: [w_burton, w_corr]\nReplay: 20K buffer",
        ha="center", color=WHITE, fontsize=7.2, alpha=0.8, fontfamily="monospace")

# ── Redis State Store ─────────────────────────────────────────────────────────
box(11.0, 7.8, 2.8, 1.3, CYAN, "REDIS STORE", "State + History")
ax.text(12.4, 8.25, "prediction:latest\nhistory:predictions",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── FastAPI ───────────────────────────────────────────────────────────────────
box(14.3, 7.8, 2.8, 1.3, GREEN, "FASTAPI", "REST + WebSocket")
ax.text(15.7, 8.25, "/api/v1/prediction\n/ws/live  /metrics",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Dashboard ─────────────────────────────────────────────────────────────────
box(17.2, 7.8, 2.5, 1.3, YELLOW, "DASHBOARD", "Dash + Plotly")
ax.text(18.45, 8.25, "localhost:8050\nNASA dark theme",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── MLflow ────────────────────────────────────────────────────────────────────
box(11.0, 5.5, 2.8, 1.3, ORANGE, "MLFLOW", "Experiment Tracking")
ax.text(12.4, 5.95, "RMSE/MAE per storm\nclass + RL maturity",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Model Files ───────────────────────────────────────────────────────────────
box(11.0, 3.2, 2.8, 1.8, PURPLE, "MODELS", "Trained Weights")
ax.text(12.4, 4.0, "anomaly_autoencoder.pt\nlstm_corrector.pt\nrl_agent.pt\nStandardScaler",
        ha="center", color=WHITE, fontsize=7.2, alpha=0.8, fontfamily="monospace")

# ── Storm Classifier box ──────────────────────────────────────────────────────
box(14.3, 5.5, 2.8, 1.3, RED, "STORM CLASS", "Real-Time Alert")
ax.text(15.7, 5.95, "QUIET/MINOR/MODERATE\nINTENSE/EXTREME",
        ha="center", color=WHITE, fontsize=7.5, alpha=0.8, fontfamily="monospace")

# ── Arrows: Data Source → Event Bus ───────────────────────────────────────────
arrow(3.2, 10.95, 4.0, 10.95, YELLOW)

# Event Bus → Agents
arrow(5.75, 10.3, 1.8,  9.1,  CYAN,   "raw")
arrow(5.75, 10.3, 5.5,  9.1,  CYAN,   "raw")
arrow(5.75, 10.3, 9.2,  9.1,  CYAN,   "raw")

# Agents → Fusion
arrow(1.8,  7.8,  5.0,  6.8,  GREEN,  "burton.output")
arrow(5.5,  7.8,  5.5,  6.8,  ORANGE, "anomaly.output")
arrow(9.2,  7.8,  6.2,  6.8,  PURPLE, "ml.output")

# Fusion → RL
arrow(5.5,  5.5,  5.5,  5.0,  TEAL,   "fusion.output")

# RL → Redis
arrow(6.9,  4.1,  11.0, 8.45, RED,    "prediction.final")

# Redis → FastAPI
arrow(13.8, 8.45, 14.3, 8.45, CYAN)

# FastAPI → Dashboard
arrow(17.1, 8.45, 17.2, 8.45, GREEN)

# RL → MLflow
arrow(6.9,  3.8,  11.0, 6.15, ORANGE, "metrics")

# Models → Agents (load)
arrow(11.0, 4.1,  7.8,  8.45, PURPLE, "load weights")
arrow(11.0, 4.1,  4.1,  8.45, PURPLE)

# Storm classifier from FastAPI
arrow(15.7, 7.8,  15.7, 6.8,  RED)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (GREEN,  "Physics Agent"),
    (ORANGE, "Anomaly Agent"),
    (PURPLE, "ML Agent"),
    (TEAL,   "Fusion Agent"),
    (RED,    "RL Agent"),
    (CYAN,   "Infrastructure"),
    (YELLOW, "Data / UI"),
]
lx, ly = 0.4, 2.8
ax.text(lx, ly + 0.3, "LEGEND", color=WHITE, fontsize=8,
        fontweight="bold", fontfamily="monospace")
for i, (c, lbl) in enumerate(legend_items):
    rx = lx + (i % 4) * 3.0
    ry = ly - (i // 4) * 0.55
    rect = FancyBboxPatch((rx, ry - 0.15), 0.35, 0.3,
                          boxstyle="round,pad=0.04",
                          linewidth=1, edgecolor=c, facecolor=PANEL)
    ax.add_patch(rect)
    ax.text(rx + 0.5, ry, lbl, color=c, fontsize=7.8,
            va="center", fontfamily="monospace")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
plt.savefig(str(OUT), dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved: {OUT}")
