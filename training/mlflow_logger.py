"""
training/mlflow_logger.py
-------------------------
MLflow tracking helpers used by run_pipeline.py and training scripts.

Tracks:
  - Model configs and hyperparameters
  - Per-storm-class RMSE / MAE
  - RL agent reward steps and blend weights
  - Artifacts: plots, CSV output, model configs
"""

from __future__ import annotations
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

mlflow.set_tracking_uri(str(ROOT / "mlruns"))


def start_pipeline_run(period: str | None = None) -> mlflow.ActiveRun:
    """Start an MLflow run for a full pipeline evaluation."""
    mlflow.set_experiment("solar_wind_pipeline")
    run = mlflow.start_run(run_name=f"pipeline_{period or 'all'}")

    # Log model configs as params
    for cfg_file, prefix in [
        ("anomaly_config.json",   "anomaly"),
        ("corrector_config.json", "corrector"),
    ]:
        path = MODELS_DIR / cfg_file
        if path.exists():
            cfg = json.loads(path.read_text())
            for k, v in cfg.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(f"{prefix}.{k}", v)

    mlflow.log_param("period", period or "all")
    return run


def log_pipeline_metrics(results_df, period: str = "all") -> None:
    """Log per-storm-class RMSE/MAE from run_pipeline results dataframe."""
    storm_bins = {
        "quiet":    (lambda d: d > -30),
        "minor":    (lambda d: (d <= -30) & (d > -50)),
        "moderate": (lambda d: (d <= -50) & (d > -100)),
        "intense":  (lambda d: (d <= -100) & (d > -200)),
        "extreme":  (lambda d: d <= -200),
    }

    def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
    def mae(e):  return float(np.mean(np.abs(e)))

    for cls, mask_fn in storm_bins.items():
        mask = mask_fn(results_df["dst"].values)
        sub  = results_df[mask]
        if len(sub) == 0:
            continue
        mlflow.log_metrics({
            f"{cls}.rmse_burton":    rmse(sub["err_burton"].values),
            f"{cls}.rmse_corrector": rmse(sub["err_corrector"].values),
            f"{cls}.rmse_final":     rmse(sub["err_final"].values),
            f"{cls}.mae_final":      mae(sub["err_final"].values),
            f"{cls}.n_samples":      len(sub),
        })

    # Overall
    mlflow.log_metrics({
        "overall.rmse_burton":    rmse(results_df["err_burton"].values),
        "overall.rmse_corrector": rmse(results_df["err_corrector"].values),
        "overall.rmse_final":     rmse(results_df["err_final"].values),
        "overall.mae_final":      mae(results_df["err_final"].values),
    })

    # RL stats
    mlflow.log_metrics({
        "rl.w_burton_mean":      float(results_df["w_burton"].mean()),
        "rl.w_corrector_mean":   float(results_df["w_corrector"].mean()),
        "rl.blend_certainty_mean": float(results_df.get("blend_certainty",
                                         results_df["w_corrector"]).mean()),
    })


def log_pipeline_artifacts(plots_dir: Path) -> None:
    """Log all PNG plots and the CSV output as MLflow artifacts."""
    if plots_dir.exists():
        for f in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(f), artifact_path="plots")
        csv = plots_dir / "pipeline_output.csv"
        if csv.exists():
            mlflow.log_artifact(str(csv), artifact_path="results")


def log_rl_checkpoint(rl_reward_steps: int, avg_reward: float | None = None) -> None:
    """Log RL agent maturity metrics."""
    mlflow.log_metric("rl.reward_steps", rl_reward_steps)
    if avg_reward is not None:
        mlflow.log_metric("rl.avg_reward", avg_reward)
    ckpt = MODELS_DIR / "rl_agent.pt"
    if ckpt.exists():
        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
