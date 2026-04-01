"""
prepare_data.py
---------------
Runs the Burton ODE on the full 8.3M row solar wind dataset,
computes all engineered features, and saves the enriched dataset
as a parquet file for training Agent 2 (Anomaly) and Agent 3 (Corrector).

Output: DATA/enriched.parquet
Columns added:
  - dst_burton       : Burton ODE prediction
  - E_field          : solar wind electric field V*Bz
  - Q                : ring current injection rate
  - dDst_dt          : rate of change of burton Dst
  - residual         : dst_actual - dst_burton (training target for Agent 3)
  - storm_phase      : quiet / onset / main / recovery
  - rolling features : bz_mean_1h, bz_std_3h, E_mean_6h etc.
  - storm_weight     : loss weight per sample
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "DATA"


# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────

def load_data():
    logger.info("Loading raw CSV files...")
    solar_wind = pd.read_csv(DATA_DIR / "solar_wind.csv")
    labels     = pd.read_csv(DATA_DIR / "labels.csv")
    sunspots   = pd.read_csv(DATA_DIR / "sunspots.csv")
    sat_pos    = pd.read_csv(DATA_DIR / "satellite_pos.csv")

    for df in [solar_wind, labels, sunspots, sat_pos]:
        df["timedelta"] = pd.to_timedelta(df["timedelta"])

    logger.info(f"solar_wind: {solar_wind.shape}  labels: {labels.shape}")
    return solar_wind, labels, sunspots, sat_pos


# ──────────────────────────────────────────────
# 2. Merge all datasets
# ──────────────────────────────────────────────

def merge_all(solar_wind, labels, sunspots, sat_pos):
    logger.info("Merging datasets...")
    df = solar_wind.copy()

    # Merge hourly Dst labels
    df["timedelta_hour"] = df["timedelta"].dt.floor("h")
    lb = labels.rename(columns={"timedelta": "timedelta_hour"})
    df = df.merge(lb, on=["period", "timedelta_hour"], how="left")

    # Merge sunspots (monthly, forward-fill)
    df["timedelta_day"] = df["timedelta"].dt.floor("D")
    ss = sunspots.rename(columns={"timedelta": "timedelta_day"})
    parts = []
    for period in df["period"].unique():
        sub_df = df[df["period"] == period].sort_values("timedelta_day")
        sub_ss = ss[ss["period"] == period].sort_values("timedelta_day")
        merged = pd.merge_asof(sub_df, sub_ss[["timedelta_day", "smoothed_ssn"]],
                               on="timedelta_day", direction="backward")
        parts.append(merged)
    df = pd.concat(parts, ignore_index=True)

    # Merge satellite positions (daily, forward-fill)
    sp = sat_pos.rename(columns={"timedelta": "timedelta_day"})
    sp["timedelta_day"] = sp["timedelta_day"].dt.floor("D")
    parts = []
    for period in df["period"].unique():
        sub_df  = df[df["period"] == period].sort_values("timedelta_day")
        sub_sp  = sp[sp["period"] == period].sort_values("timedelta_day")
        sat_cols = [c for c in sub_sp.columns if c != "period"]
        merged  = pd.merge_asof(sub_df, sub_sp[sat_cols],
                                on="timedelta_day", direction="backward")
        parts.append(merged)
    df = pd.concat(parts, ignore_index=True)

    df.drop(columns=["timedelta_hour", "timedelta_day"], inplace=True)
    logger.info(f"Merged shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 3. Burton ODE solver
# ──────────────────────────────────────────────

def compute_burton(bz: np.ndarray, speed: np.ndarray,
                   tau_hours: float = 7.7,
                   dt_minutes: float = 1.0):
    """
    Solve Burton ODE on arrays of Bz and speed.

    Physics:
        E        = -V * Bz * 1e-3          (solar wind electric field, mV/m)
        Q(E)     = -4.4*(E - 0.49) if E > 0.49 else 0   (ring current injection)
        dDst/dt  = Q - Dst/tau             (ODE)

    Returns: dst_burton, E_field, Q, dDst_dt (all numpy arrays)
    """
    dt_hours = dt_minutes / 60.0
    n = len(bz)

    E = -speed * bz * 1e-3
    Q = np.where(E > 0.49, -4.4 * (E - 0.49), 0.0)

    dst_b  = np.zeros(n, dtype=np.float32)
    dDst   = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        dDst[i-1]  = Q[i-1] - dst_b[i-1] / tau_hours
        dst_b[i]   = dst_b[i-1] + dDst[i-1] * dt_hours

    return dst_b, E.astype(np.float32), Q.astype(np.float32), dDst


# ──────────────────────────────────────────────
# 4. Storm phase labeller
# ──────────────────────────────────────────────

def label_storm_phase(dst: np.ndarray, threshold: float = -30.0) -> np.ndarray:
    """
    Labels each timestep as: 0=quiet, 1=onset, 2=main, 3=recovery
    """
    phases = np.zeros(len(dst), dtype=np.int8)  # 0 = quiet
    i = 0
    while i < len(dst):
        if dst[i] <= threshold:
            window_end    = min(i + 48, len(dst))
            local_min_idx = i + int(np.argmin(dst[i:window_end]))
            # onset
            phases[i:local_min_idx] = 1
            # main
            start_main = max(0, local_min_idx - 2)
            end_main   = min(len(dst), local_min_idx + 3)
            phases[start_main:end_main] = 2
            # recovery
            j = local_min_idx + 3
            while j < len(dst) and dst[j] <= threshold:
                phases[j] = 3
                j += 1
            i = j
        else:
            i += 1
    return phases


# ──────────────────────────────────────────────
# 5. Storm-weighted loss weights
# ──────────────────────────────────────────────

def compute_storm_weight(dst: np.ndarray) -> np.ndarray:
    """
    Returns a per-sample loss weight based on storm severity.
    Extreme storms get 100x weight so the model focuses on them.
    """
    w = np.ones(len(dst), dtype=np.float32)
    w[dst <= -30]  = 3.0
    w[dst <= -50]  = 10.0
    w[dst <= -100] = 30.0
    w[dst <= -200] = 100.0
    return w


# ──────────────────────────────────────────────
# 6. Feature engineering
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")
    df = df.copy()

    # Fill short gaps in critical features (forward-fill up to 30 min)
    fill_cols = ["bz_gsm", "by_gsm", "bt", "speed", "density", "temperature"]
    for col in fill_cols:
        df[col] = df.groupby("period")[col].transform(
            lambda x: x.ffill(limit=30).bfill(limit=30)
        )
    # Remaining NaN → median
    for col in fill_cols:
        df[col] = df[col].fillna(df[col].median())

    # Solar wind electric field
    df["E_field"] = (-df["speed"] * df["bz_gsm"] * 1e-3).astype(np.float32)

    # Rolling features (computed per period to avoid leakage across periods)
    for period in df["period"].unique():
        mask = df["period"] == period
        sub  = df.loc[mask]

        for feat in ["bz_gsm", "speed", "E_field"]:
            for window, label in [(60, "1h"), (180, "3h"), (360, "6h")]:
                df.loc[mask, f"{feat}_mean_{label}"] = (
                    sub[feat].rolling(window, min_periods=1).mean().astype(np.float32)
                )
                df.loc[mask, f"{feat}_std_{label}"] = (
                    sub[feat].rolling(window, min_periods=1).std().fillna(0).astype(np.float32)
                )

    # Source as binary feature (ACE=0, DSCOVR=1)
    df["source_encoded"] = (df["source"] == "dscovr").astype(np.int8)

    logger.info("Rolling features computed.")
    return df


# ──────────────────────────────────────────────
# 7. Run Burton per period
# ──────────────────────────────────────────────

def run_burton(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running Burton ODE on full dataset...")
    parts = []

    for period in df["period"].unique():
        logger.info(f"  Processing {period}...")
        sub = df[df["period"] == period].copy().reset_index(drop=True)

        bz    = sub["bz_gsm"].values
        speed = sub["speed"].values

        dst_b, E, Q, dDst = compute_burton(bz, speed)

        sub["dst_burton"] = dst_b
        sub["E_field"]    = E
        sub["Q"]          = Q
        sub["dDst_dt"]    = dDst

        # Rolling Burton error (last 3 hours = 180 rows)
        # Only valid where actual Dst is available (hourly)
        # Interpolate to 1-min for rolling error signal
        dst_actual_1min = sub["dst"].ffill(limit=60).values
        burton_error    = dst_actual_1min - dst_b
        sub["rolling_error_3h"] = (
            pd.Series(burton_error)
            .rolling(180, min_periods=1).mean()
            .astype(np.float32)
            .values
        )

        # Storm phase (on hourly Dst, then forward-fill to 1-min)
        dst_hourly = sub["dst"].ffill(limit=60).fillna(0).values
        sub["storm_phase"] = label_storm_phase(dst_hourly)

        # Residual (training target for Agent 3) — only where Dst is available
        sub["residual"] = (sub["dst"] - sub["dst_burton"]).astype(np.float32)

        # Storm loss weight
        dst_filled = sub["dst"].ffill(limit=60).fillna(0).values
        sub["storm_weight"] = compute_storm_weight(dst_filled)

        parts.append(sub)
        logger.info(f"  {period} done. Burton RMSE: "
                    f"{np.sqrt(((sub['residual'].dropna())**2).mean()):.2f} nT")

    df_out = pd.concat(parts, ignore_index=True)
    logger.info(f"Burton complete. Output shape: {df_out.shape}")
    return df_out


# ──────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("prepare_data.py — Building enriched training dataset")
    logger.info("=" * 60)

    solar_wind, labels, sunspots, sat_pos = load_data()
    df = merge_all(solar_wind, labels, sunspots, sat_pos)
    df = engineer_features(df)
    df = run_burton(df)

    # Final feature summary
    feature_cols = [
        "bz_gsm", "by_gsm", "bt", "speed", "density", "temperature",
        "E_field", "Q", "dst_burton", "dDst_dt", "rolling_error_3h",
        "bz_gsm_mean_1h", "bz_gsm_std_3h", "speed_mean_3h", "E_field_mean_6h",
        "smoothed_ssn", "source_encoded", "storm_phase", "storm_weight",
        "gse_x_ace", "gse_y_ace",
        "dst", "residual"
    ]
    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Final features: {available}")

    out_path = DATA_DIR / "enriched.parquet"
    df[available + ["period", "timedelta"]].to_parquet(out_path, index=False)
    logger.info(f"Saved enriched dataset to {out_path}")
    logger.info(f"Shape: {df.shape}  Size: {out_path.stat().st_size / 1e6:.1f} MB")
    logger.info("Done.")


if __name__ == "__main__":
    main()
