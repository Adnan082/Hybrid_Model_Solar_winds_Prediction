"""
main.py
-------
Launches the full Solar Wind Dst prediction pipeline.

What it starts
--------------
  Thread 1 : BurtonAgent   — subscribes to solar_wind.raw
  Thread 2 : AnomalyAgent  — subscribes to burton.output
  Thread 3 : CorrectorAgent— subscribes to burton.output
  Thread 4 : FusionAgent   — subscribes to anomaly.output + ml.output + rl.reward
  Thread 5 : DataFeeder    — replays solar_wind.csv → solar_wind.raw
                             and labels.csv         → rl.reward  (actual Dst)

Usage
-----
  # Replay at full speed (backtesting / training the RL agent fast):
  python main.py

  # Replay at real-time speed (1 row = 1 second, simulates live feed):
  python main.py --speed 1.0

  # Run only a specific period from the dataset:
  python main.py --period train_a

  # Skip the data feeder (if you have a live feed publishing to Redis already):
  python main.py --no-feeder

Environment variables
---------------------
  REDIS_HOST  (default: localhost)
  REDIS_PORT  (default: 6379)
"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from event_bus.bus import EventBus, TOPICS
from agents.burton_agent import BurtonAgent
from agents.anomaly_agent import AnomalyAgent
from agents.corrector_agent import CorrectorAgent
from agents.fusion_agent import FusionAgent
from agents.rl_agent import RLBusAgent


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "DATA"
SW_PATH     = DATA_DIR / "solar_wind.csv"
LABELS_PATH = DATA_DIR / "labels.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data Feeder
# Reads solar_wind.csv row-by-row and publishes to solar_wind.raw.
# Matches hourly labels.csv rows → publishes actual Dst to rl.reward.
# ─────────────────────────────────────────────────────────────────────────────

def run_feeder(bus: EventBus, speed: float, period_filter: str | None,
               stop_event: threading.Event) -> None:
    """
    Replay solar_wind.csv through the event bus.

    Args:
        bus           : connected EventBus
        speed         : seconds to sleep between rows (0 = as fast as possible)
        period_filter : if set, only replay rows where period == period_filter
        stop_event    : set this to stop the feeder gracefully
    """
    logger.info(f"[DataFeeder] Loading {SW_PATH} ...")
    sw_df = pd.read_csv(SW_PATH)
    if period_filter:
        sw_df = sw_df[sw_df["period"] == period_filter].reset_index(drop=True)
        logger.info(f"[DataFeeder] Filtered to period '{period_filter}': {len(sw_df):,} rows")
    else:
        logger.info(f"[DataFeeder] {len(sw_df):,} rows across all periods")

    # Build a timedelta→dst lookup from labels.csv (hourly actual Dst)
    logger.info(f"[DataFeeder] Loading {LABELS_PATH} ...")
    labels_df = pd.read_csv(LABELS_PATH)
    if period_filter:
        labels_df = labels_df[labels_df["period"] == period_filter]

    # Index: (period, timedelta string) → dst value
    label_lookup: dict[tuple[str, str], float] = {
        (row["period"], row["timedelta"]): float(row["dst"])
        for _, row in labels_df.iterrows()
    }
    logger.info(f"[DataFeeder] {len(label_lookup):,} label entries loaded")

    published = 0
    for _, row in sw_df.iterrows():
        if stop_event.is_set():
            break

        payload = {
            "period":      str(row["period"]),
            "timedelta":   str(row["timedelta"]),
            "bz_gsm":      _safe_float(row.get("bz_gsm")),
            "by_gsm":      _safe_float(row.get("by_gsm")),
            "bt":          _safe_float(row.get("bt")),
            "speed":       _safe_float(row.get("speed"),  400.0),
            "density":     _safe_float(row.get("density"), 5.0),
            "temperature": _safe_float(row.get("temperature"), 1e5),
            "source":      str(row.get("source", "ace")),
        }
        bus.publish(TOPICS["raw"], payload)

        # If there is a matching label for this (period, timedelta), publish
        # it on rl.reward so the FusionAgent can train the RL policy.
        key = (payload["period"], payload["timedelta"])
        if key in label_lookup:
            reward_payload = {
                "dst_actual": label_lookup[key],
                "period":     payload["period"],
                "timedelta":  payload["timedelta"],
            }
            bus.publish(TOPICS["rl_reward"], reward_payload)

        published += 1
        if published % 10_000 == 0:
            logger.info(f"[DataFeeder] Published {published:,} rows ...")

        if speed > 0:
            time.sleep(speed)

    logger.info(f"[DataFeeder] Done. Published {published:,} rows total.")


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v == v else default   # NaN check
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Agent thread helper
# ─────────────────────────────────────────────────────────────────────────────

def _start_agent_thread(agent_cls, bus: EventBus, **kwargs) -> threading.Thread:
    """
    Instantiate an agent and run it in a daemon thread.
    Each agent gets its own EventBus connection (own pubsub handle).
    """
    agent_bus = EventBus(
        host=bus._r.connection_pool.connection_kwargs["host"],
        port=bus._r.connection_pool.connection_kwargs["port"],
    )
    agent = agent_cls(agent_bus, **kwargs)
    t = threading.Thread(target=agent.run, name=agent.name, daemon=True)
    t.start()
    logger.info(f"[main] {agent.name} thread started.")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Solar Wind Dst Prediction Pipeline")
    parser.add_argument(
        "--speed", type=float, default=0.0,
        help="Seconds to sleep between data rows. 0 = full speed (default). "
             "Use 1.0 to simulate real-time 1-min cadence.",
    )
    parser.add_argument(
        "--period", type=str, default=None,
        help="Only replay a specific period (e.g. train_a, test_a). "
             "Default: replay all periods.",
    )
    parser.add_argument(
        "--no-feeder", action="store_true",
        help="Skip the data feeder (use if you have a live Redis feed).",
    )
    args = parser.parse_args()

    # ── Redis connection ──────────────────────────────────────────────────────
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    logger.info(f"[main] Connecting to Redis at {redis_host}:{redis_port} ...")
    try:
        main_bus = EventBus(host=redis_host, port=redis_port)
        main_bus._r.ping()
        logger.info("[main] Redis OK.")
    except Exception as e:
        logger.error(f"[main] Cannot connect to Redis: {e}")
        logger.error("[main] Start Redis first:  redis-server")
        sys.exit(1)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    stop_event = threading.Event()

    def _shutdown(sig, frame):
        logger.info(f"[main] Signal {sig} received — shutting down ...")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start agents ──────────────────────────────────────────────────────────
    # Give each agent its own bus connection so their pubsub loops don't
    # interfere with each other.
    logger.info("[main] Starting agents ...")

    _start_agent_thread(BurtonAgent,    main_bus)
    _start_agent_thread(AnomalyAgent,   main_bus)
    _start_agent_thread(CorrectorAgent, main_bus)
    _start_agent_thread(FusionAgent,    main_bus)
    _start_agent_thread(RLBusAgent,     main_bus)

    # Short pause to let all agents subscribe before the feeder starts sending
    time.sleep(1.0)

    # ── Start data feeder ─────────────────────────────────────────────────────
    if not args.no_feeder:
        feeder_thread = threading.Thread(
            target=run_feeder,
            args=(main_bus, args.speed, args.period, stop_event),
            name="DataFeeder",
            daemon=True,
        )
        feeder_thread.start()
        logger.info("[main] DataFeeder thread started.")
    else:
        logger.info("[main] --no-feeder set, skipping DataFeeder.")
        feeder_thread = None

    # ── Keep main thread alive ────────────────────────────────────────────────
    logger.info(
        "[main] Pipeline running.  "
        "Press Ctrl+C to stop.\n"
        "  Prediction output topic : prediction.final\n"
        "  Redis key (latest)      : latest:prediction.final\n"
    )

    try:
        while not stop_event.is_set():
            time.sleep(1)
            if feeder_thread and not feeder_thread.is_alive():
                logger.info("[main] DataFeeder finished — all data replayed.")
                logger.info("[main] Agents still running. Press Ctrl+C to exit.")
                feeder_thread = None   # stop checking
    except KeyboardInterrupt:
        pass

    logger.info("[main] Shutdown complete.")


if __name__ == "__main__":
    main()
