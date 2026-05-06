"""
live_feeder.py
--------------
Polls NOAA SWPC real-time solar wind JSON feeds every 60 seconds
and publishes live observations to the Redis event bus.

NOAA feeds used (free, no API key):
  Magnetic field : https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json
  Plasma/wind    : https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json
  Kyoto Dst      : https://services.swpc.noaa.gov/products/kyoto-dst.json

Data latency:
  Solar wind (DSCOVR/ACE at L1) : ~5 minutes behind real-time
  Kyoto Dst                      : ~1 hour (provisional real-time index)

Published topics:
  solar_wind.raw  : one payload per new 1-minute observation
  rl.reward       : one payload per new Kyoto Dst hourly value

Usage:
  python live_feeder.py          # standalone
  python main.py --live          # wired into full pipeline
"""

from __future__ import annotations

import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from event_bus.bus import EventBus, TOPICS


# ── NOAA endpoints ─────────────────────────────────────────────────────────────
_MAG_URL    = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
_WIND_URL   = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
_DST_URL    = "https://services.swpc.noaa.gov/products/kyoto-dst.json"

_POLL_INTERVAL   = 60    # seconds between polls
_REQUEST_TIMEOUT = 15    # seconds before giving up on a request
_MAX_CATCHUP     = 5     # max new rows to publish per poll (avoids bursting after outage)
_WARMUP_ROWS     = 150   # historical rows to replay on first startup (warms up agents)
_WARMUP_DELAY    = 0.05  # seconds between warmup rows (fast replay, ~7s total)
_HEADERS         = {"User-Agent": "SolarWindDstPipeline/1.0 (research)"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v == v else default   # NaN check
    except (TypeError, ValueError):
        return default


def _fetch_json(url: str) -> list | None:
    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=_HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[LiveFeeder] Fetch failed {url}: {e}")
        return None


class LiveFeeder:
    """
    Polls NOAA SWPC RTSW feeds and publishes new 1-minute observations
    to Redis. Tracks the last published timestamp to avoid duplicates.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._last_sw_tag:  str | None = None
        self._last_dst_tag: str | None = None
        self._sw_published  = 0
        self._dst_published = 0

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, stop_event: threading.Event | None = None) -> None:
        logger.info("[LiveFeeder] Starting — polling NOAA SWPC every 60s")
        logger.info(f"[LiveFeeder] Mag  : {_MAG_URL}")
        logger.info(f"[LiveFeeder] Wind : {_WIND_URL}")
        logger.info(f"[LiveFeeder] Dst  : {_DST_URL}")

        # Replay last 150 rows of NOAA history so agents warm up immediately
        self._warmup()

        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                self._poll_solar_wind()
                self._poll_dst()
            except Exception as e:
                logger.exception(f"[LiveFeeder] Unexpected error: {e}")

            # Sleep 1s at a time so stop_event is checked promptly
            for _ in range(_POLL_INTERVAL):
                if stop_event and stop_event.is_set():
                    return
                time.sleep(1)

    # ── Warmup: replay recent NOAA history so agents have full windows ─────────

    def _warmup(self) -> None:
        logger.info(f"[LiveFeeder] Warming up — fetching last {_WARMUP_ROWS} rows from NOAA history ...")
        mag_data  = _fetch_json(_MAG_URL)
        wind_data = _fetch_json(_WIND_URL)
        if mag_data is None or wind_data is None:
            logger.warning("[LiveFeeder] Warmup skipped — fetch failed")
            return

        mag_by_tag  = {r["time_tag"]: r for r in mag_data  if r.get("time_tag")}
        wind_by_tag = {r["time_tag"]: r for r in wind_data if r.get("time_tag")}
        common = sorted(set(mag_by_tag) & set(wind_by_tag))
        if not common:
            logger.warning("[LiveFeeder] Warmup skipped — no overlapping timestamps")
            return

        warmup_tags = common[-_WARMUP_ROWS:]
        logger.info(
            f"[LiveFeeder] Replaying {len(warmup_tags)} rows: "
            f"{warmup_tags[0]} to {warmup_tags[-1]}"
        )

        for tag in warmup_tags:
            self._publish_sw_row(tag, mag_by_tag[tag], wind_by_tag[tag])
            time.sleep(_WARMUP_DELAY)

        # Mark the last warmup row so normal polling only picks up genuinely new rows
        self._last_sw_tag = warmup_tags[-1]
        logger.info(
            f"[LiveFeeder] Warmup complete — {self._sw_published} rows published. "
            f"Agents are now warmed up. Switching to 60s live polling."
        )

    # ── Solar wind ─────────────────────────────────────────────────────────────

    def _poll_solar_wind(self) -> None:
        mag_data  = _fetch_json(_MAG_URL)
        wind_data = _fetch_json(_WIND_URL)
        if mag_data is None or wind_data is None:
            logger.warning("[LiveFeeder] Skipping solar wind poll — fetch failed")
            return

        # Index by time_tag
        mag_by_tag  = {r["time_tag"]: r for r in mag_data  if r.get("time_tag")}
        wind_by_tag = {r["time_tag"]: r for r in wind_data if r.get("time_tag")}

        # Only timestamps present in both feeds
        common = sorted(set(mag_by_tag) & set(wind_by_tag))
        if not common:
            logger.warning("[LiveFeeder] No overlapping timestamps in mag+wind")
            return

        # Determine new rows (after warmup, _last_sw_tag is always set)
        if self._last_sw_tag:
            new_tags = [t for t in common if t > self._last_sw_tag]
        else:
            new_tags = [common[-1]]

        new_tags = new_tags[-_MAX_CATCHUP:]   # cap burst

        for tag in new_tags:
            self._publish_sw_row(tag, mag_by_tag[tag], wind_by_tag[tag])

        if new_tags:
            self._last_sw_tag = new_tags[-1]
            logger.info(
                f"[LiveFeeder] {len(new_tags)} solar wind row(s) published  "
                f"latest={self._last_sw_tag}  total={self._sw_published}"
            )
        else:
            logger.debug(f"[LiveFeeder] No new solar wind rows (last={self._last_sw_tag})")

    def _publish_sw_row(self, tag: str, mag: dict, wind: dict) -> None:
        # Skip rows with known quality issues (flag > 0 means suspect data)
        if _safe_float(mag.get("overall_quality"),  0) > 0:
            logger.debug(f"[LiveFeeder] Skipping bad quality mag row at {tag}")
            return
        if _safe_float(wind.get("overall_quality"), 0) > 0:
            logger.debug(f"[LiveFeeder] Skipping bad quality wind row at {tag}")
            return

        bz_gsm  = _safe_float(mag.get("bz_gsm"),          0.0)
        by_gsm  = _safe_float(mag.get("by_gsm"),          0.0)
        bt      = _safe_float(mag.get("bt"),               5.0)
        speed   = _safe_float(wind.get("proton_speed"),  400.0)
        density = _safe_float(wind.get("proton_density"), 5.0)
        temp    = _safe_float(wind.get("proton_temperature"), 1e5)

        # Source is whichever satellite is currently active
        source = str(mag.get("source", "dscovr")).lower()

        payload = {
            "period":      "live",
            "timedelta":   tag,
            "bz_gsm":      bz_gsm,
            "by_gsm":      by_gsm,
            "bt":          bt,
            "speed":       speed,
            "density":     density,
            "temperature": temp,
            "source":      source,
            "feed_time":   datetime.now(timezone.utc).isoformat(),
        }

        self.bus.publish(TOPICS["raw"], payload)
        self._sw_published += 1

        logger.info(
            f"[LiveFeeder] SW  {tag}  "
            f"bz={bz_gsm:+.1f}nT  bt={bt:.1f}nT  "
            f"speed={speed:.0f}km/s  density={density:.1f}/cc  "
            f"src={source}"
        )

    # ── Kyoto Dst ──────────────────────────────────────────────────────────────

    def _poll_dst(self) -> None:
        dst_data = _fetch_json(_DST_URL)
        if not dst_data:
            return

        # Returns list of {"time_tag": "...", "dst": float}
        dst_by_tag = {
            r["time_tag"]: r
            for r in dst_data
            if r.get("time_tag") and r.get("dst") is not None
        }
        if not dst_by_tag:
            return

        all_tags = sorted(dst_by_tag)

        if self._last_dst_tag:
            new_tags = [t for t in all_tags if t > self._last_dst_tag]
        else:
            new_tags = [all_tags[-1]]

        for tag in new_tags:
            dst_val = _safe_float(dst_by_tag[tag].get("dst"))
            self.bus.publish(TOPICS["rl_reward"], {
                "dst_actual": dst_val,
                "period":     "live",
                "timedelta":  tag,
            })
            self._dst_published += 1
            logger.info(
                f"[LiveFeeder] Dst {tag}  dst={dst_val:.1f} nT  "
                f"total_dst={self._dst_published}"
            )

        if new_tags:
            self._last_dst_tag = new_tags[-1]


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    LiveFeeder(bus).run()
