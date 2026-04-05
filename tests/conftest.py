"""tests/conftest.py — Shared fixtures for all tests."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Mock EventBus ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_bus():
    """EventBus with Redis replaced by a MagicMock — no real Redis needed."""
    from event_bus.bus import EventBus
    bus = EventBus.__new__(EventBus)
    bus._r      = MagicMock()
    bus._pubsub = MagicMock()
    bus._r.ping.return_value = True
    return bus


# ── Sample payloads ───────────────────────────────────────────────────────────

@pytest.fixture
def burton_raw_payload():
    return {
        "period": "test", "timedelta": "0 days 00:01:00",
        "bz_gsm": -15.0, "by_gsm": 3.0, "bt": 18.0,
        "speed": 600.0, "density": 8.0, "temperature": 2e5, "source": "ace",
    }


@pytest.fixture
def burton_output_payload():
    return {
        "period": "test", "timedelta": "0 days 00:01:00",
        "dst_burton": -42.5, "dst_prev": -40.0,
        "E_field": 9.0, "Q": -37.0, "dDst_dt": -4.8,
        "bz_gsm": -15.0, "speed": 600.0, "by_gsm": 3.0,
        "bt": 18.0, "density": 8.0, "temperature": 2e5, "source": "ace",
    }


@pytest.fixture
def anomaly_output_payload():
    return {
        "period": "test", "timedelta": "0 days 00:01:00",
        "recon_error": 0.45, "anomaly_score": 0.06,
        "alert_level": "GREEN", "dst_burton": -42.5,
        "bz_gsm": -15.0, "speed": 600.0, "E_field": 9.0,
    }


@pytest.fixture
def ml_output_payload():
    return {
        "period": "test", "timedelta": "0 days 00:01:00",
        "dst_burton": -42.5, "residual_pred": -8.3, "dst_final": -50.8,
        "storm_phase": 2, "bz_gsm": -15.0, "speed": 600.0, "E_field": 9.0,
    }


@pytest.fixture
def fusion_payload():
    return {
        "period": "test", "timedelta": "0 days 00:01:00",
        "dst_burton": -42.5, "dst_corrector": -50.8,
        "anomaly_score": 0.06, "alert_level": "GREEN",
        "recon_error": 0.45, "residual_pred": -8.3,
        "storm_phase": 2, "E_field": 9.0,
        "Q": -37.0, "dDst_dt": -4.8,
        "bz_gsm": -15.0, "speed": 600.0,
    }


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    """FastAPI TestClient with Redis mocked."""
    from fastapi.testclient import TestClient
    from api.main import app
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.exists.return_value = 1
    mock_r.get.return_value = None
    mock_r.object_idletime.return_value = 5
    with patch("api.dependencies.get_redis", return_value=iter([mock_r])):
        with TestClient(app) as client:
            yield client, mock_r
