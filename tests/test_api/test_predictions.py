"""Tests for FastAPI prediction endpoints."""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


MOCK_PREDICTION = {
    "period": "test", "timedelta": "0 days 00:01:00",
    "dst_pred": -52.3, "dst_burton": -42.5, "dst_corrector": -50.8,
    "residual_pred": -8.3, "anomaly_score": 0.06,
    "alert_level": "GREEN", "recon_error": 0.045,
    "w_burton": 0.35, "w_corrector": 0.65, "blend_certainty": 0.72,
    "confidence": 0.94, "storm_phase": 2, "E_field": 9.0,
    "bz_gsm": -15.0, "speed": 600.0, "dDst_dt": -4.8, "rl_reward_steps": 120,
}


@pytest.fixture
def client_with_data():
    mock_r = MagicMock()
    mock_r.get.return_value = json.dumps(MOCK_PREDICTION)
    mock_r.ping.return_value = True
    mock_r.exists.return_value = 1
    with patch("api.routers.predictions.get_redis", return_value=lambda: mock_r):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_r


@pytest.fixture
def client_no_data():
    mock_r = MagicMock()
    mock_r.get.return_value = None
    mock_r.ping.return_value = True
    mock_r.exists.return_value = 0
    with patch("api.routers.predictions.get_redis", return_value=lambda: mock_r):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_r


def test_health_returns_200():
    from api.main import app
    from api.dependencies import get_redis
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.exists.return_value = 1

    def override():
        yield mock_r

    app.dependency_overrides[get_redis] = override
    with TestClient(app) as c:
        resp = c.get("/health")
    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["redis"] is True


def test_storm_classification_quiet():
    from api.dependencies import classify_storm
    cls, _ = classify_storm(-10.0)
    assert cls == "QUIET"


def test_storm_classification_moderate():
    from api.dependencies import classify_storm
    cls, _ = classify_storm(-75.0)
    assert cls == "MODERATE"


def test_storm_classification_extreme():
    from api.dependencies import classify_storm
    cls, _ = classify_storm(-250.0)
    assert cls == "EXTREME"


def test_storm_classification_all_boundaries():
    from api.dependencies import classify_storm
    assert classify_storm(-29.0)[0]  == "QUIET"
    assert classify_storm(-31.0)[0]  == "MINOR"
    assert classify_storm(-51.0)[0]  == "MODERATE"
    assert classify_storm(-101.0)[0] == "INTENSE"
    assert classify_storm(-201.0)[0] == "EXTREME"
