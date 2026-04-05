"""Tests for FusionAgent — cache logic and merge triggering."""

import pytest
from agents.fusion_agent import FusionAgent
from event_bus.bus import TOPICS


def test_returns_none_until_all_caches_populated(mock_bus, burton_output_payload,
                                                  anomaly_output_payload):
    agent = FusionAgent(mock_bus)

    # Only Burton arrives
    r1 = agent.process(TOPICS["burton"], burton_output_payload)
    assert r1 is None

    # Anomaly arrives — ml still missing
    r2 = agent.process(TOPICS["anomaly"], anomaly_output_payload)
    assert r2 is None


def test_fires_only_on_ml_topic(mock_bus, burton_output_payload,
                                 anomaly_output_payload, ml_output_payload):
    agent = FusionAgent(mock_bus)
    agent._cache[TOPICS["burton"]]  = burton_output_payload
    agent._cache[TOPICS["anomaly"]] = anomaly_output_payload
    agent._cache[TOPICS["ml"]]      = ml_output_payload

    # Burton update with all caches full → None (not ml topic)
    r = agent.process(TOPICS["burton"], burton_output_payload)
    assert r is None

    # ML update → fires merge
    r2 = agent.process(TOPICS["ml"], ml_output_payload)
    assert r2 is not None


def test_merge_contains_all_source_fields(mock_bus, burton_output_payload,
                                           anomaly_output_payload, ml_output_payload):
    agent = FusionAgent(mock_bus)
    agent._cache[TOPICS["burton"]]  = burton_output_payload
    agent._cache[TOPICS["anomaly"]] = anomaly_output_payload

    result = agent.process(TOPICS["ml"], ml_output_payload)

    assert "dst_burton"    in result
    assert "anomaly_score" in result
    assert "dst_corrector" in result
    assert "storm_phase"   in result
    assert "E_field"       in result


def test_dst_corrector_equals_dst_final_from_ml(mock_bus, burton_output_payload,
                                                  anomaly_output_payload, ml_output_payload):
    agent = FusionAgent(mock_bus)
    agent._cache[TOPICS["burton"]]  = burton_output_payload
    agent._cache[TOPICS["anomaly"]] = anomaly_output_payload

    result = agent.process(TOPICS["ml"], ml_output_payload)
    assert result["dst_corrector"] == ml_output_payload["dst_final"]


def test_output_topic(mock_bus):
    assert FusionAgent(mock_bus).output_topic == TOPICS["fusion"]


def test_reset_clears_all_caches(mock_bus, burton_output_payload,
                                  anomaly_output_payload, ml_output_payload):
    agent = FusionAgent(mock_bus)
    agent._cache[TOPICS["burton"]]  = burton_output_payload
    agent._cache[TOPICS["anomaly"]] = anomaly_output_payload
    agent._cache[TOPICS["ml"]]      = ml_output_payload
    agent.reset()
    assert all(v is None for v in agent._cache.values())
