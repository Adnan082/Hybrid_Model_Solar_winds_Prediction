"""Tests for BurtonAgent — pure physics, no Redis needed."""

import pytest
from agents.burton_agent import BurtonAgent
from event_bus.bus import TOPICS


def test_southward_bz_generates_injection(mock_bus, burton_raw_payload):
    agent  = BurtonAgent(mock_bus)
    result = agent.process(TOPICS["raw"], burton_raw_payload)
    # E = 600 * 15 * 1e-3 = 9 mV/m > 0.49 threshold → Q < 0
    assert result["E_field"] > 0.49
    assert result["Q"] < 0


def test_northward_bz_no_injection(mock_bus):
    agent   = BurtonAgent(mock_bus)
    payload = {"bz_gsm": 5.0, "speed": 400.0}
    result  = agent.process(TOPICS["raw"], payload)
    assert result["Q"] == 0.0


def test_dst_decreases_during_sustained_storm(mock_bus):
    agent   = BurtonAgent(mock_bus)
    payload = {"bz_gsm": -20.0, "speed": 700.0}
    prev_dst = 0.0
    for _ in range(60):
        result   = agent.process(TOPICS["raw"], payload)
        prev_dst = result["dst_burton"]
    assert prev_dst < -10.0   # Dst should have dropped significantly


def test_passthrough_fields(mock_bus, burton_raw_payload):
    agent  = BurtonAgent(mock_bus)
    result = agent.process(TOPICS["raw"], burton_raw_payload)
    assert result["period"]   == burton_raw_payload["period"]
    assert result["bz_gsm"]   == burton_raw_payload["bz_gsm"]
    assert result["speed"]    == burton_raw_payload["speed"]
    assert result["source"]   == burton_raw_payload["source"]


def test_output_topic(mock_bus):
    agent = BurtonAgent(mock_bus)
    assert agent.output_topic == TOPICS["burton"]


def test_input_topic(mock_bus):
    agent = BurtonAgent(mock_bus)
    assert TOPICS["raw"] in agent.input_topics


def test_reset_clears_state(mock_bus, burton_raw_payload):
    agent = BurtonAgent(mock_bus)
    # build up some Dst
    for _ in range(30):
        agent.process(TOPICS["raw"], {"bz_gsm": -20.0, "speed": 700.0})
    assert agent._dst != 0.0
    agent.reset()
    assert agent._dst == 0.0
    assert agent._step == 0
