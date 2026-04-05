"""Tests for RLAgent and RLBusAgent."""

import math
import pytest
import torch
import numpy as np
from agents.rl_agent import RLAgent, RLBusAgent, PolicyNet, STATE_DIM, ACTION_DIM
from event_bus.bus import TOPICS


# ── PolicyNet ────────────────────────────────────────────────���───────────────

def test_policy_net_output_shapes():
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    x   = torch.randn(4, STATE_DIM)
    weights, value = net(x)
    assert weights.shape == (4, ACTION_DIM)
    assert value.shape   == (4,)


def test_policy_net_weights_sum_to_one():
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    x   = torch.randn(8, STATE_DIM)
    weights, _ = net(x)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5)


def test_policy_net_weights_non_negative():
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    x   = torch.randn(16, STATE_DIM)
    weights, _ = net(x)
    assert (weights >= 0).all()


# ── RLAgent ───────────────────────────────────────────────────────────────────

@pytest.fixture
def rl():
    return RLAgent(device="cpu")


def test_act_returns_valid_weights(rl, fusion_payload):
    weights = rl.act(fusion_payload)
    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < 1e-5
    assert all(w >= 0 for w in weights)


def test_blend_between_burton_and_corrector(rl, fusion_payload):
    weights  = np.array([0.3, 0.7])
    dst_pred = rl.blend(fusion_payload, weights)
    expected = 0.3 * fusion_payload["dst_burton"] + 0.7 * fusion_payload["dst_corrector"]
    assert abs(dst_pred - expected) < 1e-4


def test_reward_perfect_prediction():
    assert RLAgent._reward(0.0, 0.0)   == 0.0


def test_reward_50nt_error():
    assert abs(RLAgent._reward(50.0, 0.0) + 1.0) < 1e-5


def test_reward_clipped_at_minus_one():
    assert RLAgent._reward(200.0, 0.0) == -1.0


def test_observe_stores_pending(rl, fusion_payload):
    weights = rl.act(fusion_payload)
    rl.observe(fusion_payload, -45.0)
    assert len(rl._pending) == 1


def test_give_reward_pops_pending(rl, fusion_payload):
    rl.act(fusion_payload)
    rl.observe(fusion_payload, -45.0)
    assert len(rl._pending) == 1
    rl.give_reward(-48.0)
    assert len(rl._pending) == 0


def test_give_reward_increments_step(rl, fusion_payload):
    rl.act(fusion_payload)
    rl.observe(fusion_payload, -45.0)
    rl.give_reward(-48.0)
    assert rl._reward_step == 1


def test_buffer_grows_on_reward(rl, fusion_payload):
    for _ in range(5):
        rl.act(fusion_payload)
        rl.observe(fusion_payload, -45.0)
        rl.give_reward(-48.0)
    assert len(rl.buffer) == 5


# ── RLBusAgent ───────────────────────────��───────────────────────────────────

def test_rl_bus_agent_publishes_on_fusion(mock_bus, fusion_payload):
    agent  = RLBusAgent(mock_bus)
    result = agent.process(TOPICS["fusion"], fusion_payload)
    assert result is not None
    assert "dst_pred"      in result
    assert "w_burton"      in result
    assert "w_corrector"   in result
    assert "alert_level"   in result


def test_rl_bus_agent_returns_none_on_reward(mock_bus):
    agent  = RLBusAgent(mock_bus)
    result = agent.process(TOPICS["rl_reward"], {"dst_actual": -55.0})
    assert result is None


def test_rl_bus_agent_blend_certainty_in_range(mock_bus, fusion_payload):
    agent  = RLBusAgent(mock_bus)
    result = agent.process(TOPICS["fusion"], fusion_payload)
    assert 0.0 <= result["blend_certainty"] <= 1.0


def test_rl_bus_agent_output_topic(mock_bus):
    assert RLBusAgent(mock_bus).output_topic == TOPICS["final"]
