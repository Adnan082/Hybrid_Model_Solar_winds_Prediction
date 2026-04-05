"""
agents/rl_agent.py
------------------
Online Actor-Critic RL agent that learns the optimal blend of
Burton + Corrector predictions conditioned on the full solar-wind
context and anomaly signal.

Architecture
------------
  PolicyNet: shared MLP trunk → actor head (softmax weights) + critic head (V(s))
  STATE_DIM = 9   features extracted from fused payload
  ACTION_DIM = 2  [w_burton, w_corrector] — softmax blend weights

Learning cycle (per timestep)
------------------------------
  1. fusion_agent calls  weights = rl.act(payload)
  2. fusion_agent calls  rl.observe(payload, dst_pred)   — stores transition
  3. When actual Dst arrives (rl.reward topic):
        rl.give_reward(dst_actual)  — assigns reward, pushes to replay, updates net

Reward shaping
--------------
  r = −clip(|pred − actual| / 50 nT, 0, 1)
  → perfect prediction earns 0, 50 nT error earns −1

Persistence
-----------
  Checkpoint saved every 500 reward steps to models/rl_agent.pt
  Automatically reloaded on next startup → agent matures across sessions
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────
STATE_DIM  = 9    # see _encode() for feature list
ACTION_DIM = 2    # [w_burton, w_corrector]

ALERT_ENC: dict[str, float] = {"GREEN": 0.0, "YELLOW": 0.5, "RED": 1.0}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CKPT_PATH  = MODELS_DIR / "rl_agent.pt"


# ── Neural network ─────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    Shared-trunk actor-critic.

      trunk  : 2-layer MLP with LayerNorm + ReLU
      actor  : Linear → softmax weights over blend actions
      critic : Linear → scalar state value V(s)
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (batch, STATE_DIM)
        Returns:
            weights : (batch, ACTION_DIM)  — softmax blend weights
            value   : (batch,)             — estimated V(s)
        """
        h       = self.trunk(x)
        weights = torch.softmax(self.actor(h), dim=-1)
        value   = self.critic(h).squeeze(-1)
        return weights, value


# ── Replay buffer ──────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int = 20_000):
        self._buf: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((state, action, float(reward), next_state, float(done)))

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        idx    = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        batch  = [self._buf[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.float32),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── RL agent ───────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Online Actor-Critic RL agent.

    Usage (called by FusionAgent each step)::

        weights  = rl.act(fused_payload)
        dst_pred = rl.blend(fused_payload, weights)
        rl.observe(fused_payload, dst_pred)
        # ... later, when actual Dst arrives:
        rl.give_reward(dst_actual)
    """

    def __init__(
        self,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        entropy_coef:  float = 0.01,
        update_every:  int   = 64,
        batch_size:    int   = 64,
        device:        str   = "cpu",
    ):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.update_every = update_every
        self.batch_size   = batch_size
        self.device       = device

        self.net    = PolicyNet(STATE_DIM, ACTION_DIM).to(device)
        self.optim  = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = _ReplayBuffer(capacity=20_000)

        self._reward_step  = 0
        self._total_reward = 0.0
        self._last_weights: np.ndarray | None = None

        # Queue of (state, action, dst_pred) waiting for a delayed reward
        self._pending: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=120)

        if CKPT_PATH.exists():
            self._load()

        logger.info(
            f"[RLAgent] Ready  state_dim={STATE_DIM}  action_dim={ACTION_DIM}  "
            f"device={device}  checkpoint={'loaded' if CKPT_PATH.exists() else 'none'}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def act(self, payload: dict) -> np.ndarray:
        """
        Given the fused payload, return softmax blend weights
        [w_burton, w_corrector].
        """
        state = self._encode(payload)
        with torch.no_grad():
            x              = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            weights, _     = self.net(x)
            weights_np     = weights.squeeze(0).cpu().numpy()
        self._last_weights = weights_np
        return weights_np

    def blend(self, payload: dict, weights: np.ndarray) -> float:
        """
        Compute final Dst prediction from blend weights.

            dst_final = w_burton * dst_burton + w_corrector * dst_corrector
        """
        dst_burton    = float(payload.get("dst_burton",    0.0))
        dst_corrector = float(payload.get("dst_corrector", dst_burton))
        return float(weights[0] * dst_burton + weights[1] * dst_corrector)

    def observe(self, payload: dict, dst_pred: float) -> None:
        """
        Store transition so a delayed reward can be matched to it later.
        Called immediately after act() + blend().
        """
        state = self._encode(payload)
        if self._last_weights is not None:
            self._pending.append((state, self._last_weights.copy(), dst_pred))

    def give_reward(self, dst_actual: float) -> None:
        """
        Called when an actual Dst observation arrives (from rl.reward topic).
        Matches it to the oldest pending transition, pushes to replay buffer,
        and triggers a learning update every `update_every` reward steps.
        """
        if not self._pending:
            return

        state, action, dst_pred = self._pending.popleft()
        reward                  = self._reward(dst_pred, dst_actual)
        next_state              = state          # contextual-bandit: no next-state distinction

        self.buffer.push(state, action, reward, next_state, done=False)

        self._total_reward += reward
        self._reward_step  += 1

        if (self._reward_step % self.update_every == 0
                and len(self.buffer) >= self.batch_size):
            loss = self._update()
            logger.info(
                f"[RLAgent] reward_step={self._reward_step}  "
                f"last_reward={reward:.3f}  loss={loss:.4f}  "
                f"avg_reward={self._total_reward / self._reward_step:.3f}  "
                f"weights=[{action[0]:.3f}, {action[1]:.3f}]"
            )

        if self._reward_step % 500 == 0:
            self._save()

    def weights_summary(self) -> dict:
        """Return the current blend weights for the last seen state (for logging)."""
        if self._last_weights is None:
            return {"w_burton": 0.5, "w_corrector": 0.5}
        return {
            "w_burton":    round(float(self._last_weights[0]), 4),
            "w_corrector": round(float(self._last_weights[1]), 4),
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _encode(self, payload: dict) -> np.ndarray:
        """
        Build a fixed-size (STATE_DIM=9) state vector from the fused payload.

        All features are normalized to a roughly [-1, 1] range so the
        network can learn without exploding gradients.
        """
        dst_burton    = float(payload.get("dst_burton",    0.0))
        dst_corrector = float(payload.get("dst_corrector", dst_burton))
        anomaly_score = float(payload.get("anomaly_score", 0.0))
        alert_enc     = ALERT_ENC.get(str(payload.get("alert_level", "GREEN")), 0.0)
        storm_phase   = float(payload.get("storm_phase",   0))   / 3.0    # 0–3 → 0–1
        E_field       = float(payload.get("E_field",       0.0)) / 10.0   # mV/m, typical ±10
        bz_gsm        = float(payload.get("bz_gsm",        0.0)) / 50.0   # nT, typical ±50
        speed         = float(payload.get("speed",       400.0)) / 800.0  # km/s, 300–800
        dDst_dt       = float(payload.get("dDst_dt",       0.0)) / 20.0   # nT/hr, typical ±20

        return np.array(
            [dst_burton / 500.0, dst_corrector / 500.0,
             anomaly_score, alert_enc, storm_phase,
             E_field, bz_gsm, speed, dDst_dt],
            dtype=np.float32,
        )

    @staticmethod
    def _reward(dst_pred: float, dst_actual: float) -> float:
        """
        Shaped reward:
          • Perfect prediction (0 nT error)  → 0.0
          • 50 nT error                      → −1.0
          • >50 nT error clipped to          → −1.0
        """
        mae = abs(dst_pred - dst_actual)
        return float(-np.clip(mae / 50.0, 0.0, 1.0))

    def _update(self) -> float:
        """One mini-batch actor-critic gradient step."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        weights, values      = self.net(states)
        with torch.no_grad():
            _, next_values   = self.net(next_states)

        # TD(0) targets and advantage
        td_targets = rewards + self.gamma * next_values * (1.0 - dones)
        advantages = (td_targets - values).detach()

        # Normalize advantages for stable training
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        critic_loss = nn.functional.mse_loss(values, td_targets.detach())

        # Policy gradient: maximize E[log π(a|s) * A(s,a)]
        log_probs  = torch.log(weights + 1e-8)
        actor_loss = -(log_probs * advantages.unsqueeze(-1)).sum(dim=-1).mean()

        # Entropy bonus: encourages exploration by penalizing certainty
        entropy    = -(weights * log_probs).sum(dim=-1).mean()

        loss = critic_loss + actor_loss - self.entropy_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optim.step()

        return float(loss.item())

    def _save(self) -> None:
        torch.save(
            {
                "net":          self.net.state_dict(),
                "optim":        self.optim.state_dict(),
                "reward_step":  self._reward_step,
                "total_reward": self._total_reward,
            },
            CKPT_PATH,
        )
        logger.info(
            f"[RLAgent] Checkpoint saved → {CKPT_PATH}  "
            f"reward_step={self._reward_step}"
        )

    def _load(self) -> None:
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optim.load_state_dict(ckpt["optim"])
        self._reward_step  = ckpt.get("reward_step",  0)
        self._total_reward = ckpt.get("total_reward", 0.0)
        logger.info(
            f"[RLAgent] Checkpoint loaded ← {CKPT_PATH}  "
            f"reward_step={self._reward_step}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# RLBusAgent — Agent 5, a proper standalone event-bus agent
# ═════════════════════════════════════════════════════════════════════════════

import os
import math
from agents.base_agent import BaseAgent
from event_bus.bus import EventBus, TOPICS


class RLBusAgent(BaseAgent):
    """
    Agent 5 — RL Blend Agent

    Subscribes to : fusion.output   (merged context from Agent 4)
                    rl.reward       (actual Dst observations for training)

    Publishes to  : prediction.final

    On every fusion.output message:
      1. Calls RLAgent.act()     to get blend weights [w_burton, w_corrector]
      2. Calls RLAgent.blend()   to compute dst_final
      3. Calls RLAgent.observe() to queue the transition for later reward
      4. Publishes the full forecast to prediction.final

    On every rl.reward message:
      - Calls RLAgent.give_reward() which triggers an online learning update
        and saves a checkpoint every 500 reward steps so the agent matures
        across sessions.
    """

    def __init__(self, bus: EventBus, device: str = "cpu"):
        super().__init__("RLBusAgent", bus)
        self.rl    = RLAgent(device=device)
        self._step = 0
        logger.info("[RLBusAgent] Initialised.")

    @property
    def input_topics(self) -> list[str]:
        return [TOPICS["fusion"], TOPICS["rl_reward"]]

    @property
    def output_topic(self) -> str:
        return TOPICS["final"]

    def process(self, channel: str, payload: dict) -> dict | None:
        # ── Reward signal: actual Dst arrived → train the policy ─────────────
        if channel == TOPICS["rl_reward"]:
            self.rl.give_reward(float(payload.get("dst_actual", 0.0)))
            return None

        # ── Fused context arrived → compute blended prediction ────────────────
        self._step += 1

        weights  = self.rl.act(payload)
        dst_pred = self.rl.blend(payload, weights)
        self.rl.observe(payload, dst_pred)

        w_b, w_c = float(weights[0]), float(weights[1])
        entropy         = -(w_b * math.log(w_b + 1e-8) + w_c * math.log(w_c + 1e-8))
        blend_certainty = round(1.0 - entropy / math.log(2), 4)

        if self._step % 60 == 0:
            logger.info(
                f"[RLBusAgent] step={self._step}  "
                f"dst_pred={dst_pred:.2f} nT  "
                f"w=[{w_b:.3f}, {w_c:.3f}]  "
                f"reward_steps={self.rl._reward_step}"
            )

        return {
            "period":          payload.get("period",    ""),
            "timedelta":       payload.get("timedelta", ""),
            "dst_pred":        round(dst_pred, 4),
            "dst_burton":      round(float(payload.get("dst_burton",    0.0)), 4),
            "dst_corrector":   round(float(payload.get("dst_corrector", 0.0)), 4),
            "residual_pred":   round(float(payload.get("residual_pred", 0.0)), 4),
            "anomaly_score":   round(float(payload.get("anomaly_score", 0.0)), 4),
            "alert_level":     payload.get("alert_level", "GREEN"),
            "recon_error":     round(float(payload.get("recon_error",   0.0)), 6),
            "w_burton":        round(w_b, 4),
            "w_corrector":     round(w_c, 4),
            "blend_certainty": blend_certainty,
            "confidence":      round(1.0 - float(payload.get("anomaly_score", 0.0)), 4),
            "storm_phase":     payload.get("storm_phase", 0),
            "E_field":         round(float(payload.get("E_field", 0.0)), 4),
            "bz_gsm":          payload.get("bz_gsm",  0.0),
            "speed":           payload.get("speed",  400.0),
            "dDst_dt":         round(float(payload.get("dDst_dt", 0.0)), 4),
            "rl_reward_steps": self.rl._reward_step,
        }

    def reset(self) -> None:
        self.rl._pending.clear()
        self._step = 0
        logger.info("[RLBusAgent] State reset.")


if __name__ == "__main__":
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    RLBusAgent(bus).run()
