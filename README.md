# Solar Wind Dst Prediction — Hybrid ML Pipeline

A real-time, multi-agent system that predicts the **Dst (Disturbance Storm Time) index** from solar wind measurements. Combines a physics-based Burton ODE solver, a Transformer anomaly detector, a BiLSTM residual corrector, and an Actor-Critic RL agent that learns optimal blending weights — all connected via a Redis pub/sub event bus and deployable against live NOAA satellite feeds.

---

## Architecture

```
NOAA SWPC Live Feed  ──or──  Historical CSV Replay
              |
              v
     [LiveFeeder / DataFeeder]
              |
              v  solar_wind.raw (Redis)
     ─────────────────────────────────────
     |                                   |
     v                                   v
[Agent 1: BurtonAgent]        (passes through to Agents 2 & 3)
 Burton ODE physics
 dst_burton → burton.output
     |
     +──────────────────────┐
     |                      |
     v                      v
[Agent 2: AnomalyAgent]  [Agent 3: CorrectorAgent]
 Transformer Autoencoder   BiLSTM Residual Corrector
 239K params, seq=60       546K params, seq=120
 anomaly_score 0–1         dst_corrector = dst_burton + residual
 anomaly.output            ml.output
     |                      |
     └──────────┬───────────┘
                v
        [Agent 4: FusionAgent]
         Merges all 3 outputs
         fusion.output
                |
                v
        [Agent 5: RLBusAgent]
         Actor-Critic PolicyNet (5K params)
         Learns w_burton, w_corrector
         dst_final = w_b*dst_burton + w_c*dst_corrector
         prediction.final
                |
         ───────────────
         |             |
         v             v
     FastAPI       Dashboard
     REST + WS     Dash / Plotly
```

---

## Models

| Model | Params | Architecture | Input | Output |
|---|---|---|---|---|
| TransformerAutoencoder | 239K | Transformer encoder-decoder | 8 features, seq=60 (1hr) | Reconstruction error → anomaly score 0–1 |
| LSTMCorrector | 546K | BiLSTM 2-layer hidden=128 | 17 features, seq=120 (2hr) | Scalar residual correction |
| PolicyNet (RL) | 5K | MLP 9→64→64 + actor + critic | 9-dim state vector | [w_burton, w_corrector] softmax |
| **Total** | **790K** | | | |

### Inference latency (model only, CPU)
- p50: 11.7 ms — p95: 18.7 ms — p99: 46.4 ms

---

## Performance

Trained and validated on **8.4M+ observations** from NASA OMNI 1-minute dataset.

### Per-class RMSE (BiLSTM Corrector vs Burton ODE baseline)

| Storm Class | Dst Range | Corrector RMSE | Burton RMSE | Improvement |
|---|---|---|---|---|
| Quiet | > −30 nT | 4.91 nT | — | — |
| Minor | −30 to −50 nT | 7.74 nT | — | — |
| Moderate | −50 to −100 nT | 11.84 nT | — | — |
| Intense | −100 to −200 nT | 13.71 nT | ~42 nT | **3x lower** |
| Extreme | < −200 nT | 6.50 nT | ~42 nT | **6.5x lower** |
| **Overall** | | **6.50 nT** | ~14 nT | **2x lower** |

### Anomaly Detection

| Storm Class | Reconstruction Error | Ratio vs Quiet |
|---|---|---|
| Quiet | ~0.002 | 1× |
| Moderate | ~0.006 | 2.9× |
| Intense | ~0.031 | 15× |
| Extreme | ~0.47 | **237×** |

> Per-class RMSE breakdown is not reported by any published Dst prediction paper — this is a methodological contribution of this project.

---

## Storm Classification

| Class | Dst Range | NOAA G-Scale |
|---|---|---|
| QUIET | > −30 nT | — |
| MINOR | −30 to −50 nT | G1 |
| MODERATE | −50 to −100 nT | G2 |
| INTENSE | −100 to −200 nT | G3 |
| EXTREME | < −200 nT | G4+ |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for Redis)

### Install
```bash
pip install -r requirements.txt
```

### Run — Live NOAA Feed (4 terminals)

**Terminal 1 — Redis**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

**Terminal 2 — API server**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Pipeline with live NOAA feed**
```bash
python main.py --live
```

**Terminal 4 — Dashboard**
```bash
python dashboard/app.py
```

Open **http://localhost:8050**

On startup the live feeder replays the last 150 minutes of NOAA history to warm up agent windows, then switches to 60-second polling. Full predictions appear within ~15 seconds.

### Run — Historical CSV Replay

```bash
# Full speed (backtesting / RL training)
python main.py

# Real-time speed simulation
python main.py --speed 1.0

# Specific period only
python main.py --period train_a
```

### Validate against all storm periods

```bash
# Fast sample (10K rows, keeps all extreme/intense events)
python validate_storms.py --n-rows 10000

# Full validation (all 2.8M rows, overnight run)
python validate_storms.py --n-rows 999999999 --no-plots
```

### Offline batch evaluation with MLflow

```bash
python run_pipeline.py --period train_a
mlflow ui --backend-store-uri mlruns/
```

Open **http://localhost:5000**

---

## Live NOAA Data Feeds

The system connects to NOAA SWPC free public feeds — no API key required:

| Feed | URL | Cadence |
|---|---|---|
| Magnetic field (DSCOVR/ACE) | `services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json` | 1-minute |
| Solar wind plasma | `services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | 1-minute |
| Kyoto Dst index | `services.swpc.noaa.gov/products/kyoto-dst.json` | Hourly |

Data latency from satellite to pipeline: ~5 minutes (L1 propagation + NOAA QC).

The Kyoto Dst feed is published to the `rl.reward` Redis topic which the RL agent uses as its delayed training signal — meaning the agent **learns from real geomagnetic data in production**.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Redis + pipeline status |
| GET | `/api/v1/prediction/latest` | Latest blended Dst prediction |
| GET | `/api/v1/prediction/storm-class` | Current storm classification |
| GET | `/api/v1/prediction/fusion` | Raw pre-RL fusion context |
| POST | `/api/v1/prediction/reward` | Submit actual Dst for RL training |
| GET | `/api/v1/history/predictions?n=120` | Last N predictions |
| GET | `/api/v1/agents/status` | All 5 agent heartbeat status |
| GET | `/metrics` | Prometheus metrics |
| WS | `/ws/live` | WebSocket real-time stream |

---

## Dashboard Features

- Real-time Dst prediction vs Burton physics vs BiLSTM corrector
- Anomaly score gauge with GREEN / YELLOW / RED alert levels
- RL blend weight history (stacked area — shows how agent learns to prefer corrector during storms)
- Solar wind parameters: Bz GSM + speed dual-axis
- RL learning curve (blend certainty over time)
- Agent pipeline status panel (live / stalled / offline)
- KPI cards: current Dst, anomaly score, confidence, RL reward steps

---

## Project Structure

```
Solar_Winds/
├── agents/
│   ├── base_agent.py          # Abstract BaseAgent (subscribe/publish loop)
│   ├── burton_agent.py        # Agent 1: Burton ODE physics
│   ├── anomaly_agent.py       # Agent 2: Transformer anomaly detection
│   ├── corrector_agent.py     # Agent 3: BiLSTM residual correction
│   ├── fusion_agent.py        # Agent 4: Context merger
│   └── rl_agent.py            # Agent 5: Actor-Critic blend + RLBusAgent
├── api/
│   ├── main.py                # FastAPI app, lifespan, WebSocket
│   ├── dependencies.py        # Redis DI, classify_storm, agent status
│   ├── schemas.py             # Pydantic response models
│   ├── ws_manager.py          # WebSocket broadcast loop
│   └── routers/
│       ├── predictions.py     # /prediction/latest, /storm-class, /reward
│       ├── history.py         # /history/predictions, /history/fusion
│       └── agents.py          # /agents/status
├── dashboard/
│   ├── app.py                 # Dash app, callbacks, layout
│   ├── components/charts.py   # Plotly figure builders
│   └── assets/theme.css       # NASA dark-space CSS theme
├── event_bus/
│   └── bus.py                 # Redis pub/sub topics + set_latest helpers
├── models/
│   ├── anomaly_autoencoder.py # TransformerAutoencoder class
│   ├── lstm_corrector.py      # LSTMCorrector (BiLSTM) class
│   ├── anomaly_config.json    # Anomaly model hyperparams + thresholds
│   └── corrector_config.json  # Corrector model hyperparams + val RMSE
├── training/
│   ├── prepare_data.py        # Raw OMNI CSV → enriched.parquet
│   └── mlflow_logger.py       # MLflow per-class RMSE/MAE tracking
├── tests/
│   ├── test_agents/           # Burton, Fusion, RL agent unit tests
│   ├── test_models/           # Autoencoder + corrector shape tests
│   └── test_api/              # FastAPI endpoint tests
├── notebooks/
│   └── Plots_Pipeline/        # Validation plots (PNG outputs)
├── main.py                    # Entry point: starts all 5 agents
├── live_feeder.py             # NOAA SWPC live feed poller
├── run_pipeline.py            # Offline batch evaluation with MLflow
├── validate_storms.py         # Cross-period validation + NOAA benchmark comparison
└── requirements.txt
```

---

## Event Bus Topics

| Topic | Publisher | Subscribers |
|---|---|---|
| `solar_wind.raw` | LiveFeeder / DataFeeder | BurtonAgent |
| `burton.output` | BurtonAgent | AnomalyAgent, CorrectorAgent, FusionAgent |
| `anomaly.output` | AnomalyAgent | FusionAgent |
| `ml.output` | CorrectorAgent | FusionAgent |
| `fusion.output` | FusionAgent | RLBusAgent |
| `prediction.final` | RLBusAgent | API / Dashboard |
| `rl.reward` | LiveFeeder (Kyoto Dst) / API | RLBusAgent |

---

## MLflow Tracking

Logged per pipeline run:
- RMSE and MAE by storm class (quiet / minor / moderate / intense / extreme)
- Overall RMSE: Burton vs Corrector vs RL-blended
- RL blend weights mean and certainty
- RL reward steps (agent maturity indicator)
- Model configs as parameters (hidden size, layers, dropout, seq_len)

---

## References

- Burton, R. K., McPherron, R. L., & Russell, C. T. (1975). An empirical relationship between interplanetary conditions and Dst. *Journal of Geophysical Research*, 80(31), 4204–4214.
- Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
- NOAA SWPC Real-Time Solar Wind: https://www.swpc.noaa.gov/products/real-time-solar-wind
- NASA OMNI Dataset: https://omniweb.gsfc.nasa.gov/
