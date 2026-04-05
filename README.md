# Solar Wind Dst Prediction — Hybrid ML Pipeline

A real-time, multi-agent system that predicts the **Dst (Disturbance Storm Time) index** from solar wind measurements. It combines a physics-based Burton ODE solver, a Transformer anomaly detector, a BiLSTM residual corrector, and an Actor-Critic RL agent that learns to blend all three predictions optimally over time.

---

## Architecture Overview

```
Solar Wind Data (CSV / live feed)
         |
         v
  [Event Bus — Redis Pub/Sub]
         |
    _____|_____________________________________
   |           |              |               |
   v           v              v               |
[Agent 1]  [Agent 2]      [Agent 3]           |
 Burton     Anomaly        BiLSTM             |
 ODE        Autoencoder    Corrector          |
 Solver     (Transformer)  (LSTM)             |
   |           |              |               |
   |___________|______________|               |
                    |                         |
                    v                         |
              [Agent 4]                       |
              Fusion Agent                    |
              (merges outputs)                |
                    |                         |
                    v                         |
              [Agent 5]  <--------------------+
              RL Agent (Actor-Critic)
              (learns optimal blend weights)
                    |
                    v
           Final Dst Prediction
           -> Redis -> FastAPI -> Dashboard
```

---

## Key Components

| Component | Description |
|-----------|-------------|
| `agents/burton_agent.py` | Physics ODE solver (Burton et al. 1975), Euler integration, tau=7.7hr |
| `agents/anomaly_agent.py` | Transformer autoencoder, flags corrupted/extreme solar wind data |
| `agents/corrector_agent.py` | BiLSTM predicts residual (Dst_actual - Dst_Burton), seq_len=120 |
| `agents/fusion_agent.py` | Merges outputs from all 3 agents into a single payload |
| `agents/rl_agent.py` | Actor-Critic PolicyNet learns blend weights [w_burton, w_corrector] |
| `event_bus/bus.py` | Redis pub/sub topics connecting all agents |
| `api/main.py` | FastAPI backend, WebSocket live feed, Prometheus metrics |
| `dashboard/app.py` | Dash dashboard with NASA dark-space theme, real-time charts |
| `training/mlflow_logger.py` | MLflow experiment tracking (RMSE/MAE per storm class) |
| `run_pipeline.py` | Offline batch evaluation script with metrics and plots |

---

## Models

### TransformerAutoencoder (`models/anomaly_autoencoder.py`)
- Input: 8 solar wind features, seq_len=60 (1-hour window)
- Architecture: Transformer encoder → latent (32-dim) → Transformer decoder
- Output: Reconstructed sequence; anomaly score = normalized reconstruction error
- Threshold: 0.151 (tuned on quiet-time baseline)

### LSTMCorrector (`models/lstm_corrector.py`)
- Input: 17 features (solar wind + Burton output), seq_len=120 (2-hour window)
- Architecture: Bidirectional LSTM (2 layers, hidden=128) → Linear
- Output: Scalar residual correction
- Final prediction: `dst_final = dst_burton + residual_pred`

### RL PolicyNet (`agents/rl_agent.py`)
- State: 9-dim vector [dst_burton, dst_corrector, anomaly_score, alert_level, storm_phase, E_field, Bz, speed, dDst/dt]
- Actor: softmax → [w_burton, w_corrector] blend weights
- Critic: V(s) for TD(0) advantage estimation
- Reward: `-clip(|pred - actual| / 50, 0, 1)`
- Online learning with 20K experience replay buffer, updates every 64 steps

---

## Storm Classification

| Class | Dst Range |
|-------|-----------|
| QUIET | > -30 nT |
| MINOR | -30 to -50 nT |
| MODERATE | -50 to -100 nT |
| INTENSE | -100 to -200 nT |
| EXTREME | < -200 nT |

---

## Project Structure

```
Solar_Winds/
├── agents/
│   ├── base_agent.py          # Abstract BaseAgent (subscribe/publish loop)
│   ├── burton_agent.py        # Agent 1: Burton ODE physics
│   ├── anomaly_agent.py       # Agent 2: Transformer anomaly detection
│   ├── corrector_agent.py     # Agent 3: BiLSTM residual correction
│   ├── fusion_agent.py        # Agent 4: Output merger
│   └── rl_agent.py            # Agent 5: RL blend + RLBusAgent
├── api/
│   ├── main.py                # FastAPI app, lifespan, WebSocket
│   ├── dependencies.py        # Redis DI, classify_storm, agent status
│   ├── schemas.py             # Pydantic v2 response models
│   ├── ws_manager.py          # WebSocket broadcast loop
│   └── routers/
│       ├── predictions.py     # GET /prediction/latest, /storm-class
│       ├── history.py         # GET /history/predictions
│       └── agents.py          # GET /agents/status
├── dashboard/
│   ├── app.py                 # Dash app, callbacks, layout
│   ├── components/charts.py   # Plotly chart builders
│   └── assets/theme.css       # NASA dark-space CSS theme
├── event_bus/
│   └── bus.py                 # Redis pub/sub topics + helpers
├── models/
│   ├── anomaly_autoencoder.py # TransformerAutoencoder class
│   ├── lstm_corrector.py      # LSTMCorrector class
│   ├── anomaly_config.json    # Anomaly model hyperparams
│   └── corrector_config.json  # Corrector model hyperparams
├── training/
│   ├── prepare_data.py        # Data pipeline: raw OMNI → enriched.parquet
│   └── mlflow_logger.py       # MLflow helpers for experiment tracking
├── tests/
│   ├── conftest.py
│   ├── test_agents/           # Burton, Fusion, RL agent unit tests
│   ├── test_models/           # Autoencoder + corrector shape tests
│   └── test_api/              # FastAPI endpoint + storm classification tests
├── architecture/
│   └── architecture.png       # System architecture diagram
├── notebooks/
│   └── Plots_Pipeline/        # run_pipeline.py output plots
├── main.py                    # Entry point: starts all 5 agents + feeder
├── run_pipeline.py            # Offline batch evaluation
├── requirements.txt
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Redis (Docker or Memurai on Windows)
- PyTorch (CPU build is fine)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start Redis
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### Run the full pipeline (4 terminals)

**Terminal 1 — Pipeline agents**
```bash
python main.py --speed 10
```

**Terminal 2 — FastAPI backend**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Dashboard**
```bash
python dashboard/app.py
```

Open **http://localhost:8050** in your browser.

### Offline batch evaluation
```bash
python run_pipeline.py --period 2003
```

### Run tests
```bash
pytest tests/ -v
```

### MLflow UI
```bash
mlflow ui --backend-store-uri mlruns/
```
Open **http://localhost:5000**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Redis + data availability check |
| GET | `/api/v1/prediction/latest` | Latest Dst prediction + metadata |
| GET | `/api/v1/prediction/storm-class` | Current storm classification |
| GET | `/api/v1/history/predictions?n=120` | Last N predictions |
| GET | `/api/v1/agents/status` | All 5 agent heartbeat status |
| GET | `/metrics` | Prometheus metrics |
| WS | `/ws/live` | WebSocket live prediction stream |

---

## Dashboard Features

- Real-time Dst prediction vs Burton vs Corrector (line chart)
- Anomaly score gauge
- RL blend weight history (stacked area)
- Solar wind inputs (Bz + speed dual-axis)
- RL blend certainty learning curve
- Agent status panel (live/idle/offline indicators)
- KPI cards: current Dst, storm class, anomaly score, blend certainty

---

## MLflow Tracking

Metrics logged per run:
- RMSE and MAE broken down by storm class (quiet / minor / moderate / intense / extreme)
- Overall RMSE for Burton, Corrector, and Final (RL-blended)
- RL blend weights mean and certainty
- RL reward steps (agent maturity)

```bash
mlflow ui --backend-store-uri mlruns/
```

---

## Data

Uses **NASA OMNI** 1-minute resolution solar wind data.  
Key features: `Bz_GSM`, `speed`, `pressure`, `E_field`, `beta`, `Alfven_mach`, `Kp`, `Dst`

Preprocessing: `training/prepare_data.py` → outputs `data/enriched.parquet`

---

## References

- Burton, R. K., McPherron, R. L., & Russell, C. T. (1975). An empirical relationship between interplanetary conditions and Dst. *Journal of Geophysical Research*, 80(31), 4204–4214.
- Vaswani et al. (2017). Attention Is All You Need.
- Sutton & Barto (2018). Reinforcement Learning: An Introduction.
