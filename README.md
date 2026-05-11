# ClusterMind AI

ClusterMind AI is a production-style ML infrastructure project that predicts GPU memory usage, job runtime, failure probability, and resource recommendations for ML training jobs before execution.

## Goals

- Simulate real ML platform engineering workflows for GPU cluster optimization.
- Build maintainable services with FastAPI, PostgreSQL, and modular ML pipelines.
- Ship an internship-ready project with architecture, telemetry, and evaluation discipline.

## Implementation Status

Implemented across phases 1-8 (MVP):

- Modular backend architecture (`api`, `core`, `services`, `models`, `ml`, `telemetry`, `database`)
- FastAPI endpoints for predictions, recommendations, model status, and experiment history
- Typed Pydantic request/response schemas with validation
- Config and environment management via `pydantic-settings`
- Service-layer separation for predictions and recommendations with model artifact loading
- Data pipeline with train/validation/test split, normalization, encoding, and feature engineering
- Training pipeline for runtime, VRAM, and failure models (XGBoost/LightGBM/RandomForest fallback)
- Lightweight experiment tracker (`JSONL`) and metrics persistence
- Prometheus-compatible `/metrics` endpoint and request latency instrumentation
- Frontend React + Tailwind + Recharts dashboard with live API calls
- Dockerfiles + `docker-compose` for backend, frontend, and PostgreSQL
- Test setup (`pytest`) for health and prediction endpoints

## Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── ml/
│   │   ├── models/
│   │   ├── services/
│   │   ├── telemetry/
│   │   └── main.py
│   ├── data/
│   ├── notebooks/
│   ├── tests/
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
└── docker-compose.yml
```

## Quick Start

1. Copy env file:

```bash
cp backend/.env.example backend/.env
```

2. Start local services:

```bash
docker compose up --build
```

3. Verify backend:

- API docs: `http://localhost:8000/docs`
- Health: `GET http://localhost:8000/api/v1/health`

4. Open frontend:

- `http://localhost:5173`

## ML Workflow Commands

### Phase 2: Generate Synthetic Workloads

Run from `backend/`:

```bash
python -m app.ml.generate_dataset --num-samples 50000 --output-path data/synthetic_jobs.csv
```

The generated dataset includes:

- workload metadata (model, framework, GPU, batching, compute inputs)
- target variables (`runtime_hours`, `peak_vram_gb`, `failed`)
- failure diagnostics (`failure_reason`)

### Phase 3-4: Train Models + Evaluate

```bash
python -m app.ml.training --dataset-path data/synthetic_jobs.csv --artifacts-dir data/models --experiments-path data/experiments/runs.jsonl
python -m app.ml.evaluate --artifact-path data/models/model_bundle.joblib
```

Model bundle includes:

- preprocessing pipeline (encoding + scaling)
- runtime regression model
- VRAM regression model
- failure classification model
- metrics (`MAE`, `RMSE`, `F1`, `ROC-AUC`)

### Phase 5-6: Prediction & Recommendation APIs

- `POST /api/v1/predict/runtime`
- `POST /api/v1/predict/vram`
- `POST /api/v1/predict/failure`
- `POST /api/v1/recommend/resources`

### Phase 7: Dashboard

Frontend includes:

- Cluster Overview cards + trend chart
- Job prediction form
- Failure analytics card
- Resource recommendation panel
- Experiment history viewer

### Phase 8: Experiment Tracking

- Run metadata and metrics logged to `backend/data/experiments/runs.jsonl`
- Model metrics snapshots saved to `backend/data/models/metrics.json`

## Telemetry

- Prometheus scrape endpoint: `GET /metrics`
- Request counters and latency histograms via middleware

## Remaining Advanced Phase (9)

Future extensions:

- RL-based scheduler policy
- anomaly detection pipeline
- real telemetry ingestion (Prometheus exporters)
- Kubernetes cluster integration



## Phase 9 (Advanced)

Implemented stubs and production-style hooks for:

- Anomaly detection (IsolationForest) training + `/api/v1/anomalies/score`
- Scheduler simulation (bandit-style policy) via `/api/v1/scheduler/simulate`
- Prometheus + Grafana services in `docker-compose.yml`
- Kubernetes manifests in `k8s/`

### Phase 9 commands

```bash
cd backend
python -m app.ml.anomaly --dataset-path data/synthetic_jobs.csv --artifact-path data/models/anomaly_bundle.joblib
```

