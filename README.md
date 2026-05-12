# ClusterMind AI

ClusterMind AI predicts GPU resource usage for ML training jobs before they run. Given a job description (model type, framework, GPU, batch size, etc.), it returns predicted runtime, peak VRAM usage, and failure probability — plus a resource recommendation. It is built as a production-style ML platform service with a React dashboard, REST API, and Prometheus telemetry.

## What It Does

- **Runtime prediction** — estimates how long a training job will take
- **VRAM prediction** — estimates peak GPU memory usage
- **Failure prediction** — estimates probability the job will fail
- **Resource recommendation** — suggests optimal GPU configuration for the job
- **Experiment tracking** — logs model training runs and metrics to JSONL
- **Anomaly detection** — scores jobs against historical baselines
- **Scheduler simulation** — bandit-style policy for job queue ordering

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Models | XGBoost, LightGBM, scikit-learn (RandomForest fallback) |
| Database | PostgreSQL 16 |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| Observability | Prometheus, Grafana |
| Containers | Docker + Docker Compose |

## Repository Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── core/         # Settings, logging
│   │   ├── database/     # SQLAlchemy session
│   │   ├── ml/           # Training, evaluation, data generation
│   │   ├── models/       # Pydantic schemas
│   │   ├── services/     # Prediction and recommendation logic
│   │   ├── telemetry/    # Prometheus metrics middleware
│   │   └── main.py
│   ├── data/
│   │   ├── models/       # Trained model artifacts (.joblib)
│   │   ├── experiments/  # Experiment run logs (.jsonl)
│   │   └── synthetic_jobs.csv
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── Dockerfile
├── infra/prometheus/
├── k8s/
└── docker-compose.yml
```

## Running Locally

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (only needed if running without Docker)
- Node.js 20+ (only needed if running without Docker)

### Option 1 — Docker Compose (recommended)

This starts the backend, frontend, PostgreSQL, Prometheus, and Grafana together.

**1. Copy the environment file:**

```bash
cp backend/.env.example backend/.env
```

**2. Build and start all services:**

```bash
docker compose up --build
```

**3. Access the services:**

| Service | URL |
|---|---|
| Frontend dashboard | http://localhost:5173 |
| Backend API docs | http://localhost:8000/docs |
| Backend health check | http://localhost:8000/api/v1/health |
| Prometheus metrics | http://localhost:9090 |
| Grafana | http://localhost:3000 |

> **Note:** The backend requires trained model artifacts (`data/models/model_bundle.joblib`) to serve predictions. If the `data/models/` directory is empty, run the ML pipeline steps below before starting the server.

---

### Option 2 — Manual (no Docker)

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set POSTGRES_HOST=localhost
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at http://localhost:5173 and proxies API calls to http://localhost:8000.

---

## ML Pipeline

The backend ships with a synthetic data generator and training pipeline. Run these from the `backend/` directory with the virtualenv active (or inside the Docker container).

### 1. Generate synthetic training data

```bash
python -m app.ml.generate_dataset --num-samples 50000 --output-path data/synthetic_jobs.csv
```

### 2. Train models

```bash
python -m app.ml.training \
  --dataset-path data/synthetic_jobs.csv \
  --artifacts-dir data/models \
  --experiments-path data/experiments/runs.jsonl
```

This produces `data/models/model_bundle.joblib` containing:
- preprocessing pipeline (encoding + scaling)
- runtime regression model (XGBoost/LightGBM)
- VRAM regression model
- failure classification model
- saved metrics (MAE, RMSE, F1, ROC-AUC)

### 3. Evaluate models

```bash
python -m app.ml.evaluate --artifact-path data/models/model_bundle.joblib
```

### 4. Train anomaly detection model (optional)

```bash
python -m app.ml.anomaly \
  --dataset-path data/synthetic_jobs.csv \
  --artifact-path data/models/anomaly_bundle.joblib
```

---

## API Endpoints

All endpoints are under the `/api/v1` prefix. Interactive docs at http://localhost:8000/docs.

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict/runtime` | Predict job runtime |
| POST | `/api/v1/predict/vram` | Predict peak VRAM |
| POST | `/api/v1/predict/failure` | Predict failure probability |
| POST | `/api/v1/recommend/resources` | Get GPU resource recommendation |
| GET | `/api/v1/experiments` | List experiment run history |
| GET | `/api/v1/models/status` | Model artifact status |
| POST | `/api/v1/anomalies/score` | Score a job for anomalies |
| POST | `/api/v1/scheduler/simulate` | Simulate scheduler policy |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## Running Tests

```bash
cd backend
pytest
```

Tests cover health endpoints and prediction endpoint contracts.
