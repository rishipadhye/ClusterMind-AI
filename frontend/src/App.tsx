import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { fetchExperimentHistory, predictAll, type JobMetadata } from "./services/api";

const defaultPayload: JobMetadata = {
  model_type: "gpt",
  framework: "pytorch",
  gpu_type: "a100",
  num_gpus: 4,
  batch_size: 64,
  dataset_size: 2500000,
  cpu_cores: 32,
  ram_gb: 128,
  estimated_flops: 6.2e14,
  distributed_training: true,
  mixed_precision: true,
};

type PredictionState = {
  runtime?: { value: number; confidence: number; model_version: string };
  vram?: { value: number; confidence: number; model_version: string };
  failure?: { probability: number; confidence: number; model_version: string; likely_reason: string };
  recommendation?: {
    recommended_gpu_count: number;
    recommended_vram_gb: number;
    expected_runtime_hours: number;
    failure_risk: number;
    optimization_suggestions: string[];
  };
};

export function App() {
  const [payload, setPayload] = useState<JobMetadata>(defaultPayload);
  const [prediction, setPrediction] = useState<PredictionState>({});
  const [history, setHistory] = useState<Array<Record<string, unknown>>>([]);
  const [loading, setLoading] = useState(false);

  const chartData = useMemo(
    () => [
      { name: "Runtime (hrs)", value: prediction.runtime?.value ?? 0 },
      { name: "VRAM (GB)", value: prediction.vram?.value ?? 0 },
      { name: "Failure Risk", value: prediction.failure?.probability ?? 0 },
    ],
    [prediction]
  );

  async function runPrediction() {
    setLoading(true);
    try {
      const result = await predictAll(payload);
      setPrediction(result);
    } finally {
      setLoading(false);
    }
  }

  async function loadExperiments() {
    const rows = await fetchExperimentHistory();
    setHistory(rows);
  }

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <header>
          <h1 className="text-4xl font-bold tracking-tight">ClusterMind AI</h1>
          <p className="mt-3 text-slate-300">GPU resource allocation prediction for ML infrastructure teams.</p>
        </header>

        <section className="grid gap-6 lg:grid-cols-2">
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Job Predictions</h2>
            <p className="mt-2 text-sm text-slate-400">Simulate a workload and infer runtime, VRAM, and failure risk.</p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <label className="col-span-1">
                <span className="block text-slate-400">Model Type</span>
                <select
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
                  value={payload.model_type}
                  onChange={(e) => setPayload({ ...payload, model_type: e.target.value as JobMetadata["model_type"] })}
                >
                  <option value="resnet">ResNet</option>
                  <option value="bert">BERT</option>
                  <option value="gpt">GPT</option>
                  <option value="vision_transformer">Vision Transformer</option>
                  <option value="stable_diffusion">Stable Diffusion</option>
                  <option value="recommendation">Recommendation</option>
                </select>
              </label>
              <label className="col-span-1">
                <span className="block text-slate-400">GPU Type</span>
                <select
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
                  value={payload.gpu_type}
                  onChange={(e) => setPayload({ ...payload, gpu_type: e.target.value as JobMetadata["gpu_type"] })}
                >
                  <option value="t4">T4</option>
                  <option value="v100">V100</option>
                  <option value="a100">A100</option>
                  <option value="rtx4090">RTX4090</option>
                </select>
              </label>
              <label>
                <span className="block text-slate-400">Num GPUs</span>
                <input
                  type="number"
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
                  value={payload.num_gpus}
                  onChange={(e) => setPayload({ ...payload, num_gpus: Number(e.target.value) })}
                />
              </label>
              <label>
                <span className="block text-slate-400">Batch Size</span>
                <input
                  type="number"
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2"
                  value={payload.batch_size}
                  onChange={(e) => setPayload({ ...payload, batch_size: Number(e.target.value) })}
                />
              </label>
            </div>
            <button
              onClick={runPrediction}
              className="mt-4 rounded bg-cyan-500 px-4 py-2 font-medium text-slate-950 hover:bg-cyan-400"
            >
              {loading ? "Predicting..." : "Run Predictions"}
            </button>
          </article>

          <article className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Cluster Overview</h2>
            <p className="mt-2 text-sm text-slate-400">Prediction output and utilization trend cards.</p>
            <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
              <div className="rounded border border-slate-800 bg-slate-950 p-3">
                Runtime: <strong>{prediction.runtime?.value?.toFixed(2) ?? "--"}</strong>
              </div>
              <div className="rounded border border-slate-800 bg-slate-950 p-3">
                Peak VRAM: <strong>{prediction.vram?.value?.toFixed(2) ?? "--"}</strong>
              </div>
              <div className="rounded border border-slate-800 bg-slate-950 p-3">
                Failure: <strong>{prediction.failure ? `${(prediction.failure.probability * 100).toFixed(1)}%` : "--"}</strong>
              </div>
            </div>
            <div className="mt-4 h-64 rounded border border-slate-800 bg-slate-950 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip />
                  <Bar dataKey="value" fill="#22d3ee" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Resource Recommendations</h2>
            <p className="mt-2 text-sm text-slate-400">Optimization suggestions from prediction service output.</p>
            <ul className="mt-4 space-y-2 text-sm">
              {(prediction.recommendation?.optimization_suggestions ?? ["Run prediction to see suggestions."]).map(
                (item) => (
                  <li key={item} className="rounded border border-slate-800 bg-slate-950 p-2">
                    {item}
                  </li>
                )
              )}
            </ul>
          </article>

          <article className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Experiment Tracking</h2>
            <p className="mt-2 text-sm text-slate-400">Recent model training runs and metrics.</p>
            <button
              onClick={loadExperiments}
              className="mt-4 rounded border border-slate-700 px-4 py-2 hover:border-slate-500"
            >
              Load Experiment History
            </button>
            <div className="mt-4 max-h-56 space-y-2 overflow-auto text-xs">
              {history.length === 0 ? (
                <p className="text-slate-400">No experiment history loaded.</p>
              ) : (
                history.map((row, idx) => (
                  <pre key={`${idx}-${String(row.run_id ?? "row")}`} className="rounded border border-slate-800 bg-slate-950 p-2">
                    {JSON.stringify(row, null, 2)}
                  </pre>
                ))
              )}
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}

