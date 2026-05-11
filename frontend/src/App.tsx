import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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

const BAR_COLORS = ["#22d3ee", "#a78bfa", "#f43f5e"];

function StatCard({
  label,
  value,
  unit,
  colorClass,
  confidence,
  animate,
}: {
  label: string;
  value: string;
  unit?: string;
  colorClass: string;
  confidence?: number;
  animate: boolean;
}) {
  const barColorClass = colorClass
    .replace("text-cyan-400", "bg-cyan-400")
    .replace("text-violet-400", "bg-violet-400")
    .replace("text-emerald-400", "bg-emerald-400")
    .replace("text-amber-400", "bg-amber-400")
    .replace("text-rose-400", "bg-rose-400");

  return (
    <div
      className={`rounded-lg border bg-slate-950 p-4 transition-all duration-300 ${
        animate ? "animate-fade-in-up border-slate-700" : "border-slate-800 opacity-60"
      }`}
    >
      <p className={`text-xs font-medium uppercase tracking-wider ${colorClass}`}>{label}</p>
      <p className="mt-1 text-2xl font-bold text-slate-100">
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </p>
      {confidence !== undefined && (
        <div className="mt-2">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Confidence</span>
            <span>{(confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="mt-1 h-1 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full rounded-full transition-all duration-700 ${barColorClass}`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2.5" onClick={onChange}>
      <div
        className={`relative h-5 w-9 rounded-full transition-colors duration-200 ${
          checked ? "bg-cyan-500" : "bg-slate-700"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200 ${
            checked ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </div>
      <span className="text-xs text-slate-400">{label}</span>
    </label>
  );
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 shadow-xl">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-sm font-semibold text-cyan-400">{payload[0].value.toFixed(3)}</p>
    </div>
  );
}

function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

export function App() {
  const [payload, setPayload] = useState<JobMetadata>(defaultPayload);
  const [prediction, setPrediction] = useState<PredictionState>({});
  const [history, setHistory] = useState<Array<Record<string, unknown>>>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasPrediction = Object.keys(prediction).length > 0;

  const chartData = useMemo(
    () => [
      { name: "Runtime (hrs)", value: prediction.runtime?.value ?? 0 },
      { name: "VRAM (GB)", value: prediction.vram?.value ?? 0 },
      { name: "Fail Risk (%)", value: (prediction.failure?.probability ?? 0) * 100 },
    ],
    [prediction]
  );

  async function runPrediction() {
    setLoading(true);
    setError(null);
    try {
      const result = await predictAll(payload);
      setPrediction(result);
    } catch {
      setError("Prediction failed — is the API running on port 8000?");
    } finally {
      setLoading(false);
    }
  }

  async function loadExperiments() {
    setHistoryLoading(true);
    try {
      const rows = await fetchExperimentHistory();
      setHistory(rows);
    } finally {
      setHistoryLoading(false);
    }
  }

  const failureRisk = prediction.failure?.probability ?? 0;
  const failureColorClass =
    failureRisk < 0.3 ? "text-emerald-400" : failureRisk < 0.6 ? "text-amber-400" : "text-rose-400";

  const inputCls =
    "mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 transition focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/50";
  const labelCls = "block text-xs font-medium uppercase tracking-wide text-slate-500";

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      {/* Sticky header */}
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/80 px-8 py-4 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div>
            <h1 className="bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-3xl font-extrabold tracking-tight text-transparent">
              ClusterMind AI
            </h1>
            <p className="mt-0.5 text-xs text-slate-500">GPU Resource Allocation Predictor</p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-400">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
            </span>
            ML Infrastructure
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl space-y-6 p-8">
        {error && (
          <div className="animate-fade-in rounded-lg border border-rose-800 bg-rose-950/50 px-4 py-3 text-sm text-rose-300">
            {error}
          </div>
        )}

        <section className="grid gap-6 lg:grid-cols-2">
          {/* Job Configuration */}
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-6 transition-colors hover:border-slate-700">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-cyan-500" />
              <h2 className="text-base font-semibold">Job Configuration</h2>
            </div>
            <p className="mt-1 text-sm text-slate-400">Define workload parameters for prediction.</p>

            <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
              <label>
                <span className={labelCls}>Model Type</span>
                <select
                  className={inputCls}
                  value={payload.model_type}
                  onChange={(e) =>
                    setPayload({ ...payload, model_type: e.target.value as JobMetadata["model_type"] })
                  }
                >
                  <option value="resnet">ResNet</option>
                  <option value="bert">BERT</option>
                  <option value="gpt">GPT</option>
                  <option value="vision_transformer">Vision Transformer</option>
                  <option value="stable_diffusion">Stable Diffusion</option>
                  <option value="recommendation">Recommendation</option>
                </select>
              </label>

              <label>
                <span className={labelCls}>GPU Type</span>
                <select
                  className={inputCls}
                  value={payload.gpu_type}
                  onChange={(e) =>
                    setPayload({ ...payload, gpu_type: e.target.value as JobMetadata["gpu_type"] })
                  }
                >
                  <option value="t4">NVIDIA T4</option>
                  <option value="v100">NVIDIA V100</option>
                  <option value="a100">NVIDIA A100</option>
                  <option value="rtx4090">RTX 4090</option>
                </select>
              </label>

              <label>
                <span className={labelCls}>Framework</span>
                <select
                  className={inputCls}
                  value={payload.framework}
                  onChange={(e) =>
                    setPayload({ ...payload, framework: e.target.value as JobMetadata["framework"] })
                  }
                >
                  <option value="pytorch">PyTorch</option>
                  <option value="tensorflow">TensorFlow</option>
                  <option value="jax">JAX</option>
                </select>
              </label>

              <label>
                <span className={labelCls}>Num GPUs</span>
                <input
                  type="number"
                  min={1}
                  className={inputCls}
                  value={payload.num_gpus}
                  onChange={(e) => setPayload({ ...payload, num_gpus: Number(e.target.value) })}
                />
              </label>

              <label>
                <span className={labelCls}>Batch Size</span>
                <input
                  type="number"
                  className={inputCls}
                  value={payload.batch_size}
                  onChange={(e) => setPayload({ ...payload, batch_size: Number(e.target.value) })}
                />
              </label>

              <label>
                <span className={labelCls}>RAM (GB)</span>
                <input
                  type="number"
                  className={inputCls}
                  value={payload.ram_gb}
                  onChange={(e) => setPayload({ ...payload, ram_gb: Number(e.target.value) })}
                />
              </label>

              <label>
                <span className={labelCls}>CPU Cores</span>
                <input
                  type="number"
                  className={inputCls}
                  value={payload.cpu_cores}
                  onChange={(e) => setPayload({ ...payload, cpu_cores: Number(e.target.value) })}
                />
              </label>

              <label>
                <span className={labelCls}>Dataset Size</span>
                <input
                  type="number"
                  className={inputCls}
                  value={payload.dataset_size}
                  onChange={(e) => setPayload({ ...payload, dataset_size: Number(e.target.value) })}
                />
              </label>

              <div className="col-span-2 flex gap-6 pt-1">
                <Toggle
                  checked={payload.distributed_training}
                  onChange={() =>
                    setPayload({ ...payload, distributed_training: !payload.distributed_training })
                  }
                  label="Distributed Training"
                />
                <Toggle
                  checked={payload.mixed_precision}
                  onChange={() =>
                    setPayload({ ...payload, mixed_precision: !payload.mixed_precision })
                  }
                  label="Mixed Precision"
                />
              </div>
            </div>

            <button
              onClick={runPrediction}
              disabled={loading}
              className="mt-5 flex items-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition-all hover:from-cyan-400 hover:to-cyan-300 hover:shadow-cyan-400/30 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Spinner />
                  Predicting...
                </>
              ) : (
                <>
                  <svg
                    className="h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Run Predictions
                </>
              )}
            </button>
          </article>

          {/* Prediction Results */}
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-6 transition-colors hover:border-slate-700">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-violet-500" />
              <h2 className="text-base font-semibold">Prediction Results</h2>
            </div>
            <p className="mt-1 text-sm text-slate-400">Runtime, VRAM, and failure risk estimates.</p>

            <div className="mt-5 grid grid-cols-3 gap-3">
              <StatCard
                label="Runtime"
                value={prediction.runtime?.value?.toFixed(2) ?? "--"}
                unit="hrs"
                colorClass="text-cyan-400"
                confidence={prediction.runtime?.confidence}
                animate={hasPrediction}
              />
              <StatCard
                label="Peak VRAM"
                value={prediction.vram?.value?.toFixed(1) ?? "--"}
                unit="GB"
                colorClass="text-violet-400"
                confidence={prediction.vram?.confidence}
                animate={hasPrediction}
              />
              <StatCard
                label="Fail Risk"
                value={
                  prediction.failure
                    ? `${(prediction.failure.probability * 100).toFixed(1)}%`
                    : "--"
                }
                colorClass={hasPrediction ? failureColorClass : "text-slate-400"}
                confidence={prediction.failure?.confidence}
                animate={hasPrediction}
              />
            </div>

            {prediction.failure?.likely_reason && (
              <div className="mt-3 animate-fade-in rounded-lg border border-amber-800/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
                <span className="font-semibold">Likely reason: </span>
                {prediction.failure.likely_reason}
              </div>
            )}

            <div className="mt-4 h-52 rounded-lg border border-slate-800 bg-slate-950 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} barSize={32}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#475569" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {chartData.map((_, index) => (
                      <Cell key={index} fill={BAR_COLORS[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          {/* Recommendations */}
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-6 transition-colors hover:border-slate-700">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-emerald-500" />
              <h2 className="text-base font-semibold">Resource Recommendations</h2>
            </div>
            <p className="mt-1 text-sm text-slate-400">Optimization suggestions from the prediction service.</p>

            {prediction.recommendation && (
              <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-center">
                  <p className="text-xs uppercase tracking-wide text-slate-500">GPUs</p>
                  <p className="mt-1 text-xl font-bold text-cyan-400">
                    {prediction.recommendation.recommended_gpu_count}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-center">
                  <p className="text-xs uppercase tracking-wide text-slate-500">VRAM</p>
                  <p className="mt-1 text-xl font-bold text-violet-400">
                    {prediction.recommendation.recommended_vram_gb} GB
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-center">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Est. Runtime</p>
                  <p className="mt-1 text-xl font-bold text-emerald-400">
                    {prediction.recommendation.expected_runtime_hours?.toFixed(1)}h
                  </p>
                </div>
              </div>
            )}

            <ul className="mt-4 space-y-2 text-sm">
              {(
                prediction.recommendation?.optimization_suggestions ?? [
                  "Run a prediction to see optimization suggestions.",
                ]
              ).map((item, i) => (
                <li
                  key={i}
                  className={`flex items-start gap-2.5 rounded-lg border border-slate-800 bg-slate-950 p-3 ${
                    hasPrediction ? "animate-fade-in-up" : ""
                  }`}
                  style={
                    hasPrediction
                      ? { animationDelay: `${i * 80}ms`, animationFillMode: "both" }
                      : {}
                  }
                >
                  <svg
                    className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-emerald-500"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-slate-300">{item}</span>
                </li>
              ))}
            </ul>
          </article>

          {/* Experiment Tracking */}
          <article className="rounded-xl border border-slate-800 bg-slate-900 p-6 transition-colors hover:border-slate-700">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-amber-500" />
                  <h2 className="text-base font-semibold">Experiment Tracking</h2>
                </div>
                <p className="mt-1 text-sm text-slate-400">Recent training runs and metrics.</p>
              </div>
              <button
                onClick={loadExperiments}
                disabled={historyLoading}
                className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 transition hover:border-slate-500 hover:text-slate-100 disabled:opacity-50"
              >
                {historyLoading ? (
                  <Spinner className="h-3 w-3" />
                ) : (
                  <svg
                    className="h-3 w-3"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                )}
                Load History
              </button>
            </div>

            <div className="mt-4 max-h-72 space-y-2 overflow-auto pr-1 text-xs">
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 py-10 text-center text-slate-600">
                  <svg
                    className="mb-2 h-6 w-6"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  No experiment history loaded
                </div>
              ) : (
                history.map((row, idx) => (
                  <div
                    key={`${idx}-${String(row.run_id ?? "row")}`}
                    className="animate-fade-in rounded-lg border border-slate-800 bg-slate-950 p-3"
                    style={{ animationDelay: `${idx * 50}ms`, animationFillMode: "both" }}
                  >
                    {Object.entries(row).map(([k, v]) => (
                      <div key={k} className="flex justify-between py-0.5">
                        <span className="text-slate-500">{k}</span>
                        <span className="font-mono text-slate-300">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                ))
              )}
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
