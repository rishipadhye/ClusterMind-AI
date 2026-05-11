import axios from "axios";

export type JobMetadata = {
  model_type: "resnet" | "bert" | "gpt" | "vision_transformer" | "stable_diffusion" | "recommendation";
  framework: "pytorch" | "tensorflow" | "jax";
  gpu_type: "t4" | "v100" | "a100" | "rtx4090";
  num_gpus: number;
  batch_size: number;
  dataset_size: number;
  cpu_cores: number;
  ram_gb: number;
  estimated_flops: number;
  distributed_training: boolean;
  mixed_precision: boolean;
};

const client = axios.create({
  baseURL: "http://localhost:8000/api/v1",
  timeout: 10000,
});

export async function predictAll(payload: JobMetadata) {
  const [runtime, vram, failure, recommendation] = await Promise.all([
    client.post("/predict/runtime", payload),
    client.post("/predict/vram", payload),
    client.post("/predict/failure", payload),
    client.post("/recommend/resources", payload),
  ]);
  return {
    runtime: runtime.data,
    vram: vram.data,
    failure: failure.data,
    recommendation: recommendation.data,
  };
}

export async function fetchExperimentHistory() {
  const response = await client.get("/experiments/history?limit=20");
  return response.data as Array<Record<string, unknown>>;
}

