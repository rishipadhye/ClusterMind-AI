from __future__ import annotations

import pandas as pd


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["num_gpus"] = enriched["num_gpus"].clip(lower=1)
    enriched["flops_per_gpu"] = enriched["estimated_flops"] / enriched["num_gpus"]
    enriched["memory_pressure_score"] = (
        (enriched["batch_size"] * (enriched["dataset_size"] / 1_000_000)) / enriched["ram_gb"].clip(lower=1)
    )
    enriched["compute_intensity_score"] = (
        enriched["estimated_flops"] / (enriched["cpu_cores"] * enriched["ram_gb"]).clip(lower=1)
    )
    enriched["gpu_efficiency_ratio"] = enriched["gpu_utilization"] / (enriched["num_gpus"] * 100)
    enriched["oom_risk_hint"] = (
        (enriched["batch_size"] > 256) & (enriched["gpu_type"].isin(["t4", "rtx4090"]))
    ).astype(int)
    return enriched

