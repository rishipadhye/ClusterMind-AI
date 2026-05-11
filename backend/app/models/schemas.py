from typing import Literal, Optional

from pydantic import BaseModel, Field


ModelType = Literal[
    "resnet",
    "bert",
    "gpt",
    "vision_transformer",
    "stable_diffusion",
    "recommendation",
]
FrameworkType = Literal["pytorch", "tensorflow", "jax"]
GpuType = Literal["t4", "v100", "a100", "rtx4090"]


class JobMetadata(BaseModel):
    model_type: ModelType
    framework: FrameworkType
    gpu_type: GpuType
    num_gpus: int = Field(ge=1, le=16)
    batch_size: int = Field(ge=1, le=4096)
    dataset_size: int = Field(ge=1)
    cpu_cores: int = Field(ge=1, le=256)
    ram_gb: float = Field(ge=1, le=2048)
    estimated_flops: float = Field(ge=0)
    distributed_training: bool
    mixed_precision: bool


class PredictionResponse(BaseModel):
    value: float
    confidence: float = Field(ge=0, le=1)
    model_version: str
    notes: Optional[str] = None


class FailurePredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    model_version: str
    likely_reason: Optional[str] = None


class ResourceRecommendationResponse(BaseModel):
    recommended_gpu_count: int
    recommended_vram_gb: float
    expected_runtime_hours: float
    failure_risk: float = Field(ge=0, le=1)
    optimization_suggestions: list[str]

