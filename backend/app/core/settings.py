from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ClusterMind AI API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "clustermind"
    postgres_user: str = "clustermind"
    postgres_password: str = "clustermind"

    log_level: str = "INFO"
    model_bundle_path: str = "data/models/model_bundle.joblib"
    experiments_path: str = "data/experiments/runs.jsonl"
    anomaly_bundle_path: str = "data/models/anomaly_bundle.joblib"
    scheduler_enabled: bool = True
    metrics_enabled: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()

