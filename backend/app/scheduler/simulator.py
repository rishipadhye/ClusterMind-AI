from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from app.models.schemas import JobMetadata
from app.services.prediction_service import PredictionService


@dataclass
class SimulationResult:
    avg_reward: float
    avg_runtime_hours: float
    avg_failure_prob: float
    allocation_histogram: dict[int, int]


class GreedyBanditScheduler:
    """Phase 9 stub: a simple policy that searches GPU count for best predicted reward.

    Reward = -runtime_hours - 4.0 * failure_probability
    """

    def __init__(self, predictor: PredictionService, max_gpus: int = 8) -> None:
        self.predictor = predictor
        self.max_gpus = max_gpus

    def choose_gpu_count(self, job: JobMetadata) -> int:
        best_k = job.num_gpus
        best_reward = -1e9
        for k in range(1, self.max_gpus + 1):
            candidate = job.model_copy(update={"num_gpus": k, "distributed_training": k > 1})
            runtime, _ = self.predictor.predict_runtime(candidate)
            fail_p, _, _ = self.predictor.predict_failure_probability(candidate)
            reward = -runtime - 4.0 * fail_p
            if reward > best_reward:
                best_reward = reward
                best_k = k
        return best_k


class SchedulerSimulator:
    def __init__(self, predictor: PredictionService, seed: int = 42) -> None:
        self.predictor = predictor
        self.rng = Random(seed)
        self.scheduler = GreedyBanditScheduler(predictor)

    def simulate(self, jobs: list[JobMetadata]) -> SimulationResult:
        rewards = []
        runtimes = []
        fails = []
        hist: dict[int, int] = {}

        for job in jobs:
            k = self.scheduler.choose_gpu_count(job)
            hist[k] = hist.get(k, 0) + 1

            chosen = job.model_copy(update={"num_gpus": k, "distributed_training": k > 1})
            runtime, _ = self.predictor.predict_runtime(chosen)
            fail_p, _, _ = self.predictor.predict_failure_probability(chosen)
            reward = -runtime - 4.0 * fail_p

            rewards.append(reward)
            runtimes.append(runtime)
            fails.append(fail_p)

        return SimulationResult(
            avg_reward=float(np.mean(rewards)) if rewards else 0.0,
            avg_runtime_hours=float(np.mean(runtimes)) if runtimes else 0.0,
            avg_failure_prob=float(np.mean(fails)) if fails else 0.0,
            allocation_histogram=hist,
        )
