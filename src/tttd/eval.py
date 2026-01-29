"""Baseline evaluation for Erdős problem."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
import numpy as np
import tinker
from dotenv import load_dotenv
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tttd.erdos.env import ErdosEnv
from tttd.sampler import create_initial_erdos_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


@chz.chz
class EvalConfig:
    model: Literal[
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
    ] = "openai/gpt-oss-20b"
    num_samples: int = 8
    seed: int = 42
    max_tokens: int = 26000
    temperature: float = 1.0
    timeout: int = 1100
    results_dir: str = "./results"


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


async def evaluate_model(config: EvalConfig) -> dict:
    logger.info(f"Evaluating: {config.model}")
    state = create_initial_erdos_state(seed=config.seed)
    logger.info(f"  Initial: n={len(state.h_values)}, c5={state.c5_bound:.6f}")

    # Setup
    renderer_name = get_recommended_renderer_name(config.model)
    tokenizer = get_tokenizer(config.model)
    renderer = get_renderer(renderer_name, tokenizer)

    # Use ErdosEnv for prompt and evaluation
    env = ErdosEnv(renderer, timeout=config.timeout, parent_state=state, hide_code=True)
    prompt, stop_cond = await env.initial_observation()
    stop_seqs = getattr(stop_cond, "stop_sequences", stop_cond) or []

    # Sample
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=config.model)
    response = await client.sample_async(
        prompt=prompt,
        num_samples=config.num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=stop_seqs,
        ),
    )

    # Evaluate each sample using fresh envs
    results = []
    for seq in response.sequences:
        env = ErdosEnv(renderer, timeout=config.timeout, parent_state=state, hide_code=True)
        await env.initial_observation()
        step_result = await env.step(seq.tokens)
        results.append({
            "success": step_result.metrics.get("correct", 0) == 1,
            "c5_bound": step_result.metrics.get("c5_bound"),
            "score": step_result.metrics.get("score", 0),
            "error": step_result.logs.get("msg") if not step_result.metrics.get("correct") else None,
        })

    # Metrics
    successful = [r for r in results if r["success"]]
    c5_bounds = [r["c5_bound"] for r in successful if r["c5_bound"]]

    metrics = {
        "model": config.model,
        "initial_c5": state.c5_bound,
        "num_samples": config.num_samples,
        "num_correct": len(successful),
        "success_rate": len(successful) / config.num_samples,
        "best_c5": min(c5_bounds) if c5_bounds else None,
        "mean_c5": float(np.mean(c5_bounds)) if c5_bounds else None,
        "samples": results,
    }
    for k in [1, 2, 4, 8]:
        if k <= config.num_samples:
            metrics[f"pass@{k}"] = pass_at_k(config.num_samples, len(successful), k)

    return metrics


async def main(config: EvalConfig):
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = await evaluate_model(config)

    path = results_dir / f"{config.model.replace('/', '_')}_{timestamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))

    logger.info(f"Success: {result['success_rate']:.0%} ({result['num_correct']}/{result['num_samples']})")
    if result["best_c5"]:
        logger.info(f"Best C5: {result['best_c5']:.6f}, Mean: {result['mean_c5']:.6f}")
    logger.info(f"pass@1: {result.get('pass@1', 0):.3f}")


def run():
    config = chz.entrypoint(EvalConfig)
    asyncio.run(main(config))


if __name__ == "__main__":
    run()
