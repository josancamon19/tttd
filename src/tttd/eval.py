"""Baseline evaluation for Erdős problem across multiple models."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Literal
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import tinker

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tttd.erdos.env import get_improvement_prompt
from tttd.erdos.executor import extract_code_block, run_erdos_eval
from tttd.sampler import create_initial_erdos_state, ErdosState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
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
    """Compute pass@k: probability of at least one correct in k samples."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


async def evaluate_model(model_name: str, config: EvalConfig) -> dict:
    """Run all samples for one model."""
    logger.info(f"Evaluating: {model_name}")
    state = create_initial_erdos_state(seed=config.seed)
    logger.info(f"  Initial: n={len(state.h_values)}, c5={state.c5_bound:.6f}")

    # Setup once
    renderer_name = get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=model_name)

    stop_cond = renderer.get_stop_sequences()
    if hasattr(stop_cond, "stop_sequences"):
        stop_seqs = stop_cond.stop_sequences
    elif isinstance(stop_cond, list):
        stop_seqs = stop_cond
    else:
        stop_seqs = []

    prompt = get_improvement_prompt(state, hide_code=True)
    model_input = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt}]
    )

    # Sample all at once
    response = await client.sample_async(
        prompt=model_input,
        num_samples=config.num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=stop_seqs,
        ),
    )

    # Evaluate each sample
    results = []
    for seq in response.sequences:
        text = tokenizer.decode(seq.tokens)
        code = extract_code_block(text)

        if not code or "def run" not in code:
            results.append(
                {
                    "success": False,
                    "c5_bound": None,
                    "error": "No valid code block",
                    "code": code,
                    "text": text[:1500],
                }
            )
            continue

        result = run_erdos_eval(
            code, timeout=config.timeout, initial_h_values=state.h_values
        )
        results.append(
            {
                "success": result["success"],
                "c5_bound": result.get("c5_bound"),
                "error": None if result["success"] else result.get("msg"),
                "code": code,
                "text": text[:1500],
                "stdout": result.get("stdout", "")[:500],
            }
        )

    # Metrics
    successful = [r for r in results if r["success"]]
    c5_bounds = [r["c5_bound"] for r in successful if r["c5_bound"]]

    metrics = {
        "model": model_name,
        "initial_c5": state.c5_bound,
        "num_samples": config.num_samples,
        "num_correct": len(successful),
        "success_rate": len(successful) / config.num_samples,
        "best_c5": min(c5_bounds) if c5_bounds else None,
        "mean_c5": float(np.mean(c5_bounds)) if c5_bounds else None,
        "samples": results,
    }

    # Add pass@k
    for k in [1, 2, 4, 8]:
        if k <= config.num_samples:
            metrics[f"pass@{k}"] = pass_at_k(config.num_samples, len(successful), k)

    return metrics


async def main(config: EvalConfig):
    """Run evaluation for a single model."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = await evaluate_model(config.model, config)

    # Save result
    path = results_dir / f"{config.model.replace('/', '_')}_{timestamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))

    # Log summary
    logger.info(
        f"Success: {result['success_rate']:.0%} ({result['num_correct']}/{result['num_samples']})"
    )
    if result["best_c5"]:
        logger.info(f"Best C5: {result['best_c5']:.6f}, Mean: {result['mean_c5']:.6f}")
    logger.info(f"pass@1: {result.get('pass@1', 0):.3f}")


def run():
    asyncio.run(main(EvalConfig()))


if __name__ == "__main__":
    run()
