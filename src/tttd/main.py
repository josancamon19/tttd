import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Literal

import chz
from tinker_cookbook.rl import train
import tinker_cookbook.rl.data_processing as data_processing

from tttd.advantages import compute_advantages as custom_compute_advantages
from tttd.erdos.dataset import ErdosDatasetBuilder
from tttd.erdos.executor import shutdown_executor

logger = logging.getLogger(__name__)


@chz.chz
class TrainConfig:
    """Training configuration for Erdős TTT.

    Defaults match TTT-Discover repo settings for Erdős minimum overlap.
    """

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    learning_rate: float = 4e-5
    max_tokens: int = 26000  # TTT-Discover uses 26000
    lora_rank: int = 32
    temperature: float = 1.0

    # Dataset settings
    num_batches: int = 50  # TTT-Discover num_epochs=50
    batch_size: int = 64  # TTT-Discover groups_per_batch=64
    group_size: int = 8  # TTT-Discover group_size=8
    timeout: int = 1100  # TTT-Discover eval_timeout=1100 for Erdős

    # Sampler settings
    sampler_type: str = "greedy"  # TTT-Discover uses greedy by default
    puct_c: float = 1.0
    max_buffer_size: int = 1000  # TTT-Discover uses 1000
    topk_children: int = 2

    # Training settings
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0
    eval_every: int = 3  # TTT-Discover default; runs evaluators every N steps
    save_every: int = 5
    remove_constant_reward_groups: bool = True

    # Advantage estimation - TTT-Discover uses entropic with beta=2.0
    adv_estimator: Literal["mean_baseline", "entropic", "entropic_adaptive_beta"] = (
        "entropic"
    )
    adv_estimator_beta: float = 2.0  # TTT-Discover uses beta=2.0

    # Output settings
    output_dir: str = "./outputs"
    verbose: bool = False  # Log all rollouts to output_dir


def main():
    """Entry point - parse args with chz and run training."""
    chz.entrypoint(run_training)


def run_training(config: TrainConfig):
    """Run training with the given configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    logger.info("Starting Erdős TTT training")
    logger.info(f"Config: {config}")

    asyncio.run(_async_main(config))


async def _async_main(config: TrainConfig):
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(output_dir / "sampler")

    # Create dataset builder
    dataset_builder = ErdosDatasetBuilder(
        num_batches=config.num_batches,
        batch_size=config.batch_size,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        timeout=config.timeout,
        log_path=log_path,
        sampler_type=config.sampler_type,
        puct_c=config.puct_c,
        max_buffer_size=config.max_buffer_size,
        topk_children=config.topk_children,
    )

    dataset, _ = await dataset_builder()

    cfg = train.Config(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        dataset_builder=dataset_builder,
        loss_fn=config.loss_fn,
        num_substeps=config.num_substeps,
        kl_penalty_coef=config.kl_penalty_coef,
        log_path=log_path,
        eval_every=config.eval_every,
        save_every=config.save_every,
        remove_constant_reward_groups=config.remove_constant_reward_groups,
    )

    # Monkey-patch tinker_cookbook's compute_advantages to use our custom estimator
    # This injects TTT-Discover's entropic advantages without modifying the library
    logger.info(f"Using advantage estimator: {config.adv_estimator}")
    data_processing.compute_advantages = partial(
        custom_compute_advantages,
        estimator=config.adv_estimator,
        beta=config.adv_estimator_beta,
    )

    try:
        await train.main(cfg)
    finally:
        shutdown_executor()


if __name__ == "__main__":
    main()
