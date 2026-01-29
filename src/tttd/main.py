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
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


@chz.chz
class TrainConfig:
    model_name: str = "openai/gpt-oss-20b"
    learning_rate: float = 4e-5
    max_tokens: int = 26000
    lora_rank: int = 32
    temperature: float = 1.0
    num_steps: int = 50
    batch_size: int = 64
    group_size: int = 8
    timeout: int = 1100
    sampler_type: str = "greedy"
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children: int = 2
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0
    eval_every: int = 3
    save_every: int = 5
    remove_constant_reward_groups: bool = True
    adv_estimator: Literal["mean_baseline", "entropic", "entropic_adaptive_beta"] = (
        "entropic_adaptive_beta"
    )
    adv_estimator_beta: float = 2.0
    output_dir: str = "./outputs"
    verbose: bool = False
    wandb_project: str | None = "tttd-erdos"
    wandb_name: str | None = None
    seed: int = 42


async def _async_main(config: TrainConfig):
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(output_dir / "sampler")

    # Create dataset builder
    dataset_builder = ErdosDatasetBuilder(
        num_batches=config.num_steps,
        batch_size=config.batch_size,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        timeout=config.timeout,
        log_path=log_path,
        sampler_type=config.sampler_type,
        puct_c=config.puct_c,
        max_buffer_size=config.max_buffer_size,
        topk_children=config.topk_children,
        seed=config.seed,
    )

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
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
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


def main():
    config = chz.entrypoint(TrainConfig)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger.info("Starting Erdős TTT training")
    logger.info(f"Config: {config}")
    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
