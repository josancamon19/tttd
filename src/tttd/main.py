import asyncio
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import chz
import tinker
from dotenv import load_dotenv
from tinker_cookbook.rl import train
from tinker_cookbook.rl.train import (
    do_group_rollout,
    TinkerTokenCompleter,
    TrajectoryGroup,
    all_same,
    logtree,
    scope,
)
from tinker_cookbook.rl.types import EnvGroupBuilder
import tinker_cookbook.rl.data_processing as data_processing

from tttd.advantages import compute_advantages as custom_compute_advantages
from tttd.completers import TwoPhaseTokenCompleter
from tttd.erdos.dataset import ErdosDatasetBuilder
from tttd.erdos.executor import shutdown_executor

logger = logging.getLogger(__name__)

load_dotenv()

# Global config for two-phase sampling (set during training setup)
_two_phase_config: dict = {"enabled": False, "phase1_max_tokens": 26000, "tokenizer": None}


def _create_patched_rollout_fn():
    """Create a patched version of do_group_rollout_and_filter_constant_reward that supports two-phase sampling."""

    @scope
    async def patched_do_group_rollout_and_filter_constant_reward(
        sampling_client: tinker.SamplingClient,
        env_group_builder: EnvGroupBuilder,
        max_tokens: int,
        temperature: float,
        do_remove_constant_reward_groups: bool,
        enable_logging: bool = True,
    ) -> TrajectoryGroup | None:
        # Use two-phase sampling if enabled and tokenizer is available
        if _two_phase_config["enabled"] and _two_phase_config["tokenizer"] is not None:
            policy = TwoPhaseTokenCompleter(
                sampling_client=sampling_client,
                tokenizer=_two_phase_config["tokenizer"],
                phase1_max_tokens=_two_phase_config["phase1_max_tokens"],
                temperature=temperature,
            )
        else:
            policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)

        with logtree.optional_enable_logging(enable_logging):
            trajectory_group = await do_group_rollout(env_group_builder, policy)

        # Remove if all trajectories have the same reward
        if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
            return None
        else:
            return trajectory_group

    return patched_do_group_rollout_and_filter_constant_reward


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
    run_name: str | None = None
    resume: bool = False
    load_checkpoint_path: str | None = None
    verbose: bool = False
    wandb_project: str | None = "tttd-erdos"
    wandb_name: str | None = None
    seed: int = 42
    # Two-phase sampling: if model exhausts phase1 tokens without stop, force completion
    two_phase_sampling: bool = False
    phase1_max_tokens: int = 26000


def _generate_run_name(config: TrainConfig) -> str:
    model_short = config.model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"erdos-{model_short}-{config.sampler_type}-{timestamp}"


async def _async_main(config: TrainConfig):
    # Determine output directory
    if config.resume:
        # Resume: use existing output_dir
        output_dir = Path(config.output_dir)
        if not output_dir.exists():
            raise ValueError(f"Cannot resume: {output_dir} does not exist")
        logger.info(f"Resuming from {output_dir}")
    else:
        # Fresh run: create unique directory
        run_name = config.run_name or _generate_run_name(config)
        output_dir = Path(config.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting fresh run: {output_dir}")

    log_path = str(output_dir)

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

    # Determine wandb run name
    wandb_name = config.wandb_name or output_dir.name

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
        load_checkpoint_path=config.load_checkpoint_path,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
    )

    # Monkey-patch tinker_cookbook's compute_advantages to use our custom estimator
    logger.info(f"Using advantage estimator: {config.adv_estimator}")
    data_processing.compute_advantages = partial(
        custom_compute_advantages,
        estimator=config.adv_estimator,
        beta=config.adv_estimator_beta,
    )

    # Setup two-phase sampling if enabled
    if config.two_phase_sampling:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        tokenizer = get_tokenizer(config.model_name)
        _two_phase_config["enabled"] = True
        _two_phase_config["phase1_max_tokens"] = config.phase1_max_tokens
        _two_phase_config["tokenizer"] = tokenizer
        # Monkey-patch the rollout function to use two-phase sampling
        train.do_group_rollout_and_filter_constant_reward = _create_patched_rollout_fn()
        logger.info(f"Two-phase sampling enabled: phase1_max_tokens={config.phase1_max_tokens}")

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
