"""Entry point for Erdős minimum overlap optimization training."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from tinker_cookbook.rl import train

from tttd.erdos.dataset import ErdosDatasetBuilder
from tttd.erdos.executor import shutdown_executor

logger = logging.getLogger(__name__)


def main():
    """Load environment, build config, and run the training loop."""
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    asyncio.run(_async_main())


async def _async_main():
    model_name = os.environ.get("TTTD_MODEL_NAME", "Qwen/Qwen3-8B")
    num_batches = int(os.environ.get("TTTD_NUM_BATCHES", "10"))
    batch_size = int(os.environ.get("TTTD_BATCH_SIZE", "2"))
    group_size = int(os.environ.get("TTTD_GROUP_SIZE", "4"))
    max_tokens = int(os.environ.get("TTTD_MAX_TOKENS", "4096"))
    learning_rate = float(os.environ.get("TTTD_LEARNING_RATE", "4e-5"))
    timeout = int(os.environ.get("TTTD_TIMEOUT", "60"))
    log_path = os.environ.get("TTTD_LOG_PATH", "/tmp/tttd-erdos")

    dataset_builder = ErdosDatasetBuilder(
        num_batches=num_batches,
        batch_size=batch_size,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        timeout=timeout,
    )

    cfg = train.Config(
        model_name=model_name,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        lora_rank=32,
        dataset_builder=dataset_builder,
        loss_fn="importance_sampling",
        log_path=log_path,
        eval_every=0,
        save_every=5,
        remove_constant_reward_groups=True,
    )

    try:
        await train.main(cfg)
    finally:
        shutdown_executor()
