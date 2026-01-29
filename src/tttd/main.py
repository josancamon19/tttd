"""Entry point for Erdős minimum overlap optimization training."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook.rl import train

from tttd.erdos.dataset import ErdosDatasetBuilder
from tttd.erdos.executor import shutdown_executor

logger = logging.getLogger(__name__)


@chz.chz
class TrainConfig:
    """Training configuration for Erdős TTT."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    learning_rate: float = 4e-5
    max_tokens: int = 4096
    lora_rank: int = 32

    # Dataset settings
    num_batches: int = 10
    batch_size: int = 2
    group_size: int = 4
    timeout: int = 60

    # Sampler settings
    sampler_type: str = "puct"  # "puct" or "greedy"
    puct_c: float = 1.0
    max_buffer_size: int = 500
    topk_children: int = 2

    # Training settings
    loss_fn: str = "importance_sampling"  # "importance_sampling" or "ppo"
    eval_every: int = 0
    save_every: int = 5
    remove_constant_reward_groups: bool = True

    # Output settings
    output_dir: str = "./outputs"
    verbose: bool = False  # Log all rollouts to output_dir


class RolloutLogger:
    """Logs rollouts and sampler state to files."""

    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.enabled:
            self.run_dir = self.output_dir / f"run_{self.run_id}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verbose logging enabled. Output dir: {self.run_dir}")

    def log_sampler_state(self, step: int, sampler):
        """Log sampler state at the beginning of a step."""
        if not self.enabled:
            return

        state_file = self.run_dir / f"step_{step:04d}_sampler_state.json"

        stats = sampler.get_stats()
        best_state = sampler.get_best_state()

        state_info = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "best_state": {
                "id": best_state.id if best_state else None,
                "value": best_state.value if best_state else None,
                "c5_bound": best_state.c5_bound if best_state else None,
            }
            if best_state
            else None,
            "buffer_summary": self._get_buffer_summary(sampler),
        }

        with open(state_file, "w") as f:
            json.dump(state_info, f, indent=2)

        logger.info(
            f"[Step {step}] Sampler: buffer={stats['puct/buffer_size']}, "
            f"T={stats['puct/T']}, best_c5={best_state.c5_bound:.4f}"
            if best_state
            else "empty"
        )

    def _get_buffer_summary(self, sampler) -> list[dict]:
        """Get summary of states in buffer."""
        summary = []
        for s in sampler._states[:10]:  # Top 10 by insertion order
            summary.append(
                {
                    "id": s.id[:8],
                    "value": s.value,
                    "c5_bound": s.c5_bound,
                    "n_visits": sampler._n.get(s.id, 0),
                }
            )
        return summary

    def log_rollouts(self, step: int, builders, children_by_parent: dict):
        """Log all rollouts from a step."""
        if not self.enabled:
            return

        rollouts_file = self.run_dir / f"step_{step:04d}_rollouts.json"

        rollouts = []
        for builder in builders:
            parent = builder.get_parent_state()
            if parent is None:
                continue

            parent_info = {
                "id": parent.id[:8],
                "value": parent.value,
                "c5_bound": parent.c5_bound,
            }

            children_info = []
            for child in builder.get_child_states():
                if child is not None:
                    children_info.append(
                        {
                            "id": child.id[:8],
                            "value": child.value,
                            "c5_bound": child.c5_bound,
                            "code_preview": child.code[:200] if child.code else None,
                            "improved": child.c5_bound < parent.c5_bound
                            if child.c5_bound and parent.c5_bound
                            else None,
                        }
                    )
                else:
                    children_info.append({"status": "failed"})

            rollouts.append(
                {
                    "parent": parent_info,
                    "children": children_info,
                    "num_successful": sum(
                        1 for c in children_info if c.get("value") is not None
                    ),
                    "num_improved": sum(1 for c in children_info if c.get("improved")),
                }
            )

        output = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_parents": len(rollouts),
            "total_rollouts": sum(len(r["children"]) for r in rollouts),
            "total_successful": sum(r["num_successful"] for r in rollouts),
            "total_improved": sum(r["num_improved"] for r in rollouts),
            "rollouts": rollouts,
        }

        with open(rollouts_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            f"[Step {step}] Rollouts: {output['total_rollouts']} total, "
            f"{output['total_successful']} successful, {output['total_improved']} improved"
        )

    def log_final_summary(self, sampler):
        """Log final training summary."""
        if not self.enabled:
            return

        summary_file = self.run_dir / "final_summary.json"

        best = sampler.get_best_state()
        stats = sampler.get_stats()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "final_stats": stats,
            "best_solution": {
                "id": best.id,
                "c5_bound": best.c5_bound,
                "value": best.value,
                "code": best.code,
                "h_values_len": len(best.h_values) if best.h_values else 0,
            }
            if best
            else None,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if best:
            # Also save the best solution code separately
            code_file = self.run_dir / "best_solution.py"
            with open(code_file, "w") as f:
                f.write(f"# Best solution found\n")
                f.write(f"# C5 bound: {best.c5_bound}\n")
                f.write(f"# Value: {best.value}\n\n")
                f.write(best.code or "# No code available")

            logger.info(f"Final best c5_bound: {best.c5_bound:.6f}")


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

    # Initialize rollout logger
    rollout_logger = RolloutLogger(config.output_dir, enabled=config.verbose)

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

    # Build dataset to get sampler reference
    dataset, _ = await dataset_builder()

    # Wrap dataset to add logging hooks
    if config.verbose:
        dataset = VerboseDatasetWrapper(dataset, rollout_logger)

    # Create training config
    cfg = train.Config(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        dataset_builder=dataset_builder,
        loss_fn=config.loss_fn,
        log_path=log_path,
        eval_every=config.eval_every,
        save_every=config.save_every,
        remove_constant_reward_groups=config.remove_constant_reward_groups,
    )

    try:
        # Log initial sampler state
        if config.verbose:
            rollout_logger.log_sampler_state(0, dataset.get_sampler())

        await train.main(cfg)

        # Log final summary
        if config.verbose:
            rollout_logger.log_final_summary(dataset.get_sampler())

    finally:
        shutdown_executor()


class VerboseDatasetWrapper:
    """Wrapper that adds verbose logging to dataset operations."""

    def __init__(self, dataset, rollout_logger: RolloutLogger):
        self._dataset = dataset
        self._logger = rollout_logger
        self._current_step = 0

    def get_batch(self, index: int):
        # Log sampler state before getting batch
        self._logger.log_sampler_state(index, self._dataset.get_sampler())

        # Get the batch
        builders = self._dataset.get_batch(index)

        # Log rollouts from previous batch (now that they've been processed)
        if index > 0 and self._dataset._last_builders:
            self._logger.log_rollouts(index - 1, self._dataset._last_builders, {})

        self._current_step = index
        return builders

    def __len__(self):
        return len(self._dataset)

    def get_sampler(self):
        return self._dataset.get_sampler()

    def finalize(self, step: int):
        # Log final rollouts
        if self._dataset._last_builders:
            self._logger.log_rollouts(step, self._dataset._last_builders, {})
        self._dataset.finalize(step)


if __name__ == "__main__":
    main()
