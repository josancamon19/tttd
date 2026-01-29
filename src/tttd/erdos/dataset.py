"""Erdős dataset for RL training with PUCT sampler integration."""

from __future__ import annotations

from typing import Sequence

import chz

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tttd.erdos.env import ErdosEnvGroupBuilder
from tttd.sampler import ErdosState, PUCTSampler, GreedySampler, StateSampler


class ErdosEvalDataset(RLDataset):
    """Simple eval dataset - fresh random states with no prior context.

    Used to measure if the model can produce good solutions from scratch.
    """

    def __init__(
        self,
        renderer: Renderer,
        num_batches: int,
        group_size: int,
        timeout: int,
    ):
        self._renderer = renderer
        self._num_batches = num_batches
        self._group_size = group_size
        self._timeout = timeout

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        from tttd.sampler import create_initial_erdos_state

        # Single fresh state per eval batch, no prior code shown
        fresh_state = create_initial_erdos_state()
        return [
            ErdosEnvGroupBuilder(
                renderer=self._renderer,
                group_size=self._group_size,
                timeout=self._timeout,
                parent_state=fresh_state,
                hide_code=True,  # Don't show any prior code
            )
        ]

    def __len__(self) -> int:
        return self._num_batches


class ErdosDataset(RLDataset):
    """Dataset that produces batches of ErdosEnvGroupBuilder instances.

    Uses PUCT sampler to select which historical states to expand.
    After each batch, child states should be fed back to the sampler.
    """

    def __init__(
        self,
        renderer: Renderer,
        num_batches: int,
        batch_size: int,
        group_size: int,
        timeout: int,
        sampler: StateSampler,
    ):
        self._renderer = renderer
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._group_size = group_size
        self._timeout = timeout
        self._sampler = sampler

        # Track builders from last batch for state updates
        self._last_builders: list[ErdosEnvGroupBuilder] = []

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # First, update sampler with results from previous batch
        self._update_sampler_from_last_batch(step=index)

        # Sample parent states from PUCT
        parent_states = self._sampler.sample_states(self._batch_size)

        # Create env group builders, one per parent state
        builders = [
            ErdosEnvGroupBuilder(
                renderer=self._renderer,
                group_size=self._group_size,
                timeout=self._timeout,
                parent_state=parent_state,
            )
            for parent_state in parent_states
        ]

        self._last_builders = builders
        return builders

    def _update_sampler_from_last_batch(self, step: int):
        """Collect child states from last batch and update sampler."""
        if not self._last_builders:
            return

        all_children: list[ErdosState] = []
        all_parents: list[ErdosState] = []

        for builder in self._last_builders:
            parent = builder.get_parent_state()
            if parent is None:
                continue

            has_valid_child = False
            for child in builder.get_child_states():
                if child is not None and child.value is not None:
                    child.timestep = step
                    all_children.append(child)
                    all_parents.append(parent)
                    has_valid_child = True

            # Record failed rollout if no valid children from this parent
            if not has_valid_child:
                self._sampler.record_failed_rollout(parent)

        if all_children:
            self._sampler.update_states(all_children, all_parents, step=step)

    def __len__(self) -> int:
        return self._num_batches

    def get_sampler(self) -> StateSampler:
        """Get the sampler for external access."""
        return self._sampler

    def finalize(self, step: int):
        """Finalize after training - update sampler with last batch results."""
        self._update_sampler_from_last_batch(step=step)
        self._sampler.flush(step=step)


@chz.chz
class ErdosDatasetBuilder(RLDatasetBuilder):
    """Builder for ErdosDataset with PUCT sampler."""

    num_batches: int = 10
    batch_size: int = 2
    group_size: int = 4
    model_name_for_tokenizer: str = "Qwen/Qwen3-8B"
    renderer_name: str = ""  # Empty string means auto-detect
    timeout: int = 60

    # Sampler config
    log_path: str = "./tmp/tttd-erdos"
    sampler_type: str = "puct"  # "puct" or "greedy"
    puct_c: float = 1.0
    max_buffer_size: int = 500
    topk_children: int = 2

    # Eval config
    eval_batches: int = 1  # Number of eval batches (fresh starts)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer_name = self.renderer_name
        if not renderer_name:
            renderer_name = get_recommended_renderer_name(self.model_name_for_tokenizer)

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(renderer_name, tokenizer)

        # Create sampler
        if self.sampler_type == "puct":
            sampler = PUCTSampler(
                log_path=self.log_path,
                max_buffer_size=self.max_buffer_size,
                puct_c=self.puct_c,
                topk_children=self.topk_children,
            )
        elif self.sampler_type == "greedy":
            sampler = GreedySampler(
                log_path=self.log_path,
                max_buffer_size=self.max_buffer_size,
                topk_children=self.topk_children,
            )
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

        dataset = ErdosDataset(
            renderer=renderer,
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            group_size=self.group_size,
            timeout=self.timeout,
            sampler=sampler,
        )

        # Eval dataset: fresh starts, no prior code context
        eval_dataset = ErdosEvalDataset(
            renderer=renderer,
            num_batches=self.eval_batches,
            group_size=self.group_size,
            timeout=self.timeout,
        )

        return dataset, eval_dataset
