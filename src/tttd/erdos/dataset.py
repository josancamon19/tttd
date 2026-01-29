"""Erdős dataset for RL training."""

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

from tttd.erdos.env import ErdosEnvGroupBuilder, ErdosState


class ErdosDataset(RLDataset):
    """Dataset that produces batches of ErdosEnvGroupBuilder instances.

    Since Erdős is a single optimization problem (not a dataset of problems),
    every batch returns the same set of env group builders with shared state.
    """

    def __init__(
        self,
        renderer: Renderer,
        num_batches: int,
        batch_size: int,
        group_size: int,
        timeout: int,
    ):
        self._renderer = renderer
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._group_size = group_size
        self._timeout = timeout
        # Shared state across all batches - best solution persists
        self._shared_state = ErdosState()

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            ErdosEnvGroupBuilder(
                renderer=self._renderer,
                group_size=self._group_size,
                timeout=self._timeout,
                shared_state=self._shared_state,
            )
            for _ in range(self._batch_size)
        ]

    def __len__(self) -> int:
        return self._num_batches

    def get_best_state(self) -> ErdosState:
        """Get the best solution found so far."""
        return self._shared_state


@chz.chz
class ErdosDatasetBuilder(RLDatasetBuilder):
    """Builder for ErdosDataset, compatible with tinker_cookbook's train.Config."""

    num_batches: int = 10
    batch_size: int = 2
    group_size: int = 4
    model_name_for_tokenizer: str = "Qwen/Qwen3-8B"
    renderer_name: str = ""  # Empty string means auto-detect
    timeout: int = 60

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer_name = self.renderer_name
        if not renderer_name:
            renderer_name = get_recommended_renderer_name(self.model_name_for_tokenizer)

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(renderer_name, tokenizer)

        dataset = ErdosDataset(
            renderer=renderer,
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            group_size=self.group_size,
            timeout=self.timeout,
        )

        return dataset, None
