"""Erdős reinforcement learning environment.

Single-episode environment for the Erdős minimum overlap problem.
Integrates with PUCT sampler for state management.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message, Renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    StepResult,
    Trajectory,
)

from tttd.erdos.executor import extract_code_block, run_erdos_eval_async
from tttd.erdos.prompt import SYSTEM_PROMPT, get_erdos_prompt
from tttd.sampler import ErdosState

logger = logging.getLogger(__name__)


class ErdosEnv(Env):
    """Environment for a single Erdős optimization episode.

    Each episode:
    1. Presents the Erdős prompt (with initial state from sampler)
    2. Receives generated Python code as the action
    3. Evaluates the code to get h_values and c5_bound
    4. Returns reward = 1 / c5_bound (higher reward for lower bound)
    """

    def __init__(
        self,
        renderer: Renderer,
        timeout: int = 60,
        parent_state: ErdosState | None = None,
    ):
        self._renderer = renderer
        self._timeout = timeout
        self._parent_state = parent_state
        # Will be set after step() if successful
        self._child_state: ErdosState | None = None

    async def initial_observation(self):
        """Build the initial prompt for the LLM."""
        h_values = None
        c5_bound = None

        if self._parent_state and self._parent_state.h_values:
            h_values = np.array(self._parent_state.h_values)
            c5_bound = self._parent_state.c5_bound

        prompt_text = get_erdos_prompt(
            initial_h_values=h_values,
            c5_bound=c5_bound,
        )

        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        observation = self._renderer.build_generation_prompt(messages)
        stop_condition: StopCondition = self._renderer.get_stop_sequences()

        return observation, stop_condition

    async def step(self, action: Action) -> StepResult:
        """Evaluate the generated code."""
        metrics: Metrics = {}

        # Parse the model response
        message, _parse_ok = self._renderer.parse_response(action)
        text = get_text_content(message)

        # Extract code block
        code = extract_code_block(text)

        if code is None or "def run" not in code:
            metrics["format"] = 0
            metrics["correct"] = 0
            metrics["c5_bound"] = float("inf")
            metrics["score"] = 0.0
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=self._renderer.build_generation_prompt([]),
                next_stop_condition=self._renderer.get_stop_sequences(),
                metrics=metrics,
                logs={"response_text": text[:200] if text else "(empty)"},
            )

        metrics["format"] = 1

        # Evaluate the code
        result = await run_erdos_eval_async(code, self._timeout)

        if result["success"]:
            metrics["correct"] = 1
            metrics["c5_bound"] = result["c5_bound"]
            metrics["score"] = result["score"]
            reward = result["score"]  # 1 / c5_bound

            # Create child state for sampler
            self._child_state = ErdosState(
                timestep=-1,  # Will be set by sampler
                value=-result["c5_bound"],  # Higher = better
                c5_bound=result["c5_bound"],
                h_values=result["h_values"].tolist() if result["h_values"] is not None else None,
                code=code,
            )
        else:
            metrics["correct"] = 0
            metrics["c5_bound"] = float("inf")
            metrics["score"] = 0.0
            reward = 0.0
            self._child_state = None

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self._renderer.build_generation_prompt([]),
            next_stop_condition=self._renderer.get_stop_sequences(),
            metrics=metrics,
            logs={"eval_msg": result["msg"][:300]},
        )

    def get_child_state(self) -> ErdosState | None:
        """Get the child state created during step()."""
        return self._child_state

    def get_parent_state(self) -> ErdosState | None:
        """Get the parent state this env was initialized with."""
        return self._parent_state


class ErdosEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of ErdosEnv instances for GRPO-style training.

    Each env in the group starts from the same parent state (sampled from PUCT).
    After rollouts, child states are collected and sent back to the sampler.
    """

    def __init__(
        self,
        renderer: Renderer,
        group_size: int = 4,
        timeout: int = 60,
        parent_state: ErdosState | None = None,
    ):
        self._renderer = renderer
        self._group_size = group_size
        self._timeout = timeout
        self._parent_state = parent_state
        self._envs: list[ErdosEnv] = []

    async def make_envs(self) -> Sequence[Env]:
        self._envs = [
            ErdosEnv(
                renderer=self._renderer,
                timeout=self._timeout,
                parent_state=self._parent_state,
            )
            for _ in range(self._group_size)
        ]
        return self._envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        # All reward comes from step(); no additional group-level reward
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["erdos"]

    def get_parent_state(self) -> ErdosState | None:
        """Get the parent state for this group."""
        return self._parent_state

    def get_child_states(self) -> list[ErdosState | None]:
        """Get child states from all envs after rollouts."""
        return [env.get_child_state() for env in self._envs]
