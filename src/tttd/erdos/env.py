"""Erdős reinforcement learning environment.

Exact port from TTT-Discover's tinker_cookbook/recipes/ttt/env_erdos.py
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

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
from tttd.erdos.prompt import SYSTEM_PROMPT
from tttd.sampler import ErdosState

logger = logging.getLogger(__name__)

# Default config matching original
DEFAULT_BUDGET_S = 1000
DEFAULT_NUM_CPUS = 2


def _truncate_code(code: str, max_lines: int = 200) -> str:
    """Keep top and bottom lines, truncate middle."""
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code
    half = max_lines // 2
    return '\n'.join(lines[:half]) + "\n# ...(middle truncated)...\n" + '\n'.join(lines[-half:])


def get_improvement_prompt(
    state: ErdosState,
    budget_s: int = DEFAULT_BUDGET_S,
    num_cpus: int = DEFAULT_NUM_CPUS,
    hide_code: bool = False,
) -> str:
    """Build contextual improvement prompt with before/after values and stdout.

    Exact port from TTT-Discover's env_erdos.py:_get_improvement_prompt
    """
    has_code = state.code and state.code.strip() and not hide_code

    # Value context: show before/after if we have parent values
    # state.value is -c5_bound (higher=better for RL), so negate to get c5_bound (lower=better)
    if state.parent_values and state.value is not None:
        before_bound = -state.parent_values[0]
        after_bound = -state.value
        value_ctx = f"\nHere are the C₅ bounds before and after running the code above (lower is better): {before_bound:.6f} -> {after_bound:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.value is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {-state.value:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.c5_bound is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {state.c5_bound:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    else:
        value_ctx = "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."

    # Add stdout from previous execution if available
    if state.observation and state.observation.strip():
        stdout = state.observation.strip()
        if len(stdout) > 500:
            stdout = "...(truncated)\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    prompt = SYSTEM_PROMPT
    prompt = prompt.replace("<<<BUDGET_S>>>", str(budget_s))
    prompt = prompt.replace("<<<CPUS>>>", str(num_cpus))

    # h_values section - provide access to initial construction
    h_values_section = ""
    if state.h_values is not None and len(state.h_values) > 0:
        h_values_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.h_values)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

    # Handle code section
    if has_code:
        clean_code = state.code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[len("```python"):].strip()
        if clean_code.startswith("```"):
            clean_code = clean_code[3:].strip()
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3].strip()
        code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing constructions.{value_ctx}

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded.
"""
    else:
        code_section = f"""
{value_ctx}

Write code to optimize this construction.
"""

    return f"""{prompt}
{h_values_section}{code_section}"""


class ErdosEnv(Env):
    """Environment for a single Erdős optimization episode.

    Exact port from TTT-Discover's ErdosMinOverlapEnv.
    """

    def __init__(
        self,
        renderer: Renderer,
        timeout: int = 60,
        parent_state: ErdosState | None = None,
        budget_s: int = DEFAULT_BUDGET_S,
        num_cpus: int = DEFAULT_NUM_CPUS,
        hide_code: bool = False,
    ):
        self._renderer = renderer
        self._timeout = timeout
        self._parent_state = parent_state
        self._budget_s = budget_s
        self._num_cpus = num_cpus
        self._hide_code = hide_code
        self._child_state: ErdosState | None = None

    async def initial_observation(self):
        """Build the initial prompt for the LLM."""
        if self._parent_state is None:
            # Create a default initial state
            from tttd.sampler import create_initial_erdos_state
            self._parent_state = create_initial_erdos_state()

        prompt_text = get_improvement_prompt(
            self._parent_state,
            budget_s=self._budget_s,
            num_cpus=self._num_cpus,
            hide_code=self._hide_code,
        )

        messages: list[Message] = [
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
            metrics["correctness"] = 0.0
            metrics["c5_bound"] = None
            metrics["performance"] = None
            metrics["score"] = 0.0
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=self._renderer.build_generation_prompt([]),
                next_stop_condition=self._renderer.get_stop_sequences(),
                metrics=metrics,
                logs={"response_text": text[:200] if text else "(empty)", "msg": "No valid code block"},
            )

        metrics["format"] = 1

        # Evaluate the code - pass parent state for initial_h_values injection
        result = await run_erdos_eval_async(
            code,
            self._timeout,
            initial_h_values=self._parent_state.h_values if self._parent_state else None,
        )

        if result["success"]:
            metrics["correct"] = 1
            metrics["correctness"] = 1.0
            metrics["c5_bound"] = result["c5_bound"]
            # performance = -c5_bound (higher = better, matching original)
            performance = -result["c5_bound"]
            metrics["performance"] = performance
            metrics["score"] = result["score"]

            # Reward = 1/c5_bound for entropic advantage (matching original)
            reward = result["score"]

            # Create child state for sampler
            self._child_state = ErdosState(
                timestep=-1,  # Will be set by dataset
                value=performance,  # -c5_bound
                c5_bound=result["c5_bound"],
                h_values=result["h_values"].tolist() if result["h_values"] is not None else None,
                code=code,
                observation=result.get("stdout", ""),
            )
        else:
            metrics["correct"] = 0
            metrics["correctness"] = 0.0
            metrics["c5_bound"] = None
            metrics["performance"] = None
            metrics["score"] = 0.0
            reward = 0.0
            self._child_state = None

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self._renderer.build_generation_prompt([]),
            next_stop_condition=self._renderer.get_stop_sequences(),
            metrics=metrics,
            logs={"msg": result["msg"][:300]},
        )

    def get_child_state(self) -> ErdosState | None:
        """Get the child state created during step()."""
        return self._child_state

    def get_parent_state(self) -> ErdosState | None:
        """Get the parent state this env was initialized with."""
        return self._parent_state


class ErdosEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of ErdosEnv instances for GRPO-style training."""

    def __init__(
        self,
        renderer: Renderer,
        group_size: int = 4,
        timeout: int = 60,
        parent_state: ErdosState | None = None,
        budget_s: int = DEFAULT_BUDGET_S,
        num_cpus: int = DEFAULT_NUM_CPUS,
        hide_code: bool = False,
    ):
        self._renderer = renderer
        self._group_size = group_size
        self._timeout = timeout
        self._parent_state = parent_state
        self._budget_s = budget_s
        self._num_cpus = num_cpus
        self._hide_code = hide_code
        self._envs: list[ErdosEnv] = []

    async def make_envs(self) -> Sequence[Env]:
        self._envs = [
            ErdosEnv(
                renderer=self._renderer,
                timeout=self._timeout,
                parent_state=self._parent_state,
                budget_s=self._budget_s,
                num_cpus=self._num_cpus,
                hide_code=self._hide_code,
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
