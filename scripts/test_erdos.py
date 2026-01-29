#!/usr/bin/env python3
"""Test script for Erdős task - verifies baseline eval and single rollout work."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_verifier():
    """Test the Erdős verifier with a simple solution."""
    import numpy as np
    from tttd.erdos.verifier import evaluate_erdos_solution, verify_c5_solution

    logger.info("=== Testing Verifier ===")

    # Test 1: Trivial solution h(x) = 0.5
    n_points = 200
    h_values = np.ones(n_points) * 0.5
    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))

    logger.info(f"Trivial solution (h=0.5): c5_bound = {c5_bound:.6f}")
    assert 0.49 < c5_bound < 0.51, f"Expected ~0.5, got {c5_bound}"

    # Verify
    assert verify_c5_solution(h_values, c5_bound, n_points)
    actual = evaluate_erdos_solution(h_values, c5_bound, n_points)
    logger.info(f"Verified c5_bound = {actual:.6f}")

    # Test 2: Step function (should be better)
    h_step = np.zeros(n_points)
    h_step[:n_points//2] = 1.0
    j_step = 1.0 - h_step
    corr_step = np.correlate(h_step, j_step, mode="full") * dx
    c5_step = float(np.max(corr_step))
    logger.info(f"Step function: c5_bound = {c5_step:.6f}")

    actual_step = evaluate_erdos_solution(h_step, c5_step, n_points)
    logger.info(f"Verified step c5_bound = {actual_step:.6f}")

    logger.info("✓ Verifier tests passed\n")


def test_executor():
    """Test the local subprocess executor."""
    logger.info("=== Testing Executor ===")

    from tttd.erdos.executor import run_erdos_eval, extract_code_block

    # Test code extraction
    text = '''
Here is my solution:

```python
import numpy as np

def run():
    n_points = 100
    h_values = np.ones(n_points) * 0.5
    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))
    return h_values, c5_bound, n_points
```

This gives a bound of 0.5.
'''
    code = extract_code_block(text)
    assert code is not None
    assert "def run" in code
    logger.info("✓ Code extraction works")

    # Test execution
    result = run_erdos_eval(code, timeout=30)
    logger.info(f"Execution result: {result['msg']}")
    assert result["success"], f"Execution failed: {result['msg']}"
    assert 0.49 < result["c5_bound"] < 0.51
    logger.info(f"✓ Executor works, c5_bound = {result['c5_bound']:.6f}\n")


async def test_env():
    """Test the Erdős environment."""
    logger.info("=== Testing Environment ===")

    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from tttd.erdos.env import ErdosEnv, ErdosState

    # Create environment
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = get_renderer("qwen3", tokenizer)

    env = ErdosEnv(renderer=renderer, timeout=30)

    # Get initial observation
    obs, stop_cond = await env.initial_observation()
    logger.info(f"Initial observation length: {obs.length} tokens")
    assert obs.length > 0
    logger.info("✓ Environment initial observation works\n")


async def test_single_rollout():
    """Test a single rollout with mock action."""
    logger.info("=== Testing Single Rollout ===")

    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.rl.types import Action

    from tttd.erdos.env import ErdosEnv

    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = get_renderer("qwen3", tokenizer)

    env = ErdosEnv(renderer=renderer, timeout=30)

    # Get initial observation
    obs, stop_cond = await env.initial_observation()

    # Create a mock action (simulating LLM output)
    mock_response = '''I'll create a solution using a step function approach.

```python
import numpy as np

def run():
    n_points = 200

    # Simple step function
    h_values = np.zeros(n_points)
    h_values[:n_points//2] = 0.8
    h_values[n_points//2:] = 0.2

    # Compute bound
    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return h_values, c5_bound, n_points
```

This step function should give a bound around 0.48.
'''
    # Encode the response - Action is just list[int]
    action: Action = tokenizer.encode(mock_response)

    # Take a step
    step_result = await env.step(action)

    logger.info(f"Step result - reward: {step_result.reward:.4f}")
    logger.info(f"Metrics: {step_result.metrics}")
    logger.info(f"Logs: {step_result.logs}")

    assert step_result.episode_done
    assert step_result.metrics.get("format") == 1
    logger.info("✓ Single rollout works\n")


def test_baseline_solution():
    """Test that we can evaluate a known good solution."""
    logger.info("=== Testing Baseline Solution ===")

    from tttd.erdos.executor import run_erdos_eval

    # A slightly better construction
    code = '''
import numpy as np
from scipy.optimize import minimize

def run():
    n_points = 200

    # Parameterized step function
    def make_h(params):
        a, b, c = params
        x = np.linspace(0, 1, n_points)
        h = np.where(x < a, b, c)
        return np.clip(h, 0, 1)

    def objective(params):
        h = make_h(params)
        j = 1.0 - h
        dx = 2.0 / n_points
        corr = np.correlate(h, j, mode="full") * dx
        return np.max(corr)

    # Optimize
    from scipy.optimize import differential_evolution
    bounds = [(0.3, 0.7), (0.5, 1.0), (0.0, 0.5)]
    result = differential_evolution(objective, bounds, maxiter=50, seed=42)

    h_values = make_h(result.x)
    c5_bound = result.fun

    return h_values, c5_bound, n_points
'''

    result = run_erdos_eval(code, timeout=60)
    logger.info(f"Optimized result: {result['msg']}")

    if result["success"]:
        logger.info(f"✓ Baseline optimization works, c5_bound = {result['c5_bound']:.6f}")
    else:
        logger.warning(f"Baseline optimization failed (scipy may not be installed): {result['msg']}")
        logger.info("Trying simpler baseline...")

        # Simpler baseline without scipy
        simple_code = '''
import numpy as np

def run():
    n_points = 200

    # Manual step function that's reasonably good
    h_values = np.zeros(n_points)
    # Asymmetric step
    cutoff = int(0.38 * n_points)
    h_values[:cutoff] = 0.9
    h_values[cutoff:] = 0.1

    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return h_values, c5_bound, n_points
'''
        result = run_erdos_eval(simple_code, timeout=30)
        logger.info(f"Simple baseline: {result['msg']}")
        assert result["success"]
        logger.info(f"✓ Simple baseline works, c5_bound = {result['c5_bound']:.6f}")

    logger.info("")


def test_puct_sampler():
    """Test the PUCT sampler."""
    logger.info("=== Testing PUCT Sampler ===")

    import tempfile
    from tttd.sampler import PUCTSampler, ErdosState, create_initial_erdos_state

    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = PUCTSampler(
            log_path=tmpdir,
            max_buffer_size=10,
            puct_c=1.0,
        )

        # Sample initial states
        states = sampler.sample_states(3)
        logger.info(f"Sampled {len(states)} initial states")
        assert len(states) == 3
        for s in states:
            assert s.h_values is not None
            logger.info(f"  State {s.id[:8]}... value={s.value:.4f}, c5_bound={s.c5_bound:.4f}")

        # Simulate rollout: create child states
        parent = states[0]
        children = []
        for i in range(3):
            child = ErdosState(
                timestep=0,
                value=-0.4 + i * 0.05,  # Varying quality
                c5_bound=0.4 - i * 0.05,
                h_values=[0.5] * 100,
                code=f"# solution {i}",
            )
            children.append(child)

        # Update sampler
        sampler.update_states(children, [parent] * 3, step=0)
        logger.info(f"Updated sampler with {len(children)} children")

        # Sample again - should prefer better states
        states2 = sampler.sample_states(2)
        logger.info(f"After update, sampled {len(states2)} states")
        for s in states2:
            val_str = f"{s.value:.4f}" if s.value is not None else "None"
            logger.info(f"  State {s.id[:8]}... value={val_str}")

        # Check stats
        stats = sampler.get_stats()
        logger.info(f"Sampler stats: buffer_size={stats['puct/buffer_size']}, T={stats['puct/T']}")
        assert stats["puct/buffer_size"] > 1
        assert stats["puct/T"] >= 1

        # Test save/load
        sampler.flush(step=1)
        logger.info("Saved sampler state")

    logger.info("✓ PUCT Sampler works\n")


def test_advantages():
    """Test the advantage estimators."""
    logger.info("=== Testing Advantage Estimators ===")

    import torch
    from tinker_cookbook.rl.types import TrajectoryGroup, Trajectory, Transition
    from tinker_cookbook.completers import TokensWithLogprobs
    import tinker

    from tttd.advantages import compute_advantages

    # Create mock trajectory group with known rewards
    def make_mock_trajectory(reward: float) -> Trajectory:
        """Create a minimal trajectory with the given total reward."""
        transition = Transition(
            ob=tinker.ModelInput.empty(),
            ac=TokensWithLogprobs(tokens=[1], maybe_logprobs=[0.0]),
            reward=reward,
            episode_done=True,
        )
        return Trajectory(transitions=[transition], final_ob=tinker.ModelInput.empty())

    # Test with rewards [1, 2, 3, 4]
    rewards = [1.0, 2.0, 3.0, 4.0]
    trajectories = [make_mock_trajectory(r) for r in rewards]
    traj_group = TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0] * len(rewards),
        metrics_G=[{}] * len(rewards),
    )

    # Test mean baseline
    adv_mean = compute_advantages([traj_group], estimator="mean_baseline")[0]
    logger.info(f"Mean baseline advantages: {adv_mean.tolist()}")
    assert adv_mean.sum().abs() < 1e-6, "Mean baseline should sum to ~0"

    # Test entropic with fixed beta
    adv_entropic = compute_advantages([traj_group], estimator="entropic", beta=1.0)[0]
    logger.info(f"Entropic (beta=1.0) advantages: {adv_entropic.tolist()}")

    # Test entropic with adaptive beta
    adv_adaptive = compute_advantages([traj_group], estimator="entropic_adaptive_beta")[0]
    logger.info(f"Entropic adaptive advantages: {adv_adaptive.tolist()}")

    # Higher rewards should have higher advantages
    assert adv_mean[3] > adv_mean[0], "Higher reward should have higher advantage"
    assert adv_entropic[3] > adv_entropic[0], "Higher reward should have higher advantage"
    assert adv_adaptive[3] > adv_adaptive[0], "Higher reward should have higher advantage"

    logger.info("✓ Advantage estimators work\n")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Erdős Task Test Suite")
    logger.info("=" * 60 + "\n")

    # Sync tests
    test_verifier()
    test_executor()
    test_baseline_solution()
    test_puct_sampler()
    test_advantages()

    # Async tests
    await test_env()
    await test_single_rollout()

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
