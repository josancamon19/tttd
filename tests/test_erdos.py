#!/usr/bin/env python3
"""Test script for Erdős task - verifies baseline eval and single rollout work."""

import logging

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.anyio


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

    # Verify - now returns computed value or raises
    computed = verify_c5_solution(h_values, c5_bound, n_points)
    assert abs(computed - c5_bound) < 1e-4
    actual = evaluate_erdos_solution(h_values, c5_bound, n_points)
    logger.info(f"Verified c5_bound = {actual:.6f}")

    # Test 2: Step function (should be better)
    h_step = np.zeros(n_points)
    h_step[: n_points // 2] = 1.0
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
    text = """
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
"""
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


@pytest.mark.anyio
async def test_env():
    """Test the Erdős environment."""
    logger.info("=== Testing Environment ===")

    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from tttd.erdos.env import ErdosEnv
    from tttd.sampler import ErdosState

    # Create environment
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = get_renderer("qwen3", tokenizer)

    env = ErdosEnv(renderer=renderer, timeout=30)

    # Get initial observation
    obs, stop_cond = await env.initial_observation()
    logger.info(f"Initial observation length: {obs.length} tokens")
    assert obs.length > 0
    logger.info("✓ Environment initial observation works\n")


@pytest.mark.anyio
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
    mock_response = """I'll create a solution using a step function approach.

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
"""
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
    code = """
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
"""

    result = run_erdos_eval(code, timeout=60)
    logger.info(f"Optimized result: {result['msg']}")

    if result["success"]:
        logger.info(
            f"✓ Baseline optimization works, c5_bound = {result['c5_bound']:.6f}"
        )
    else:
        logger.warning(
            f"Baseline optimization failed (scipy may not be installed): {result['msg']}"
        )
        logger.info("Trying simpler baseline...")

        # Simpler baseline without scipy
        # Must satisfy sum(h) = n_points / 2
        simple_code = """
import numpy as np

def run():
    n_points = 200

    # Simple uniform h = 0.5 satisfies constraint sum(h) = 100
    h_values = np.ones(n_points) * 0.5

    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return h_values, c5_bound, n_points
"""
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
            logger.info(
                f"  State {s.id[:8]}... value={s.value:.4f}, c5_bound={s.c5_bound:.4f}"
            )

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
        logger.info(
            f"Sampler stats: buffer_size={stats['puct/buffer_size']}, T={stats['puct/T']}"
        )
        assert stats["puct/buffer_size"] > 1
        assert stats["puct/T"] >= 1

        # Test save/load
        sampler.flush(step=1)
        logger.info("Saved sampler state")

    logger.info("✓ PUCT Sampler works\n")


def test_puct_advanced():
    """Test advanced PUCT sampler features."""
    logger.info("=== Testing PUCT Advanced Features ===")

    import tempfile
    from tttd.sampler import PUCTSampler, ErdosState, create_initial_erdos_state

    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = PUCTSampler(
            log_path=tmpdir,
            max_buffer_size=20,
            puct_c=1.0,
        )

        # Test 1: record_failed_rollout
        logger.info("Testing record_failed_rollout...")
        initial_T = sampler._T
        parent = sampler.sample_states(1)[0]
        initial_n = sampler._n.get(parent.id, 0)

        sampler.record_failed_rollout(parent)

        assert sampler._T == initial_T + 1, "T should increment on failed rollout"
        assert sampler._n.get(parent.id, 0) == initial_n + 1, (
            "n should increment on failed rollout"
        )
        logger.info("  ✓ record_failed_rollout increments T and n")

        # Test 2: Ancestor visit count propagation
        logger.info("Testing ancestor visit count propagation...")
        # Create a parent with existing ancestors
        grandparent = create_initial_erdos_state()
        parent_with_ancestry = ErdosState(
            timestep=0,
            value=-0.45,
            c5_bound=0.45,
            h_values=[0.5] * 100,
            code="# parent",
            parents=[{"id": grandparent.id, "timestep": -1}],
        )
        sampler._states.append(grandparent)
        sampler._states.append(parent_with_ancestry)

        # Create child
        child = ErdosState(
            timestep=1,
            value=-0.40,
            c5_bound=0.40,
            h_values=[0.6] * 100,
            code="# child",
        )

        gp_n_before = sampler._n.get(grandparent.id, 0)
        p_n_before = sampler._n.get(parent_with_ancestry.id, 0)

        sampler.update_states([child], [parent_with_ancestry], step=1)

        gp_n_after = sampler._n.get(grandparent.id, 0)
        p_n_after = sampler._n.get(parent_with_ancestry.id, 0)

        assert p_n_after == p_n_before + 1, "Parent n should increment"
        assert gp_n_after == gp_n_before + 1, "Grandparent n should also increment"
        logger.info("  ✓ Ancestor visit counts propagate correctly")

        # Test 3: Initial state refreshing
        logger.info("Testing initial state refreshing...")
        sampler2 = PUCTSampler(log_path=tmpdir + "/test2", max_buffer_size=10)
        initial = sampler2._initial_states[0]
        original_h = initial.h_values.copy() if initial.h_values else None

        # Sample the initial state - should refresh
        sampled = sampler2.sample_states(1)[0]
        if sampled.id == initial.id:
            # The h_values should be different after refresh
            assert sampled.h_values != original_h or len(sampler2._states) == 1, (
                "Initial state should be refreshed with new h_values"
            )
        logger.info("  ✓ Initial states refresh when re-sampled")

        # Test 4: Full lineage blocking
        logger.info("Testing full lineage blocking...")
        sampler3 = PUCTSampler(log_path=tmpdir + "/test3", max_buffer_size=20)

        # Create a tree: root -> child1 -> grandchild
        root = sampler3._states[0]
        child1 = ErdosState(
            timestep=0,
            value=-0.45,
            c5_bound=0.45,
            h_values=[0.55] * 100,
            code="# c1",
            parents=[{"id": root.id, "timestep": -1}],
        )
        grandchild = ErdosState(
            timestep=1,
            value=-0.40,
            c5_bound=0.40,
            h_values=[0.60] * 100,
            code="# gc",
            parents=[{"id": child1.id, "timestep": 0}, {"id": root.id, "timestep": -1}],
        )
        sampler3._states.extend([child1, grandchild])

        # When sampling 2 states, we should not get both root and grandchild
        # (they're in the same lineage)
        children_map = sampler3._build_children_map()
        lineage = sampler3._get_full_lineage(root, children_map)

        assert child1.id in lineage, "Child should be in root's lineage"
        assert grandchild.id in lineage, "Grandchild should be in root's lineage"
        logger.info("  ✓ Full lineage blocking works correctly")

    logger.info("✓ PUCT Advanced Features work\n")


def test_advantages():
    """Test the advantage estimators."""
    logger.info("=== Testing Advantage Estimators ===")

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
    adv_adaptive = compute_advantages([traj_group], estimator="entropic_adaptive_beta")[
        0
    ]
    logger.info(f"Entropic adaptive advantages: {adv_adaptive.tolist()}")

    # Higher rewards should have higher advantages
    assert adv_mean[3] > adv_mean[0], "Higher reward should have higher advantage"
    assert adv_entropic[3] > adv_entropic[0], (
        "Higher reward should have higher advantage"
    )
    assert adv_adaptive[3] > adv_adaptive[0], (
        "Higher reward should have higher advantage"
    )

    logger.info("✓ Advantage estimators work\n")


