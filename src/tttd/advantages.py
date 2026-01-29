"""Alternative advantage estimators including entropic advantages from TTT-Discover.

The tinker_cookbook only provides mean baseline advantages. This module adds
the entropic advantage estimator used in TTT-Discover.
"""

from __future__ import annotations

import math
from typing import List, Literal

import torch
from tinker_cookbook.rl.types import TrajectoryGroup


AdvantagEstimator = Literal["mean_baseline", "entropic", "entropic_adaptive_beta"]


def compute_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
    estimator: AdvantagEstimator = "mean_baseline",
    beta: float = 1.0,
) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups.

    Args:
        trajectory_groups_P: List of trajectory groups
        estimator: Which advantage estimator to use:
            - "mean_baseline": Simple mean subtraction (default, matches tinker_cookbook)
            - "entropic": Entropic advantages with fixed beta
            - "entropic_adaptive_beta": Entropic with adaptive beta (TTT-Discover default)
        beta: Temperature parameter for entropic estimators (only used if estimator="entropic")

    Returns:
        List of advantage tensors, one per trajectory group
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float32)

        if estimator == "mean_baseline":
            # Simple mean baseline (same as tinker_cookbook)
            advantages_G = rewards_G - rewards_G.mean()

        elif estimator == "entropic":
            # Entropic advantages with fixed beta
            advantages_G = _compute_entropic_advantages(rewards_G, beta)

        elif estimator == "entropic_adaptive_beta":
            # Entropic advantages with adaptive beta
            advantages_G = _compute_entropic_adaptive_beta(rewards_G)

        else:
            raise ValueError(f"Unknown advantage estimator: {estimator}")

        advantages_P.append(advantages_G)

    return advantages_P


def _compute_entropic_advantages(rewards_G: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute entropic advantages with fixed beta.

    Entropic advantages reweight samples based on exponential of reward,
    using leave-one-out normalization for stability.

    Args:
        rewards_G: Tensor of rewards for each trajectory in the group
        beta: Temperature parameter (higher = more exploitation)

    Returns:
        Tensor of advantages
    """
    k = rewards_G.shape[0]
    eps = 1e-12

    # Stabilize by subtracting max
    s_safe = rewards_G - rewards_G.max()
    e = torch.exp(beta * s_safe)

    if k == 1:
        Z = e
    else:
        # Leave-one-out normalization: Z_i = (sum(e) - e_i) / (k - 1)
        Z = (e.sum() - e) / (k - 1)

    # Importance weights
    w = e / (Z + eps)

    # Advantages are weights minus 1 (centered)
    return w - 1.0


def _compute_entropic_adaptive_beta(rewards_G: torch.Tensor) -> torch.Tensor:
    """Compute entropic advantages with adaptive beta.

    Finds beta such that KL(q_beta || uniform) = delta, where q_beta is the
    softmax distribution over rewards with temperature 1/beta.

    This is the default used in TTT-Discover.

    Args:
        rewards_G: Tensor of rewards for each trajectory in the group

    Returns:
        Tensor of advantages
    """
    delta = math.log(2)  # Target KL divergence
    beta_max = 1e6
    iters = 60
    eps = 1e-12

    r = rewards_G.float()
    k = r.shape[0]

    if k < 2:
        return torch.zeros_like(r)

    log_k = math.log(k)

    def kl_hat(beta_scalar: float) -> float:
        """Compute KL(q_beta || uniform) for a given beta."""
        b = torch.tensor(beta_scalar, dtype=r.dtype)
        logits = b * (r - r.max())  # Stabilized
        logq = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(logq)
        kl = (q * (logq + log_k)).sum()
        return float(kl.item())

    # Binary search to find beta
    lo, hi = 0.0, 1.0

    # First, expand upper bound if needed
    if kl_hat(hi) < delta:
        while hi < beta_max and kl_hat(hi) < delta:
            hi *= 2.0
        if kl_hat(hi) < delta:
            beta = hi  # Best effort
        else:
            # Binary search
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if kl_hat(mid) < delta:
                    lo = mid
                else:
                    hi = mid
            beta = hi
    else:
        # Binary search from the start
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < delta:
                lo = mid
            else:
                hi = mid
        beta = hi

    # Compute advantages with the found beta
    e = torch.exp(beta * (r - r.max()))

    if k == 1:
        Z = e
    else:
        Z = (e.sum() - e) / (k - 1)

    w = e / (Z + eps)
    return w - 1.0


# For convenience, expose the tinker_cookbook version too
def compute_mean_baseline_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[torch.Tensor]:
    """Compute mean baseline advantages (same as tinker_cookbook)."""
    return compute_advantages(trajectory_groups_P, estimator="mean_baseline")
