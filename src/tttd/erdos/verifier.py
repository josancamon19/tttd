"""Erdős Minimum Overlap verifier.

Ported from TTT-Discover's tasks/erdos_min_overlap/verifier.py.
Evaluates solutions to the Erdős minimum overlap problem.
"""

import numpy as np


def verify_c5_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> bool:
    """Verify that the solution is valid.

    Args:
        h_values: Array of h values in [0, 1], length n_points
        c5_bound: The claimed C5 bound
        n_points: Number of discretization points

    Returns:
        True if valid, False otherwise
    """
    if len(h_values) != n_points:
        return False
    if not np.all((h_values >= 0) & (h_values <= 1)):
        return False
    if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
        return False
    return True


def evaluate_erdos_solution(
    h_values: np.ndarray,
    c5_bound: float,
    n_points: int,
) -> float:
    """Evaluate an Erdős minimum overlap solution.

    The Erdős minimum overlap problem asks: given a sequence of length n,
    what is the minimum overlap when the sequence is shifted?

    We discretize the problem: h(x) for x in [0,1] represents the "height"
    function. The overlap is computed via correlation of h and (1-h).

    Args:
        h_values: Array of h values in [0, 1], shape (n_points,)
        c5_bound: The claimed C5 bound (overlap)
        n_points: Number of discretization points

    Returns:
        The actual C5 bound (overlap value). Lower is better.
    """
    if not verify_c5_solution(h_values, c5_bound, n_points):
        return float('inf')

    h_values = np.asarray(h_values, dtype=np.float64)

    # Compute j_values = 1 - h_values
    j_values = 1.0 - h_values

    # Discretization step
    dx = 2.0 / n_points

    # Compute the correlation (overlap) via convolution
    # The overlap at shift t is integral of h(x) * j(x + t) dx
    correlation = np.correlate(h_values, j_values, mode="full") * dx

    # The C5 bound is the maximum overlap
    c5_actual = float(np.max(correlation))

    return c5_actual


# Known good solutions for reference
KNOWN_BEST_BOUND = 0.380926  # Approximate best known value
