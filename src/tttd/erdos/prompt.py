"""Erdős Minimum Overlap prompt template.

Ported from TTT-Discover's tasks/erdos_min_overlap/prompt.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SYSTEM_PROMPT = """\
You are an expert mathematician and algorithm designer. Your task is to find optimal solutions to the Erdős minimum overlap problem.

You will write Python code that defines a function called `run()` which returns a tuple of (h_values, c5_bound, n_points).

Important rules:
- Your code MUST define a function called `run()` that takes no arguments
- The function MUST return a tuple: (h_values, c5_bound, n_points)
- h_values should be a numpy array of floats in [0, 1]
- c5_bound should be your computed overlap bound (a positive float)
- n_points should be the length of h_values
- Lower c5_bound is better
"""

ERDOS_PROMPT = """\
# Task: Minimize the Erdős Overlap Bound

## Problem Description

The Erdős minimum overlap problem asks: given a measurable function h: [0,1] → [0,1],
what is the minimum possible value of the maximum overlap?

The overlap at shift t is defined as:
```
overlap(t) = ∫₀¹ h(x) · (1 - h(x + t mod 1)) dx
```

The C5 bound is the maximum overlap over all shifts:
```
C5 = max_t overlap(t)
```

Your goal is to find h(x) that minimizes C5.

## Mathematical Background

- The trivial solution h(x) = 0.5 gives C5 = 0.5
- The best known bound is approximately 0.3809
- Good solutions often have interesting structure (steps, smooth transitions, etc.)

## Your Task

Write a `run()` function that:
1. Constructs an h_values array (discretized h(x) on n_points points)
2. Computes the resulting C5 bound
3. Returns (h_values, c5_bound, n_points)

```python
import numpy as np

def run():
    n_points = 200  # Number of discretization points

    # Your h(x) construction here
    h_values = np.ones(n_points) * 0.5  # Example: constant 0.5

    # Compute overlap bound
    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = np.max(correlation)

    return h_values, c5_bound, n_points
```

## Tips for Better Solutions

1. **Step functions**: Try piecewise constant functions with optimized breakpoints
2. **Smooth functions**: Try trigonometric or polynomial constructions
3. **Optimization**: Use scipy.optimize to tune parameters
4. **Known constructions**: The best solutions use carefully designed step functions

## Scoring

Your reward is proportional to 1/c5_bound. Lower c5_bound = higher reward.

- c5_bound = 0.50 → reward ≈ 2.0
- c5_bound = 0.40 → reward ≈ 2.5
- c5_bound = 0.381 → reward ≈ 2.62 (near optimal)

<<<INITIAL_STATE>>>

Write your complete solution in a single Python code block (```python ... ```).
"""

_INITIAL_STATE_TEMPLATE = """
## Starting Point

Here is a good starting solution to improve upon:

```python
import numpy as np

initial_h_values = np.array({h_values_repr})
```

Current bound: {c5_bound:.6f}
Try to find a better solution with lower c5_bound.
"""

_NO_INITIAL_STATE = """
## No Starting Point

This is your first attempt. Start with a simple construction and iterate.
"""


def get_erdos_prompt(initial_h_values: "np.ndarray | None" = None, c5_bound: float | None = None) -> str:
    """Build the full Erdős prompt with optional initial state.

    Args:
        initial_h_values: Optional starting h_values array
        c5_bound: The bound achieved by initial_h_values

    Returns:
        The complete prompt string
    """
    import numpy as np

    if initial_h_values is not None and c5_bound is not None:
        # Format array compactly
        h_list = initial_h_values.tolist()
        h_values_repr = repr(h_list)
        initial_state_section = _INITIAL_STATE_TEMPLATE.format(
            h_values_repr=h_values_repr,
            c5_bound=c5_bound,
        )
    else:
        initial_state_section = _NO_INITIAL_STATE

    return ERDOS_PROMPT.replace("<<<INITIAL_STATE>>>", initial_state_section)
