"""PUCT-based state sampler for TTT.

Maintains a tree of historical solutions and uses PUCT to balance
exploration vs exploitation when selecting which state to expand.

Ported from TTT-Discover's tinker_cookbook/recipes/ttt/sampler.py
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Erdős-specific constants
MAX_ERDOS_CONSTRUCTION_LEN = 1000  # Max h_values array length
MIN_ERDOS_CONSTRUCTION_LEN = 50   # Min h_values array length


@dataclass
class ErdosState:
    """State for Erdős problem - holds h_values and metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestep: int = -1  # Training step when this state was created (-1 = initial)
    value: float | None = None  # Higher = better (we use -c5_bound)
    c5_bound: float | None = None  # The actual overlap bound
    h_values: list[float] | None = None  # The solution array
    code: str = ""  # Code that generated this solution
    parent_values: list[float] = field(default_factory=list)  # Ancestor values
    parents: list[dict] = field(default_factory=list)  # Parent refs

    def to_dict(self) -> dict:
        return {
            "type": "ErdosState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "c5_bound": self.c5_bound,
            "h_values": self.h_values,
            "code": self.code,
            "parent_values": self.parent_values,
            "parents": self.parents,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ErdosState":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestep=d.get("timestep", -1),
            value=d.get("value"),
            c5_bound=d.get("c5_bound"),
            h_values=d.get("h_values"),
            code=d.get("code", ""),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
        )


def create_initial_erdos_state() -> ErdosState:
    """Create an initial random Erdős state."""
    rng = np.random.default_rng()
    n_points = rng.integers(100, 300)

    # Start with h = 0.5 + small perturbation
    h_values = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.3, 0.3, n_points)
    perturbation = perturbation - np.mean(perturbation)  # Center
    h_values = np.clip(h_values + perturbation, 0, 1)

    # Compute initial bound
    j_values = 1.0 - h_values
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return ErdosState(
        timestep=-1,
        value=-c5_bound,  # Higher value = better (lower bound)
        c5_bound=c5_bound,
        h_values=h_values.tolist(),
        code="",
    )


class StateSampler(ABC):
    """Abstract base class for state samplers."""

    @abstractmethod
    def sample_states(self, num_states: int) -> list[ErdosState]:
        """Sample states to start rollouts from."""
        pass

    @abstractmethod
    def update_states(
        self,
        states: list[ErdosState],
        parent_states: list[ErdosState],
        step: int | None = None,
    ):
        """Update sampler with new states."""
        pass

    def record_failed_rollout(self, parent: ErdosState):
        """Record a failed rollout (default: no-op, override in subclasses)."""
        pass

    @abstractmethod
    def flush(self, step: int | None = None):
        """Force save to disk."""
        pass


class PUCTSampler(StateSampler):
    """
    PUCT-style sampler with state archive.

    Uses the PUCT formula to balance exploration vs exploitation:

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T/G) / (1 + n[i]/G)

    where:
      Q(i) = best reachable value from state i (or current value if unexplored)
      P(i) = rank-based prior (higher value = higher prior)
      n[i] = visit count for state i
      T = total visits
      G = group size (for normalization)
      scale = max(values) - min(values)
    """

    def __init__(
        self,
        log_path: str,
        max_buffer_size: int = 500,
        puct_c: float = 1.0,
        topk_children: int = 2,
        group_size: int = 1,
    ):
        self.log_path = log_path
        self.max_buffer_size = max_buffer_size
        self.puct_c = puct_c
        self.topk_children = topk_children
        self.group_size = group_size

        self._states: list[ErdosState] = []
        self._initial_states: list[ErdosState] = []
        self._lock = threading.Lock()
        self._current_step = 0

        # PUCT statistics
        self._n: dict[str, int] = {}  # Visit counts
        self._m: dict[str, float] = {}  # Best child value
        self._T: int = 0  # Total visits

        # For logging
        self._last_sampled_states: list[ErdosState] = []
        self._last_scale: float = 1.0

        # Create initial state if empty
        if not self._states:
            initial = create_initial_erdos_state()
            self._initial_states.append(initial)
            self._states.append(initial)

        # Ensure log directory exists
        Path(log_path).mkdir(parents=True, exist_ok=True)

    def _save_path(self, step: int) -> str:
        return os.path.join(self.log_path, f"puct_sampler_step_{step:06d}.json")

    def _save(self, step: int):
        """Save sampler state to disk."""
        data = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "initial_states": [s.to_dict() for s in self._initial_states],
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
        }
        path = self._save_path(step)
        with open(path, "w") as f:
            json.dump(data, f)

    def _load(self, step: int):
        """Load sampler state from disk."""
        path = self._save_path(step)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sampler state not found: {path}")
        with open(path) as f:
            data = json.load(f)
        self._states = [ErdosState.from_dict(s) for s in data.get("states", [])]
        self._initial_states = [ErdosState.from_dict(s) for s in data.get("initial_states", [])]
        self._n = data.get("puct_n", {})
        self._m = data.get("puct_m", {})
        self._T = data.get("puct_T", 0)

    def _compute_scale(self, values: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Compute value scale for PUCT bonus normalization.

        Args:
            values: Array of state values
            mask: Optional boolean mask to filter which values to use
        """
        if len(values) == 0:
            return 1.0
        v = values[mask] if mask is not None else values
        valid = v[~np.isnan(v) & ~np.isinf(v)]
        if len(valid) == 0:
            return 1.0
        return max(float(np.max(valid) - np.min(valid)), 1e-6)

    def _compute_prior(self, values: np.ndarray) -> np.ndarray:
        """Compute rank-based prior probabilities."""
        if len(values) == 0:
            return np.array([])
        n = len(values)
        ranks = np.argsort(np.argsort(-values))  # Higher value = lower rank
        weights = (n - ranks).astype(np.float64)
        return weights / weights.sum()

    def _build_children_map(self) -> dict[str, list[str]]:
        """Build a map from parent_id -> list of child_ids."""
        children_map: dict[str, list[str]] = {}
        for s in self._states:
            if s.parents:
                parent_id = s.parents[0].get("id")
                if parent_id:
                    if parent_id not in children_map:
                        children_map[parent_id] = []
                    children_map[parent_id].append(s.id)
        return children_map

    def _get_full_lineage(self, state: ErdosState, children_map: dict[str, list[str]]) -> set[str]:
        """Get full lineage: ancestors + descendants (to avoid sampling related states)."""
        lineage = {state.id}

        # Add ancestors
        for p in state.parents:
            pid = p.get("id")
            if pid:
                lineage.add(pid)

        # Add descendants (BFS)
        queue = [state.id]
        while queue:
            current = queue.pop(0)
            for child_id in children_map.get(current, []):
                if child_id not in lineage:
                    lineage.add(child_id)
                    queue.append(child_id)

        return lineage

    def _refresh_random_construction(self, state: ErdosState):
        """Re-randomize the h_values for an initial state when re-sampled."""
        rng = np.random.default_rng()
        n_points = len(state.h_values) if state.h_values else rng.integers(100, 300)

        # Generate new random h_values
        h_values = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.3, 0.3, n_points)
        perturbation = perturbation - np.mean(perturbation)
        h_values = np.clip(h_values + perturbation, 0, 1)

        # Recompute bound
        j_values = 1.0 - h_values
        dx = 2.0 / n_points
        correlation = np.correlate(h_values, j_values, mode="full") * dx
        c5_bound = float(np.max(correlation))

        # Update state in place
        state.h_values = h_values.tolist()
        state.c5_bound = c5_bound
        state.value = -c5_bound

    def sample_states(self, num_states: int) -> list[ErdosState]:
        """Sample states using PUCT formula."""
        if not self._states:
            picked = [create_initial_erdos_state() for _ in range(num_states)]
            self._last_sampled_states = picked
            return picked

        # Get values
        values = np.array([
            s.value if s.value is not None else float("-inf")
            for s in self._states
        ])

        # Build mask for non-initial states (for scale computation)
        initial_ids = {s.id for s in self._initial_states}
        non_initial_mask = np.array([s.id not in initial_ids for s in self._states])

        # Compute scale excluding initial states (if any non-initial exist)
        scale = self._compute_scale(
            values,
            non_initial_mask if non_initial_mask.any() else None
        )
        self._last_scale = scale
        P = self._compute_prior(values)
        G = self.group_size
        sqrt_T = np.sqrt(1.0 + self._T / G)

        # Compute PUCT scores
        scores = []
        for i, s in enumerate(self._states):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, values[i])
            Q = m if n > 0 else values[i]
            bonus = self.puct_c * scale * P[i] * sqrt_T / (1.0 + n / G)
            score = Q + bonus
            scores.append((score, values[i], s, i))

        # Sort by score (descending)
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Pick top states, avoiding duplicates in lineage
        # When sampling multiple states, block full lineage (ancestors + descendants)
        if num_states > 1:
            children_map = self._build_children_map()
            picked = []
            blocked_ids: set[str] = set()
            for score, val, state, idx in scores:
                if state.id in blocked_ids:
                    continue
                picked.append(state)
                # Block full lineage to ensure diversity
                blocked_ids.update(self._get_full_lineage(state, children_map))
                if len(picked) >= num_states:
                    break
        else:
            # Single state: just pick the best
            picked = [scores[0][2]] if scores else []

        # Pad with new initial states if needed
        while len(picked) < num_states:
            picked.append(create_initial_erdos_state())

        # Refresh random construction for initial states (so re-sampling explores new starting points)
        for s in picked:
            if s.id in initial_ids:
                self._refresh_random_construction(s)

        self._last_sampled_states = picked
        return picked

    def record_failed_rollout(self, parent: ErdosState):
        """Record a failed rollout - still updates visit counts for exploration tracking."""
        anc_ids = [parent.id]
        if parent.parents:
            anc_ids.extend(str(p.get("id", "")) for p in parent.parents if p.get("id"))

        for aid in anc_ids:
            if aid:
                self._n[aid] = self._n.get(aid, 0) + 1
        self._T += 1

    def update_states(
        self,
        states: list[ErdosState],
        parent_states: list[ErdosState],
        step: int | None = None,
    ):
        """Update sampler with new child states."""
        if not states:
            return

        # Update PUCT statistics - track best child per parent and update ancestors
        parent_map: dict[str, ErdosState] = {p.id: p for p in parent_states}
        parent_max: dict[str, float] = {}

        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue
            pid = parent.id
            parent_max[pid] = max(parent_max.get(pid, float("-inf")), child.value)

        for pid, best_child_value in parent_max.items():
            # Update best reachable value for parent
            self._m[pid] = max(self._m.get(pid, best_child_value), best_child_value)

            # Get ancestor IDs: parent + all its ancestors
            parent = parent_map.get(pid)
            anc_ids = [pid]
            if parent and parent.parents:
                anc_ids.extend(str(p.get("id", "")) for p in parent.parents if p.get("id"))

            # Increment visit counts for parent AND all ancestors
            for aid in anc_ids:
                if aid:
                    self._n[aid] = self._n.get(aid, 0) + 1

            self._T += 1

        # Filter to top-k children per parent
        states, parent_states = self._filter_topk_per_parent(states, parent_states)

        # Deduplicate by h_values
        existing_keys = set()
        for s in self._states:
            if s.h_values:
                existing_keys.add(tuple(s.h_values))

        new_states = []
        for child, parent in zip(states, parent_states):
            if child.value is None or child.h_values is None:
                continue

            # Validate construction length
            if not (MIN_ERDOS_CONSTRUCTION_LEN <= len(child.h_values) <= MAX_ERDOS_CONSTRUCTION_LEN):
                continue

            key = tuple(child.h_values)
            if key in existing_keys:
                continue

            # Set parent info
            child.parent_values = [parent.value] + parent.parent_values if parent.value else []
            child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents
            new_states.append(child)
            existing_keys.add(key)

        with self._lock:
            self._states.extend(new_states)
            self._prune_buffer()
            if step is not None:
                self._current_step = step
                self._save(step)

    def _filter_topk_per_parent(
        self,
        states: list[ErdosState],
        parent_states: list[ErdosState],
    ) -> tuple[list[ErdosState], list[ErdosState]]:
        """Keep only top-k children per parent (by value)."""
        if self.topk_children <= 0 or not states:
            return states, parent_states

        # Group by parent
        parent_to_children: dict[str, list[tuple[ErdosState, ErdosState]]] = {}
        for child, parent in zip(states, parent_states):
            pid = parent.id
            if pid not in parent_to_children:
                parent_to_children[pid] = []
            parent_to_children[pid].append((child, parent))

        # Keep top-k per parent
        filtered_children = []
        filtered_parents = []
        for children in parent_to_children.values():
            sorted_pairs = sorted(
                children,
                key=lambda x: x[0].value if x[0].value is not None else float("-inf"),
                reverse=True,
            )
            for child, parent in sorted_pairs[: self.topk_children]:
                filtered_children.append(child)
                filtered_parents.append(parent)

        return filtered_children, filtered_parents

    def _prune_buffer(self):
        """Prune buffer to max size, keeping best states."""
        if len(self._states) <= self.max_buffer_size:
            return

        # Sort by value (descending)
        values = [s.value if s.value is not None else float("-inf") for s in self._states]
        indices = np.argsort(values)[::-1]

        # Always keep initial states
        initial_ids = {s.id for s in self._initial_states}
        keep = set()
        for i, s in enumerate(self._states):
            if s.id in initial_ids:
                keep.add(i)

        # Keep top states by value
        for idx in indices:
            if len(keep) >= self.max_buffer_size:
                break
            keep.add(idx)

        self._states = [self._states[i] for i in sorted(keep)]

    def flush(self, step: int | None = None):
        """Force save to disk."""
        with self._lock:
            if step is not None:
                self._current_step = step
            self._save(self._current_step)

    def get_best_state(self) -> ErdosState | None:
        """Get the best state in the buffer."""
        if not self._states:
            return None
        best = max(self._states, key=lambda s: s.value if s.value is not None else float("-inf"))
        return best

    def get_stats(self) -> dict[str, Any]:
        """Get sampler statistics for logging."""
        values = [s.value for s in self._states if s.value is not None]
        return {
            "puct/buffer_size": len(self._states),
            "puct/T": self._T,
            "puct/scale": self._last_scale,
            "puct/value_mean": float(np.mean(values)) if values else 0.0,
            "puct/value_max": float(np.max(values)) if values else 0.0,
            "puct/value_min": float(np.min(values)) if values else 0.0,
        }


class GreedySampler(StateSampler):
    """Simple greedy sampler - always returns best state with epsilon exploration."""

    def __init__(
        self,
        log_path: str,
        max_buffer_size: int = 100,
        epsilon: float = 0.1,
    ):
        self.log_path = log_path
        self.max_buffer_size = max_buffer_size
        self.epsilon = epsilon
        self._states: list[ErdosState] = []
        self._lock = threading.Lock()

        Path(log_path).mkdir(parents=True, exist_ok=True)

        if not self._states:
            self._states.append(create_initial_erdos_state())

    def sample_states(self, num_states: int) -> list[ErdosState]:
        if not self._states:
            return [create_initial_erdos_state() for _ in range(num_states)]

        result = []
        for _ in range(num_states):
            if np.random.random() < self.epsilon and len(self._states) > 1:
                result.append(np.random.choice(self._states))
            else:
                # Return best
                best = max(self._states, key=lambda s: s.value if s.value else float("-inf"))
                result.append(best)
        return result

    def update_states(
        self,
        states: list[ErdosState],
        parent_states: list[ErdosState],
        step: int | None = None,
    ):
        with self._lock:
            for child, parent in zip(states, parent_states):
                if child.value is not None:
                    child.parent_values = [parent.value] + parent.parent_values if parent.value else []
                    child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents
                    self._states.append(child)

            # Keep top states
            self._states.sort(key=lambda s: s.value if s.value else float("-inf"), reverse=True)
            self._states = self._states[: self.max_buffer_size]

    def flush(self, step: int | None = None):
        pass

    def get_best_state(self) -> ErdosState | None:
        if not self._states:
            return None
        return max(self._states, key=lambda s: s.value if s.value else float("-inf"))
