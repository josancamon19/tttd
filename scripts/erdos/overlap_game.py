#!/usr/bin/env python3
"""
Interactive game visualization of the Erdős Minimum Overlap Problem.

THE GAME:
=========
You are trying to paint a fence (the interval [0,1]) with a pattern.
Your pattern is a "height function" h(x) where h(x)=1 means fully painted, h(x)=0 means unpainted.

An ADVERSARY will then SHIFT a copy of your inverse pattern and overlay it.
The "overlap" is how much your painted area overlaps with the shifted unpainted area.

YOUR GOAL: Design h(x) to MINIMIZE the MAXIMUM overlap the adversary can achieve.

- If you paint everything (h=1): adversary shifts (1-h)=0 → zero overlap everywhere. Sounds good?
  But wait - you must integrate to 0.5 (paint exactly half the fence on average).

- The trivial solution h(x)=0.5 everywhere gives max overlap = 0.5

- The BEST KNOWN solution achieves max overlap ≈ 0.38 (discovered ~2006)

This is what the LLM is learning to optimize!

Run with: uv run python scripts/erdos/overlap_game.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import FancyBboxPatch


def compute_all_overlaps(h_values: np.ndarray) -> np.ndarray:
    """Compute overlap for all possible shifts using the verifier's method.

    This matches the actual evaluation in tttd/erdos/verifier.py:
    - Uses np.correlate with mode="full"
    - dx = 2.0 / n_points
    """
    n = len(h_values)
    j_values = 1.0 - h_values
    dx = 2.0 / n
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    return correlation


def compute_c5_bound(h_values: np.ndarray) -> tuple[float, int]:
    """Compute the C5 bound (max overlap) and the worst shift."""
    overlaps = compute_all_overlaps(h_values)
    worst_shift = np.argmax(overlaps)
    return float(overlaps[worst_shift]), int(worst_shift)


def compute_overlap_at_shift(h_values: np.ndarray, shift_idx: int) -> float:
    """Compute overlap at a specific shift index in the correlation array."""
    overlaps = compute_all_overlaps(h_values)
    return float(overlaps[shift_idx])


# Preset solutions
def constant_half(n):
    """Trivial solution: h(x) = 0.5 everywhere. Gives C5 = 0.5"""
    return np.ones(n) * 0.5


def two_step_moderate(n):
    """Two-step with moderate heights (not extreme 0/1)."""
    h = np.ones(n) * 0.3
    h[:n//2] = 0.7
    return h


def three_step(n):
    """Three-step function with balanced heights."""
    h = np.zeros(n)
    third = n // 3
    h[:third] = 0.75
    h[third:2*third] = 0.5
    h[2*third:] = 0.25
    return h


def smooth_bump(n):
    """Smooth bump function."""
    x = np.linspace(0, 1, n)
    # A smooth bump centered at 0.5
    h = 0.5 + 0.4 * np.exp(-((x - 0.5) ** 2) / 0.05)
    return np.clip(h, 0, 1)


def staircase(n):
    """Staircase pattern with 5 steps."""
    h = np.zeros(n)
    steps = 5
    for i in range(steps):
        start = int(i * n / steps)
        end = int((i + 1) * n / steps)
        # Alternate high and low
        h[start:end] = 0.8 if i % 2 == 0 else 0.2
    return h


def best_known_approx(n):
    """Approximation of best known solution (~0.38).

    Based on research papers showing optimal solutions use
    carefully tuned step functions.
    """
    h = np.zeros(n)
    # A known good 3-step construction
    # These values are tuned to give low C5
    h[:n//3] = 0.62
    h[n//3:2*n//3] = 0.50
    h[2*n//3:] = 0.38
    return h


PRESETS = {
    "Constant 0.5 (trivial)": constant_half,
    "Two-Step (0.3/0.7)": two_step_moderate,
    "Three-Step": three_step,
    "5-Step Staircase": staircase,
    "Smooth Bump": smooth_bump,
    "~Best Known": best_known_approx,
}


class OverlapGame:
    """Interactive visualization of the Erdős Minimum Overlap Problem."""

    def __init__(self, n_points=200):
        self.n_points = n_points
        self.x = np.linspace(0, 1, n_points)

        # Start with trivial solution
        self.h_values = constant_half(n_points)
        self.shift = n_points - 1  # Start at center (zero shift)
        self.show_adversary = True

        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#f0f0f0')

        # Main plot: h(x) and shifted (1-h)
        self.ax_main = self.fig.add_axes([0.1, 0.55, 0.55, 0.38])

        # Overlap plot: overlap vs shift
        self.ax_overlap = self.fig.add_axes([0.1, 0.15, 0.55, 0.30])

        # Info panel
        self.ax_info = self.fig.add_axes([0.70, 0.55, 0.25, 0.38])
        self.ax_info.axis('off')

        # Controls - slider covers the full correlation range (2n-1 positions)
        self.n_shifts = 2 * n_points - 1
        self.ax_slider = self.fig.add_axes([0.1, 0.05, 0.55, 0.03])
        self.slider = Slider(self.ax_slider, 'Adversary Shift', 0, self.n_shifts - 1,
                            valinit=self.n_shifts // 2, valstep=1)  # Start at center
        self.slider.on_changed(self.update_shift)

        # Preset buttons
        self.ax_radio = self.fig.add_axes([0.70, 0.15, 0.25, 0.30])
        self.radio = RadioButtons(self.ax_radio, list(PRESETS.keys()))
        self.radio.on_clicked(self.select_preset)

        # Worst shift button
        self.ax_worst = self.fig.add_axes([0.70, 0.05, 0.12, 0.05])
        self.btn_worst = Button(self.ax_worst, 'Worst Shift', color='#ffcccc')
        self.btn_worst.on_clicked(self.goto_worst_shift)

        # Animate button
        self.ax_animate = self.fig.add_axes([0.83, 0.05, 0.12, 0.05])
        self.btn_animate = Button(self.ax_animate, 'Animate', color='#ccffcc')
        self.btn_animate.on_clicked(self.animate_shifts)

        self.update_plots()

    def update_shift(self, val):
        self.shift = int(val)
        self.update_plots()

    def select_preset(self, label):
        self.h_values = PRESETS[label](self.n_points)
        self.shift = self.n_points - 1  # Reset to center
        self.slider.set_val(self.shift)
        self.update_plots()

    def goto_worst_shift(self, event):
        _, worst_shift = compute_c5_bound(self.h_values)
        self.shift = worst_shift
        self.slider.set_val(worst_shift)
        self.update_plots()

    def animate_shifts(self, event):
        """Animate through all shifts to show the adversary searching."""
        for t in range(0, self.n_shifts, 4):
            self.shift = t
            self.slider.set_val(t)
            self.update_plots()
            plt.pause(0.01)
        # End at worst shift
        self.goto_worst_shift(None)

    def update_plots(self):
        # Clear axes
        self.ax_main.clear()
        self.ax_overlap.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Compute values
        j_values = 1.0 - self.h_values
        all_overlaps = compute_all_overlaps(self.h_values)
        current_overlap = all_overlaps[self.shift] if self.shift < len(all_overlaps) else 0
        c5_bound, worst_shift = compute_c5_bound(self.h_values)

        # For visualization, convert correlation index to a roll amount
        # The correlation array is centered at index n-1 (zero shift)
        visual_shift = self.shift - (self.n_points - 1)
        j_shifted = np.roll(j_values, visual_shift)

        # === Main plot: h(x) and adversary's shifted (1-h) ===
        self.ax_main.fill_between(self.x, 0, self.h_values, alpha=0.6,
                                   color='#3498db', label='Your pattern h(x)')
        self.ax_main.plot(self.x, self.h_values, color='#2980b9', linewidth=2)

        # Show adversary's shifted inverse
        self.ax_main.fill_between(self.x, 0, j_shifted, alpha=0.4,
                                   color='#e74c3c', label=f"Adversary's (1-h) shifted by {visual_shift/self.n_points:.2f}")
        self.ax_main.plot(self.x, j_shifted, color='#c0392b', linewidth=2, linestyle='--')

        # Highlight overlap region (visual approximation)
        overlap_region = np.minimum(self.h_values, j_shifted)
        self.ax_main.fill_between(self.x, 0, overlap_region, alpha=0.7,
                                   color='#9b59b6', label=f'Overlap ≈ {current_overlap:.4f}')

        self.ax_main.set_xlim(0, 1)
        self.ax_main.set_ylim(0, 1.1)
        self.ax_main.set_xlabel('x (position on fence)', fontsize=11)
        self.ax_main.set_ylabel('Height', fontsize=11)
        self.ax_main.set_title('YOUR PATTERN vs ADVERSARY', fontsize=14, fontweight='bold')
        self.ax_main.legend(loc='upper right', fontsize=9)
        self.ax_main.grid(True, alpha=0.3)

        # === Overlap plot: overlap vs shift ===
        # The correlation array has length 2n-1, representing shifts from -(n-1) to (n-1)
        n_overlaps = len(all_overlaps)
        shift_x = np.linspace(-1, 1, n_overlaps)
        self.ax_overlap.fill_between(shift_x, 0, all_overlaps, alpha=0.4, color='#9b59b6')
        self.ax_overlap.plot(shift_x, all_overlaps, color='#8e44ad', linewidth=2)

        # Mark current shift (convert from index to position)
        current_shift_pos = shift_x[self.shift] if self.shift < len(shift_x) else 0
        self.ax_overlap.axvline(current_shift_pos, color='#e74c3c',
                                linewidth=2, linestyle='--', label='Current shift')
        self.ax_overlap.scatter([current_shift_pos], [current_overlap],
                                color='#e74c3c', s=100, zorder=5)

        # Mark worst shift (max overlap)
        worst_shift_pos = shift_x[worst_shift] if worst_shift < len(shift_x) else 0
        self.ax_overlap.axhline(c5_bound, color='#c0392b', linewidth=2,
                                linestyle=':', label=f'MAX (C5) = {c5_bound:.4f}')
        self.ax_overlap.scatter([worst_shift_pos], [c5_bound],
                                color='#c0392b', s=150, marker='*', zorder=5)

        self.ax_overlap.set_xlim(-1, 1)
        self.ax_overlap.set_ylim(0, max(0.6, c5_bound * 1.2))
        self.ax_overlap.set_xlabel('Shift t (adversary\'s choice)', fontsize=11)
        self.ax_overlap.set_ylabel('Overlap', fontsize=11)
        self.ax_overlap.set_title('OVERLAP AT EACH SHIFT (Adversary picks the MAX)',
                                   fontsize=14, fontweight='bold')
        self.ax_overlap.legend(loc='upper right', fontsize=9)
        self.ax_overlap.grid(True, alpha=0.3)

        # === Info panel ===
        info_text = """THE GAME
════════════════════

YOU design a pattern h(x)
  • h(x) ∈ [0, 1] for all x
  • Think: painting a fence

ADVERSARY picks shift t
  • Overlays (1-h) shifted by t
  • Picks t to MAXIMIZE overlap

YOUR GOAL
  • MINIMIZE the MAX overlap
  • This is the C5 bound

════════════════════
SCOREBOARD
════════════════════

Your C5 bound: {c5:.4f}

Trivial (h=0.5): 0.5000
Best known:      0.3809

{rating}
════════════════════

The LLM learns to design
better h(x) patterns!
""".format(
            c5=c5_bound,
            rating=self.get_rating(c5_bound)
        )

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.canvas.draw_idle()

    def get_rating(self, c5):
        if c5 > 0.49:
            return "Rating: TRIVIAL 😐"
        elif c5 > 0.45:
            return "Rating: BASIC ⭐"
        elif c5 > 0.42:
            return "Rating: GOOD ⭐⭐"
        elif c5 > 0.40:
            return "Rating: GREAT ⭐⭐⭐"
        elif c5 > 0.385:
            return "Rating: EXCELLENT ⭐⭐⭐⭐"
        else:
            return "Rating: OPTIMAL! ⭐⭐⭐⭐⭐"

    def show(self):
        plt.show()


def main():
    print("=" * 70)
    print("THE ERDŐS MINIMUM OVERLAP GAME")
    print("=" * 70)
    print()
    print("THE SETUP:")
    print("  You design a 'height function' h(x) on [0,1]")
    print("  An adversary shifts (1-h) to maximize overlap with your h")
    print("  Your goal: minimize the MAXIMUM overlap (C5 bound)")
    print()
    print("THE MATH:")
    print("  overlap(t) = ∫ h(x) · (1 - h(x+t)) dx")
    print("  C5 = max_t overlap(t)  ← this is what you minimize")
    print()
    print("KNOWN RESULTS:")
    print("  • Trivial h(x)=0.5:  C5 = 0.5000")
    print("  • Best known:        C5 ≈ 0.3809")
    print()
    print("THIS IS WHAT THE LLM IS LEARNING TO OPTIMIZE!")
    print()
    print("Try the presets on the right, then click 'Worst Shift' to see")
    print("where the adversary attacks. Click 'Animate' to watch the search.")
    print()

    game = OverlapGame()
    game.show()


if __name__ == "__main__":
    main()
