#!/usr/bin/env python3
"""
Erdős Minimum Overlap Game

YOU design a pattern of n buckets (each 0.0 to 1.0).
ADVERSARY rotates their inverse pattern to maximize overlap.
YOUR GOAL: Minimize the maximum overlap (C5 score).

Run: uv run python scripts/erdos/game.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle


class ErdosGame:
    def __init__(self):
        self.n = 6  # Default pattern size
        self.h = np.ones(self.n) * 0.5  # Default: all 0.5

        # Create figure
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor('#1a1a2e')

        # Layout
        self.ax_pattern = self.fig.add_axes([0.08, 0.55, 0.84, 0.35])
        self.ax_rotations = self.fig.add_axes([0.08, 0.18, 0.84, 0.30])
        self.ax_n_slider = self.fig.add_axes([0.25, 0.08, 0.50, 0.025])

        # Pattern sliders (will be created dynamically)
        self.pattern_sliders = []
        self.slider_axes = []

        # N slider
        self.n_slider = Slider(self.ax_n_slider, 'Pattern Size (n)', 2, 20,
                               valinit=self.n, valstep=1, color='#e94560')
        self.n_slider.label.set_color('white')
        self.n_slider.valtext.set_color('white')
        self.n_slider.on_changed(self.update_n)

        # Reset button
        self.ax_reset = self.fig.add_axes([0.80, 0.08, 0.12, 0.025])
        self.btn_reset = Button(self.ax_reset, 'Reset to 0.5', color='#0f3460', hovercolor='#16213e')
        self.btn_reset.label.set_color('white')
        self.btn_reset.on_clicked(self.reset_pattern)

        self.create_pattern_sliders()
        self.draw()

    def create_pattern_sliders(self):
        # Remove old sliders
        for ax in self.slider_axes:
            ax.remove()
        self.slider_axes = []
        self.pattern_sliders = []

        # Create new sliders for each bucket
        slider_width = 0.8 / self.n
        for i in range(self.n):
            ax = self.fig.add_axes([0.1 + i * slider_width, 0.92, slider_width * 0.8, 0.015])
            ax.set_facecolor('#16213e')
            slider = Slider(ax, '', 0.0, 1.0, valinit=self.h[i], color='#e94560')
            slider.valtext.set_color('white')
            slider.valtext.set_fontsize(8)
            slider.on_changed(lambda val, idx=i: self.update_bucket(idx, val))
            self.slider_axes.append(ax)
            self.pattern_sliders.append(slider)

    def update_n(self, val):
        self.n = int(val)
        self.h = np.ones(self.n) * 0.5
        self.create_pattern_sliders()
        self.draw()

    def update_bucket(self, idx, val):
        self.h[idx] = val
        self.draw()

    def reset_pattern(self, event):
        self.h = np.ones(self.n) * 0.5
        for i, slider in enumerate(self.pattern_sliders):
            slider.set_val(0.5)
        self.draw()

    def compute_overlap(self, rotation):
        """Compute overlap for a given rotation."""
        j = 1.0 - self.h  # Adversary's inverse
        j_rotated = np.roll(j, rotation)
        return np.sum(self.h * j_rotated) / self.n

    def draw(self):
        # Clear axes
        self.ax_pattern.clear()
        self.ax_rotations.clear()

        # Style
        self.ax_pattern.set_facecolor('#16213e')
        self.ax_rotations.set_facecolor('#16213e')

        j = 1.0 - self.h  # Adversary's inverse

        # === TOP: Your pattern and adversary's inverse ===
        bar_width = 0.35
        x = np.arange(self.n)

        # Your pattern
        bars1 = self.ax_pattern.bar(x - bar_width/2, self.h, bar_width,
                                     color='#4ecca3', label='Your pattern h', edgecolor='white', linewidth=1)
        # Adversary's inverse (no rotation shown here)
        bars2 = self.ax_pattern.bar(x + bar_width/2, j, bar_width,
                                     color='#e94560', alpha=0.7, label="Adversary's inverse (1-h)",
                                     edgecolor='white', linewidth=1)

        self.ax_pattern.set_xlim(-0.5, self.n - 0.5)
        self.ax_pattern.set_ylim(0, 1.15)
        self.ax_pattern.set_xticks(x)
        self.ax_pattern.set_xticklabels([f'{i}' for i in range(self.n)], color='white')
        self.ax_pattern.set_ylabel('Value', color='white', fontsize=11)
        self.ax_pattern.tick_params(colors='white')
        self.ax_pattern.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        self.ax_pattern.set_title('YOUR PATTERN vs ADVERSARY INVERSE', color='white', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars1, self.h):
            self.ax_pattern.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=8)

        # === BOTTOM: All rotations and their overlap scores ===
        overlaps = [self.compute_overlap(r) for r in range(self.n)]
        worst_rotation = np.argmax(overlaps)
        c5_score = overlaps[worst_rotation]

        colors = ['#e94560' if r == worst_rotation else '#4ecca3' for r in range(self.n)]
        bars = self.ax_rotations.bar(x, overlaps, color=colors, edgecolor='white', linewidth=1)

        # Add value labels
        for bar, val, r in zip(bars, overlaps, range(self.n)):
            label_color = 'white'
            fontweight = 'bold' if r == worst_rotation else 'normal'
            self.ax_rotations.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{val:.3f}', ha='center', va='bottom', color=label_color,
                                   fontsize=9, fontweight=fontweight)

        self.ax_rotations.set_xlim(-0.5, self.n - 0.5)
        self.ax_rotations.set_ylim(0, max(0.6, max(overlaps) * 1.2))
        self.ax_rotations.set_xticks(x)
        self.ax_rotations.set_xticklabels([f'rot {i}' for i in range(self.n)], color='white', fontsize=9)
        self.ax_rotations.set_ylabel('Overlap', color='white', fontsize=11)
        self.ax_rotations.tick_params(colors='white')

        # Horizontal line at C5
        self.ax_rotations.axhline(c5_score, color='#e94560', linestyle='--', linewidth=2, alpha=0.8)

        # Title with score
        rating = self.get_rating(c5_score)
        title = f'ADVERSARY ROTATIONS — C5 Score: {c5_score:.4f}  {rating}'
        self.ax_rotations.set_title(title, color='white', fontsize=14, fontweight='bold')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e94560', edgecolor='white', label=f'Worst rotation (adversary picks this)'),
            Patch(facecolor='#4ecca3', edgecolor='white', label='Other rotations'),
        ]
        self.ax_rotations.legend(handles=legend_elements, loc='upper right',
                                  facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=9)

        # Instructions
        self.fig.suptitle('ERDŐS MINIMUM OVERLAP GAME\n' +
                         'Adjust sliders above to change your pattern. Minimize the C5 score (red bar).',
                         color='white', fontsize=12, y=0.99)

        self.fig.canvas.draw_idle()

    def get_rating(self, c5):
        if c5 <= 0.25:
            return '★★★★★ OPTIMAL!'
        elif c5 <= 0.30:
            return '★★★★ Excellent'
        elif c5 <= 0.38:
            return '★★★ Great'
        elif c5 <= 0.45:
            return '★★ Good'
        elif c5 < 0.50:
            return '★ OK'
        else:
            return '— Baseline'

    def show(self):
        plt.show()


def main():
    print("=" * 60)
    print("ERDŐS MINIMUM OVERLAP GAME")
    print("=" * 60)
    print()
    print("GOAL: Minimize the C5 score (maximum overlap)")
    print()
    print("- Adjust the small sliders at top to set each bucket value")
    print("- Use 'Pattern Size' slider to change n (2-20)")
    print("- The RED bar shows the adversary's best attack")
    print("- Try to make all bars as LOW as possible")
    print()
    print("TIPS:")
    print("- All 0.5 gives C5 = 0.25 (baseline)")
    print("- Extreme values (0 or 1) are easy to exploit")
    print("- The best patterns are non-obvious!")
    print()

    game = ErdosGame()
    game.show()


if __name__ == "__main__":
    main()
