#!/usr/bin/env python3
"""
Erdős Minimum Overlap Game - Educational Video

A 2-3 minute explainer for the ML/optimization crowd.
Shows the minimax optimization problem with visual overlap competition.

Run: manim -pqh scripts/erdos/erdos_video.py ErdosGame
"""

from manim import *
import numpy as np


# Color palette
PLAYER_COLOR = "#58C4DD"  # Blue - your pattern
ADVERSARY_COLOR = "#FF6666"  # Red - adversary's inverse
GOOD_COLOR = "#83C167"  # Green - low C5, success
BAD_COLOR = "#FF6666"  # Red - high C5, failure
HIGHLIGHT_COLOR = "#FFFF00"  # Yellow - emphasis
OVERLAP_COLOR = "#9B59B6"  # Purple - overlap product
BG_COLOR = "#1C1C1C"  # Dark background


class ErdosGame(Scene):
    """Complete Erdős Minimum Overlap Game explainer."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        self.scene1_problem_setup()
        self.scene2_what_is_rotation()
        self.scene3_overlap_mechanics()
        self.scene4_naive_fails()
        self.scene5_uniform_insight()
        self.scene6_conclusion()

    def create_bars(self, values, color, bar_width=0.5, max_height=2.0):
        """Create bar chart as VGroup of rectangles."""
        bars = VGroup()
        n = len(values)
        total_width = n * (bar_width + 0.1)

        for i, v in enumerate(values):
            height = max(v * max_height, 0.05)  # minimum height for visibility
            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_color=color,
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=1.5,
            )
            # Position from left to right, bottom-aligned
            x_pos = (i - (n - 1) / 2) * (bar_width + 0.1)
            bar.move_to(RIGHT * x_pos + UP * height / 2)
            bars.add(bar)

        return bars

    def create_labeled_bars(self, values, color, label_text, bar_width=0.5):
        """Create bars with value labels on top."""
        bars = self.create_bars(values, color, bar_width)

        labels = VGroup()
        for i, (bar, v) in enumerate(zip(bars, values)):
            label = Text(f"{v:.1f}", font_size=16, color=WHITE)
            label.next_to(bar, UP, buff=0.05)
            labels.add(label)

        title = Text(label_text, font_size=22, color=color)
        title.next_to(bars, DOWN, buff=0.3)

        return VGroup(bars, labels, title)

    def compute_overlaps(self, h):
        """Compute overlap for all rotations."""
        n = len(h)
        j = 1 - h
        overlaps = []
        for r in range(n):
            j_rot = np.roll(j, r)
            overlap = np.sum(h * j_rot) / n
            overlaps.append(overlap)
        return overlaps

    def scene1_problem_setup(self):
        """Scene 1: Frame as optimization problem (~40s)"""

        # Title
        title = Text("A Minimax Optimization Problem", font_size=44, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.2)
        self.wait(0.5)

        # The setup
        setup_text = VGroup(
            Text("The Erdős Minimum Overlap Game", font_size=32, color=HIGHLIGHT_COLOR),
            Text("", font_size=10),
            Text("Two players. One designs, one attacks.", font_size=26, color=GREY_B),
        ).arrange(DOWN, buff=0.15)
        setup_text.next_to(title, DOWN, buff=0.5)

        self.play(Write(setup_text[0]), run_time=1)
        self.play(FadeIn(setup_text[2]), run_time=0.8)
        self.wait(0.8)

        # Player descriptions
        you_box = VGroup(
            Text("YOU", font_size=28, color=PLAYER_COLOR, weight=BOLD),
            Text("Design a pattern", font_size=20, color=WHITE),
            Text("h = [h₀, h₁, ..., hₙ₋₁]", font_size=18, color=GREY_B),
            Text("each hᵢ ∈ [0, 1]", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.1)

        adv_box = VGroup(
            Text("ADVERSARY", font_size=28, color=ADVERSARY_COLOR, weight=BOLD),
            Text("Gets the inverse", font_size=20, color=WHITE),
            Text("j = 1 - h", font_size=18, color=GREY_B),
            Text("Can ROTATE it", font_size=18, color=HIGHLIGHT_COLOR),
        ).arrange(DOWN, buff=0.1)

        boxes = VGroup(you_box, adv_box).arrange(RIGHT, buff=2)
        boxes.move_to(DOWN * 0.5)

        self.play(FadeIn(you_box), run_time=0.8)
        self.wait(0.3)
        self.play(FadeIn(adv_box), run_time=0.8)
        self.wait(1)

        # The objective
        objective = VGroup(
            Text("YOUR GOAL:", font_size=24, color=WHITE),
            MathTex(r"\min_h \max_r \text{overlap}(h, r)", font_size=36, color=HIGHLIGHT_COLOR),
        ).arrange(RIGHT, buff=0.3)
        objective.to_edge(DOWN, buff=0.8)

        self.play(Write(objective), run_time=1.2)
        self.wait(1)

        # Emphasize minimax
        minimax_text = Text("← Minimax: you minimize, adversary maximizes",
                           font_size=20, color=GREY_B)
        minimax_text.next_to(objective, DOWN, buff=0.2)
        self.play(FadeIn(minimax_text))
        self.wait(1.5)

        # Cleanup
        self.play(
            FadeOut(title), FadeOut(setup_text), FadeOut(boxes),
            FadeOut(objective), FadeOut(minimax_text),
            run_time=0.8
        )

    def scene2_what_is_rotation(self):
        """Scene 2: Explain what rotation means with clear L→R animation (~40s)"""

        title = Text("What Does 'Rotate' Mean?", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # Show adversary's pattern
        j = np.array([0.7, 0.3, 0.6, 0.2, 0.8, 0.4])

        # Create bars with index labels below
        bars = self.create_bars(j, ADVERSARY_COLOR, bar_width=0.6, max_height=2.0)
        bars.move_to(ORIGIN)

        # Index labels
        index_labels = VGroup()
        for i, bar in enumerate(bars):
            idx = Text(f"{i}", font_size=18, color=GREY_B)
            idx.next_to(bar, DOWN, buff=0.15)
            index_labels.add(idx)

        # Value labels
        value_labels = VGroup()
        for i, (bar, v) in enumerate(zip(bars, j)):
            val = Text(f"{v:.1f}", font_size=16, color=WHITE)
            val.next_to(bar, UP, buff=0.08)
            value_labels.add(val)

        adv_label = Text("Adversary's pattern", font_size=22, color=ADVERSARY_COLOR)
        adv_label.next_to(bars, DOWN, buff=0.6)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.1),
            run_time=1
        )
        self.play(FadeIn(index_labels), FadeIn(value_labels), FadeIn(adv_label))
        self.wait(0.5)

        # Explain rotation
        explain = Text("Rotation = shift all values cyclically →", font_size=24, color=HIGHLIGHT_COLOR)
        explain.next_to(title, DOWN, buff=0.4)
        self.play(Write(explain))
        self.wait(0.5)

        # Show rotation animation - values shift LEFT, wrapping around
        # We'll do this 3 times to make it clear

        for rot in range(3):
            # Current array state
            j_rotated = np.roll(j, rot + 1)

            # Create arrow showing movement direction
            arrow = Arrow(
                start=bars.get_right() + RIGHT * 0.3,
                end=bars.get_left() + LEFT * 0.3,
                color=HIGHLIGHT_COLOR,
                stroke_width=3
            )
            arrow.shift(UP * 2.5)

            wrap_text = Text("wraps around", font_size=16, color=GREY_B)
            wrap_text.next_to(arrow, UP, buff=0.1)

            if rot == 0:
                self.play(GrowArrow(arrow), FadeIn(wrap_text))

            # Animate bars shifting
            # Create new bars at new positions
            new_bars = self.create_bars(j_rotated, ADVERSARY_COLOR, bar_width=0.6, max_height=2.0)
            new_bars.move_to(ORIGIN)

            new_value_labels = VGroup()
            for i, (bar, v) in enumerate(zip(new_bars, j_rotated)):
                val = Text(f"{v:.1f}", font_size=16, color=WHITE)
                val.next_to(bar, UP, buff=0.08)
                new_value_labels.add(val)

            # Show rotation number
            rot_label = Text(f"Rotation {rot + 1}", font_size=28, color=WHITE)
            rot_label.to_edge(RIGHT, buff=1)
            rot_label.shift(UP * 0.5)

            if rot == 0:
                self.play(Write(rot_label))
            else:
                new_rot_label = Text(f"Rotation {rot + 1}", font_size=28, color=WHITE)
                new_rot_label.to_edge(RIGHT, buff=1)
                new_rot_label.shift(UP * 0.5)
                self.play(Transform(rot_label, new_rot_label))

            # Animate the shift with a smooth left motion
            self.play(
                Transform(bars, new_bars),
                Transform(value_labels, new_value_labels),
                run_time=0.7
            )
            self.wait(0.3)

            if rot == 0:
                # Clean up arrow after first rotation demo
                self.wait(0.3)
                self.play(FadeOut(arrow), FadeOut(wrap_text))

        self.wait(0.5)

        # Now explain WHY adversary rotates
        why_title = Text("WHY does adversary rotate?", font_size=28, color=HIGHLIGHT_COLOR)
        why_title.next_to(explain, DOWN, buff=0.4)

        self.play(
            FadeOut(rot_label),
            Write(why_title)
        )

        # Show the goal
        goal_box = VGroup(
            Text("Adversary's goal:", font_size=22, color=ADVERSARY_COLOR),
            Text("Find the rotation that MAXIMIZES overlap", font_size=24, color=WHITE),
            Text("with YOUR pattern", font_size=24, color=PLAYER_COLOR),
        ).arrange(DOWN, buff=0.1)
        goal_box.to_edge(DOWN, buff=0.8)

        self.play(Write(goal_box), run_time=1.2)
        self.wait(1)

        # Emphasize
        max_emphasis = Text("They're searching for your WEAKNESS", font_size=26, color=BAD_COLOR)
        max_emphasis.next_to(goal_box, DOWN, buff=0.3)
        self.play(Write(max_emphasis))
        self.wait(1.5)

        # Cleanup
        self.play(
            FadeOut(title), FadeOut(explain), FadeOut(bars),
            FadeOut(index_labels), FadeOut(value_labels), FadeOut(adv_label),
            FadeOut(why_title), FadeOut(goal_box), FadeOut(max_emphasis),
            run_time=0.8
        )

    def scene3_overlap_mechanics(self):
        """Scene 2: Show how overlap works with visual competition (~50s)"""

        title = Text("How Overlap Works", font_size=38, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # Example pattern
        h = np.array([0.3, 0.7, 0.4, 0.8, 0.2, 0.6])
        j = 1 - h

        # Create your pattern on the left
        your_bars = self.create_bars(h, PLAYER_COLOR, bar_width=0.4)
        your_bars.move_to(LEFT * 3.5 + DOWN * 0.5)

        your_label = Text("Your pattern h", font_size=20, color=PLAYER_COLOR)
        your_label.next_to(your_bars, DOWN, buff=0.4)

        # Value labels
        your_values = VGroup()
        for i, (bar, v) in enumerate(zip(your_bars, h)):
            lbl = Text(f"{v:.1f}", font_size=14, color=WHITE)
            lbl.next_to(bar, UP, buff=0.05)
            your_values.add(lbl)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in your_bars], lag_ratio=0.08),
            run_time=1
        )
        self.play(FadeIn(your_label), FadeIn(your_values))
        self.wait(0.5)

        # Create adversary inverse on the right
        adv_bars = self.create_bars(j, ADVERSARY_COLOR, bar_width=0.4)
        adv_bars.move_to(RIGHT * 3.5 + DOWN * 0.5)

        adv_label = Text("Adversary: 1 - h", font_size=20, color=ADVERSARY_COLOR)
        adv_label.next_to(adv_bars, DOWN, buff=0.4)

        adv_values = VGroup()
        for i, (bar, v) in enumerate(zip(adv_bars, j)):
            lbl = Text(f"{v:.1f}", font_size=14, color=WHITE)
            lbl.next_to(bar, UP, buff=0.05)
            adv_values.add(lbl)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in adv_bars], lag_ratio=0.08),
            run_time=1
        )
        self.play(FadeIn(adv_label), FadeIn(adv_values))
        self.wait(0.5)

        # Show the overlap formula
        formula = MathTex(
            r"\text{overlap}(r) = \frac{1}{n}\sum_{i=0}^{n-1} h_i \cdot j_{(i+r) \bmod n}",
            font_size=28
        )
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula), run_time=1)
        self.wait(0.8)

        # Now show overlap visualization - bring bars together
        explanation = Text("Let's see the overlap at rotation r=0...",
                          font_size=22, color=GREY_B)
        explanation.next_to(title, DOWN, buff=0.3)
        self.play(Write(explanation))

        # Move both to center, overlapping
        center_pos = DOWN * 0.3

        your_bars_copy = your_bars.copy()
        adv_bars_copy = adv_bars.copy()

        self.play(
            your_bars.animate.move_to(center_pos + LEFT * 0.15),
            adv_bars.animate.move_to(center_pos + RIGHT * 0.15),
            FadeOut(your_label), FadeOut(adv_label),
            FadeOut(your_values), FadeOut(adv_values),
            run_time=1
        )
        self.wait(0.3)

        # Show products at each position
        products = h * j
        overlap_r0 = np.sum(products) / len(h)

        product_labels = VGroup()
        for i, (ybar, abar, prod) in enumerate(zip(your_bars, adv_bars, products)):
            # Highlight the multiplication
            prod_text = MathTex(f"{h[i]:.1f} \\times {j[i]:.1f} = {prod:.2f}", font_size=16)
            prod_text.next_to(VGroup(ybar, abar), UP, buff=0.3)
            product_labels.add(prod_text)

        self.play(
            LaggedStart(*[FadeIn(p, shift=DOWN*0.2) for p in product_labels], lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(0.5)

        # Show sum
        overlap_text = MathTex(
            f"\\text{{overlap}}(0) = \\frac{{1}}{{6}}({' + '.join([f'{p:.2f}' for p in products])}) = {overlap_r0:.3f}",
            font_size=24
        )
        overlap_text.next_to(product_labels, UP, buff=0.3)
        self.play(Write(overlap_text), run_time=1)
        self.wait(1)

        # Now show rotation
        rotate_text = Text("But adversary can ROTATE to find worst case...",
                          font_size=22, color=HIGHLIGHT_COLOR)
        rotate_text.next_to(title, DOWN, buff=0.3)
        self.play(ReplacementTransform(explanation, rotate_text))
        self.wait(0.5)

        # Compute all overlaps
        overlaps = self.compute_overlaps(h)
        worst_r = np.argmax(overlaps)

        # Show adversary searching indicator
        search_label = Text("Adversary searching for MAX...", font_size=22, color=ADVERSARY_COLOR)
        search_label.to_edge(RIGHT, buff=0.5).shift(UP * 2)
        self.play(FadeIn(search_label))

        # Track best so far
        best_so_far = overlaps[0]
        best_r_so_far = 0

        best_indicator = VGroup(
            Text("Best found:", font_size=18, color=GREY_B),
            Text(f"r={best_r_so_far}: {best_so_far:.3f}", font_size=20, color=ADVERSARY_COLOR)
        ).arrange(DOWN, buff=0.1)
        best_indicator.next_to(search_label, DOWN, buff=0.3)
        self.play(FadeIn(best_indicator))

        # Animate through rotations, showing overlap changing
        for r in range(1, len(h)):
            j_rot = np.roll(j, r)
            new_adv_bars = self.create_bars(j_rot, ADVERSARY_COLOR, bar_width=0.4)
            new_adv_bars.move_to(center_pos + RIGHT * 0.15)

            # Update products
            new_products = h * j_rot
            new_overlap = overlaps[r]

            new_product_labels = VGroup()
            for i, (ybar, prod) in enumerate(zip(your_bars, new_products)):
                prod_text = MathTex(f"{h[i]:.1f} \\times {j_rot[i]:.1f} = {new_products[i]:.2f}", font_size=16)
                prod_text.next_to(ybar, UP, buff=0.3)
                new_product_labels.add(prod_text)

            # Color based on whether this is better for adversary
            is_new_best = new_overlap > best_so_far
            color = ADVERSARY_COLOR if is_new_best else GREY_B
            new_overlap_text = MathTex(
                f"\\text{{overlap}}({r}) = {new_overlap:.3f}",
                font_size=28,
                color=color
            )
            new_overlap_text.next_to(new_product_labels, UP, buff=0.3)

            self.play(
                Transform(adv_bars, new_adv_bars),
                Transform(product_labels, new_product_labels),
                Transform(overlap_text, new_overlap_text),
                run_time=0.5
            )

            # Update best if this is better for adversary
            if is_new_best:
                best_so_far = new_overlap
                best_r_so_far = r
                new_best_indicator = VGroup(
                    Text("Best found:", font_size=18, color=GREY_B),
                    Text(f"r={best_r_so_far}: {best_so_far:.3f}", font_size=20, color=ADVERSARY_COLOR)
                ).arrange(DOWN, buff=0.1)
                new_best_indicator.next_to(search_label, DOWN, buff=0.3)
                self.play(
                    Transform(best_indicator, new_best_indicator),
                    Indicate(overlap_text, color=ADVERSARY_COLOR),
                    run_time=0.4
                )

            if r == worst_r:
                self.wait(0.3)
                worst_label = Text("← MAXIMUM! Adversary picks this!", font_size=18, color=BAD_COLOR)
                worst_label.next_to(overlap_text, RIGHT, buff=0.2)
                self.play(FadeIn(worst_label))
                self.wait(0.5)

        self.wait(0.5)

        # C5 definition
        c5_def = MathTex(
            r"C_5 = \max_r \text{overlap}(r) = " + f"{max(overlaps):.3f}",
            font_size=32,
            color=HIGHLIGHT_COLOR
        )
        c5_def.to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(formula, c5_def))
        self.wait(1.5)

        # Cleanup
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.8
        )

    def scene4_naive_fails(self):
        """Scene 4: Show why extreme patterns fail (~40s)"""

        title = Text("Why Extreme Values Fail", font_size=38, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # Intuition
        intuition = Text(
            "Intuition: If I'm at 1, their inverse is 0. Product = 0!",
            font_size=24, color=GREY_B
        )
        intuition.next_to(title, DOWN, buff=0.3)
        self.play(Write(intuition), run_time=1)
        self.wait(0.5)

        # Extreme alternating pattern
        h = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        j = 1 - h  # [0, 1, 0, 1, 0, 1]

        # Show pattern
        your_bars = self.create_bars(h, PLAYER_COLOR, bar_width=0.5, max_height=1.8)
        your_bars.move_to(LEFT * 2.5 + DOWN * 0.3)

        pattern_label = MathTex(r"h = [1, 0, 1, 0, 1, 0]", font_size=24, color=PLAYER_COLOR)
        pattern_label.next_to(your_bars, DOWN, buff=0.4)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in your_bars], lag_ratio=0.08),
            Write(pattern_label),
            run_time=1
        )

        # Show inverse
        adv_bars = self.create_bars(j, ADVERSARY_COLOR, bar_width=0.5, max_height=1.8)
        adv_bars.move_to(RIGHT * 2.5 + DOWN * 0.3)

        inv_label = MathTex(r"1-h = [0, 1, 0, 1, 0, 1]", font_size=24, color=ADVERSARY_COLOR)
        inv_label.next_to(adv_bars, DOWN, buff=0.4)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in adv_bars], lag_ratio=0.08),
            Write(inv_label),
            run_time=1
        )
        self.wait(0.5)

        # Rotation 0: overlap = 0
        r0_text = Text("Rotation 0: overlap = 0 ✓", font_size=24, color=GOOD_COLOR)
        r0_text.move_to(DOWN * 2.5)
        self.play(Write(r0_text))
        self.wait(0.3)

        looks_good = Text("Looks great!", font_size=22, color=GOOD_COLOR)
        looks_good.next_to(r0_text, DOWN, buff=0.2)
        self.play(FadeIn(looks_good))
        self.wait(0.5)

        # But wait...
        but_wait = Text("But the adversary rotates by 1...", font_size=24, color=HIGHLIGHT_COLOR)
        but_wait.move_to(DOWN * 2.5)
        self.play(
            ReplacementTransform(r0_text, but_wait),
            FadeOut(looks_good),
            FadeOut(intuition)
        )

        # Rotate inverse by 1
        j_rot = np.roll(j, 1)  # [1, 0, 1, 0, 1, 0] - same as h!
        new_adv_bars = self.create_bars(j_rot, ADVERSARY_COLOR, bar_width=0.5, max_height=1.8)
        new_adv_bars.move_to(RIGHT * 2.5 + DOWN * 0.3)

        self.play(Transform(adv_bars, new_adv_bars), run_time=0.8)

        new_inv_label = MathTex(r"\text{rotated} = [1, 0, 1, 0, 1, 0]", font_size=24, color=ADVERSARY_COLOR)
        new_inv_label.next_to(adv_bars, DOWN, buff=0.4)
        self.play(Transform(inv_label, new_inv_label))
        self.wait(0.3)

        # They match!
        match_text = Text("They're IDENTICAL!", font_size=28, color=BAD_COLOR)
        match_text.move_to(DOWN * 2.5)
        self.play(ReplacementTransform(but_wait, match_text))

        # Flash both red
        self.play(
            *[bar.animate.set_fill(BAD_COLOR) for bar in your_bars],
            *[bar.animate.set_fill(BAD_COLOR) for bar in adv_bars],
            run_time=0.4
        )
        self.wait(0.3)
        self.play(
            *[bar.animate.set_fill(PLAYER_COLOR) for bar in your_bars],
            *[bar.animate.set_fill(ADVERSARY_COLOR) for bar in adv_bars],
            run_time=0.3
        )

        # Show the disaster
        overlap_calc = MathTex(
            r"\text{overlap}(1) = \frac{1}{6}(1 \cdot 1 + 0 \cdot 0 + \cdots) = \frac{3}{6} = 0.5",
            font_size=26
        )
        overlap_calc.move_to(DOWN * 2.8)

        disaster = Text("Worst possible overlap!", font_size=24, color=BAD_COLOR)
        disaster.next_to(overlap_calc, DOWN, buff=0.2)

        self.play(
            FadeOut(match_text),
            Write(overlap_calc),
            run_time=1
        )
        self.play(FadeIn(disaster))
        self.wait(0.5)

        # C5 score
        c5_bad = MathTex(r"C_5 = 0.5", font_size=40, color=BAD_COLOR)
        c5_bad.move_to(DOWN * 2.3)

        self.play(
            FadeOut(overlap_calc),
            FadeOut(disaster),
            Write(c5_bad)
        )
        self.wait(0.8)

        # Lesson
        lesson = Text(
            "Extreme patterns have predictable inverses → easy to exploit",
            font_size=22, color=GREY_B
        )
        lesson.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(lesson))
        self.wait(1.5)

        # Cleanup
        self.play(
            FadeOut(title), FadeOut(your_bars), FadeOut(adv_bars),
            FadeOut(pattern_label), FadeOut(inv_label),
            FadeOut(c5_bad), FadeOut(lesson),
            run_time=0.8
        )

    def scene5_uniform_insight(self):
        """Scene 5: The uniform solution and why it works (~50s)"""

        title = Text("The Optimal Strategy", font_size=38, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # Question
        question = Text("What if we use uniform 0.5 everywhere?", font_size=26, color=GREY_B)
        question.next_to(title, DOWN, buff=0.3)
        self.play(Write(question))
        self.wait(0.5)

        # Uniform pattern
        h = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        j = 1 - h  # Also [0.5, 0.5, ...]

        your_bars = self.create_bars(h, PLAYER_COLOR, bar_width=0.5, max_height=1.8)
        your_bars.move_to(LEFT * 2.5 + DOWN * 0.3)

        pattern_label = MathTex(r"h = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]", font_size=22, color=PLAYER_COLOR)
        pattern_label.next_to(your_bars, DOWN, buff=0.4)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in your_bars], lag_ratio=0.08),
            Write(pattern_label),
            run_time=1
        )

        # Show inverse - also uniform!
        adv_bars = self.create_bars(j, ADVERSARY_COLOR, bar_width=0.5, max_height=1.8)
        adv_bars.move_to(RIGHT * 2.5 + DOWN * 0.3)

        inv_label = MathTex(r"1-h = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]", font_size=22, color=ADVERSARY_COLOR)
        inv_label.next_to(adv_bars, DOWN, buff=0.4)

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in adv_bars], lag_ratio=0.08),
            Write(inv_label),
            run_time=1
        )

        # Key insight
        key_insight = Text("The inverse is ALSO uniform!", font_size=28, color=HIGHLIGHT_COLOR)
        key_insight.move_to(UP * 0.8)
        self.play(
            FadeOut(question),
            Write(key_insight)
        )
        self.wait(0.5)

        # Rotate multiple times - nothing changes!
        rotate_text = Text("Rotate all you want...", font_size=22, color=GREY_B)
        rotate_text.next_to(key_insight, DOWN, buff=0.3)
        self.play(FadeIn(rotate_text))

        for r in range(4):
            j_rot = np.roll(j, 1)
            new_adv_bars = self.create_bars(j_rot, ADVERSARY_COLOR, bar_width=0.5, max_height=1.8)
            new_adv_bars.move_to(RIGHT * 2.5 + DOWN * 0.3)
            self.play(Transform(adv_bars, new_adv_bars), run_time=0.3)

        same_text = Text("...it looks IDENTICAL", font_size=22, color=GOOD_COLOR)
        same_text.next_to(key_insight, DOWN, buff=0.3)
        self.play(ReplacementTransform(rotate_text, same_text))
        self.wait(0.5)

        # Show the math
        math_box = VGroup(
            MathTex(r"\text{Every product:} \quad 0.5 \times 0.5 = 0.25", font_size=26),
            MathTex(r"\text{Every rotation:} \quad \text{overlap} = 0.25", font_size=26),
            MathTex(r"C_5 = \max_r \text{overlap}(r) = 0.25", font_size=30, color=GOOD_COLOR),
        ).arrange(DOWN, buff=0.2)
        math_box.move_to(DOWN * 2.3)

        self.play(Write(math_box[0]), run_time=0.8)
        self.wait(0.3)
        self.play(Write(math_box[1]), run_time=0.8)
        self.wait(0.3)
        self.play(Write(math_box[2]), run_time=1)

        # Flash green
        self.play(
            *[bar.animate.set_fill(GOOD_COLOR) for bar in your_bars],
            *[bar.animate.set_fill(GOOD_COLOR) for bar in adv_bars],
            run_time=0.4
        )
        self.play(
            *[bar.animate.set_fill(PLAYER_COLOR) for bar in your_bars],
            *[bar.animate.set_fill(ADVERSARY_COLOR) for bar in adv_bars],
            run_time=0.3
        )
        self.wait(0.5)

        # The insight
        insight_box = VGroup(
            Text("Rotation Invariance = No Exploitable Weakness", font_size=24, color=WHITE),
            Text("The adversary has no advantage!", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.15)
        insight_box.to_edge(DOWN, buff=0.3)

        self.play(
            FadeOut(math_box),
            Write(insight_box),
            run_time=1
        )
        self.wait(1)

        # This is a Nash equilibrium
        nash = Text("This is a Nash Equilibrium", font_size=28, color=HIGHLIGHT_COLOR)
        nash.move_to(DOWN * 2.5)

        self.play(
            FadeOut(insight_box),
            Write(nash)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(title), FadeOut(key_insight), FadeOut(same_text),
            FadeOut(your_bars), FadeOut(adv_bars),
            FadeOut(pattern_label), FadeOut(inv_label),
            FadeOut(nash),
            run_time=0.8
        )

    def scene6_conclusion(self):
        """Scene 6: Summary and conclusion (~30s)"""

        # Summary
        title = Text("The Erdős Minimum Overlap Game", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        summary = VGroup(
            Text("• Minimax optimization: you minimize, adversary maximizes", font_size=22, color=WHITE),
            Text("• Extreme patterns are exploitable (C₅ = 0.5)", font_size=22, color=BAD_COLOR),
            Text("• Uniform 0.5 achieves rotation invariance", font_size=22, color=WHITE),
            Text("• C₅ = 0.25 is optimal for balanced allocations", font_size=22, color=GOOD_COLOR),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        summary.move_to(UP * 0.3)

        for item in summary:
            self.play(Write(item), run_time=0.8)
            self.wait(0.3)

        self.wait(1)

        # The deeper question
        deeper = VGroup(
            Text("The deeper question:", font_size=26, color=HIGHLIGHT_COLOR),
            Text("Can non-uniform patterns ever beat 0.25?", font_size=24, color=WHITE),
            Text("(Under different constraints, the search continues...)", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.15)
        deeper.move_to(DOWN * 2)

        self.play(Write(deeper), run_time=1.5)
        self.wait(1.5)

        # CTA
        cta = Text("Try it yourself!", font_size=36, color=GOOD_COLOR)
        cta.move_to(DOWN * 2.5)

        self.play(
            FadeOut(deeper),
            Write(cta)
        )
        self.wait(2)

        # Fade all
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1
        )
        self.wait(0.5)


if __name__ == "__main__":
    print("Run with: manim -pql scripts/erdos/erdos_video.py ErdosGame")
    print("For high quality: manim -pqh scripts/erdos/erdos_video.py ErdosGame")
