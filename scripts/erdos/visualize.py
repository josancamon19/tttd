#!/usr/bin/env python3
"""
Interactive visualization of the Erdős Friends & Strangers Problem (Ramsey Theory).

This demonstrates Ramsey numbers - how many people you need to GUARANTEE finding
a group of k mutual friends or k mutual strangers.

Known Ramsey numbers R(k,k):
  - R(3,3) = 6   → 6 people guarantees 3 mutual friends or 3 mutual strangers
  - R(4,4) = 18  → 18 people guarantees 4 mutual friends or 4 mutual strangers
  - R(5,5) = ?   → Unknown! Somewhere between 43 and 48

Run with: uv run python scripts/erdos/visualize.py
Or with args: uv run python scripts/erdos/visualize.py --people 12 --clique 4
"""

import argparse
import itertools
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.widgets import Button, Slider


# Ramsey numbers R(k,k) - minimum people needed to guarantee k mutual friends/strangers
RAMSEY_NUMBERS = {
    3: 6,
    4: 18,
    5: (43, 48),  # Unknown! Between 43 and 48
    6: (102, 165),  # Unknown! Between 102 and 165
}


def get_names(n):
    """Generate n names for party guests."""
    names = [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
        "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo",
        "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby",
        "Sam", "Tina", "Uma", "Vic", "Wendy", "Xavier", "Yara", "Zack"
    ]
    if n <= len(names):
        return names[:n]
    # Generate more names if needed
    return [f"P{i+1}" for i in range(n)]


def create_random_relationships(n_people):
    """
    Create random friend/stranger relationships between n people.
    Returns a set of tuples representing friendships.
    """
    friendships = set()
    for i, j in itertools.combinations(range(n_people), 2):
        if random.random() < 0.5:
            friendships.add((i, j))
    return friendships


def find_clique(n_people, friendships, clique_size, find_friends=True):
    """
    Find a clique of clique_size mutual friends or mutual strangers.

    Args:
        n_people: Total number of people
        friendships: Set of (i, j) tuples representing friendships
        clique_size: Size of clique to find (3, 4, 5, etc.)
        find_friends: If True, find mutual friends. If False, find mutual strangers.

    Returns:
        Tuple of indices forming the clique, or None if not found.
    """
    for group in itertools.combinations(range(n_people), clique_size):
        # Check all pairs in this group
        all_match = True
        for i, j in itertools.combinations(group, 2):
            is_friend = (min(i, j), max(i, j)) in friendships
            if find_friends and not is_friend:
                all_match = False
                break
            if not find_friends and is_friend:
                all_match = False
                break

        if all_match:
            return group
    return None


def find_guaranteed_clique(n_people, friendships, clique_size):
    """
    Try to find a clique of the given size (either friends or strangers).
    Returns (clique, type) or (None, None) if not found.
    """
    # Try to find mutual friends
    friends_clique = find_clique(n_people, friendships, clique_size, find_friends=True)
    if friends_clique:
        return friends_clique, "friends"

    # Try to find mutual strangers
    strangers_clique = find_clique(n_people, friendships, clique_size, find_friends=False)
    if strangers_clique:
        return strangers_clique, "strangers"

    return None, None


def draw_party(n_people, friendships, highlight_clique=None, clique_type=None,
               clique_size=3, ax=None):
    """
    Draw the party graph showing all relationships.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        ax.clear()

    names = get_names(n_people)

    # Create graph and position nodes in a circle
    G = nx.Graph()
    G.add_nodes_from(range(n_people))
    pos = nx.circular_layout(G)

    # Adjust node size based on number of people
    base_radius = min(0.1, 0.6 / n_people)
    font_size = max(7, min(11, 70 // n_people))

    # Draw all edges (both friends and strangers)
    for i, j in itertools.combinations(range(n_people), 2):
        is_friend = (min(i, j), max(i, j)) in friendships

        # Check if this edge is part of the highlighted clique
        in_clique = False
        if highlight_clique and i in highlight_clique and j in highlight_clique:
            in_clique = True

        # Determine edge style
        if is_friend:
            color = '#27ae60' if in_clique else '#2ecc71'  # Green
            style = 'solid'
            width = 4 if in_clique else (1.5 if n_people <= 10 else 0.8)
            alpha = 1.0 if in_clique else (0.7 if n_people <= 10 else 0.3)
        else:
            color = '#c0392b' if in_clique else '#e74c3c'  # Red
            style = 'dashed'
            width = 4 if in_clique else (1.0 if n_people <= 10 else 0.5)
            alpha = 1.0 if in_clique else (0.5 if n_people <= 10 else 0.15)

        # Draw edge
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color=color, linestyle=style, linewidth=width,
                alpha=alpha, zorder=1 if not in_clique else 2)

    # Draw nodes
    for i in range(n_people):
        in_clique = highlight_clique and i in highlight_clique

        # Node appearance
        node_color = '#f39c12' if in_clique else '#3498db'
        edge_color = '#d35400' if in_clique else '#2980b9'
        radius = base_radius * (1.3 if in_clique else 1.0)

        circle = plt.Circle(pos[i], radius,
                           color=node_color, ec=edge_color, linewidth=2, zorder=3)
        ax.add_patch(circle)

        # Add name label
        ax.text(pos[i][0], pos[i][1], names[i],
                ha='center', va='center', fontsize=font_size,
                fontweight='bold', zorder=4)

    # Add legend
    friend_patch = mpatches.Patch(color='#2ecc71', label='Friends')
    stranger_patch = mpatches.Patch(color='#e74c3c', label='Strangers')
    ax.legend(handles=[friend_patch, stranger_patch], loc='upper left', fontsize=9)

    # Title with Ramsey number info
    ramsey_info = ""
    if clique_size in RAMSEY_NUMBERS:
        r = RAMSEY_NUMBERS[clique_size]
        if isinstance(r, tuple):
            ramsey_info = f"  [R({clique_size},{clique_size}) = {r[0]}-{r[1]}, unknown!]"
        else:
            ramsey_info = f"  [R({clique_size},{clique_size}) = {r}]"
            if n_people >= r:
                ramsey_info += " ← GUARANTEED!"

    ax.set_title(f"Party with {n_people} People — Finding groups of {clique_size}{ramsey_info}",
                 fontsize=14, fontweight='bold', pad=20)

    # Show result
    if highlight_clique and clique_type:
        clique_names = [names[i] for i in highlight_clique]
        if clique_type == "friends":
            explanation = f"Found {len(highlight_clique)} MUTUAL FRIENDS: {', '.join(clique_names)}"
            color = '#27ae60'
        else:
            explanation = f"Found {len(highlight_clique)} MUTUAL STRANGERS: {', '.join(clique_names)}"
            color = '#c0392b'
        ax.text(0.5, -0.12, explanation, transform=ax.transAxes,
                ha='center', fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    elif highlight_clique is None and clique_type == "not_found":
        ax.text(0.5, -0.12, f"No group of {clique_size} mutual friends or strangers found!\n(Need more people for a guarantee)",
                transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
                color='#7f8c8d', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax


class InteractiveParty:
    """Interactive visualization with controls for party size and clique size."""

    def __init__(self, initial_people=6, initial_clique=3):
        self.n_people = initial_people
        self.clique_size = initial_clique

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.28, top=0.92)

        # Sliders
        ax_people = plt.axes([0.25, 0.18, 0.5, 0.03])
        self.slider_people = Slider(ax_people, 'People', 4, 20, valinit=initial_people, valstep=1)
        self.slider_people.on_changed(self.update_people)

        ax_clique = plt.axes([0.25, 0.13, 0.5, 0.03])
        self.slider_clique = Slider(ax_clique, 'Group Size', 3, 5, valinit=initial_clique, valstep=1)
        self.slider_clique.on_changed(self.update_clique)

        # Buttons
        ax_random = plt.axes([0.25, 0.04, 0.2, 0.05])
        self.btn_random = Button(ax_random, 'Randomize', color='lightblue')
        self.btn_random.on_clicked(self.randomize)

        ax_solve = plt.axes([0.55, 0.04, 0.2, 0.05])
        self.btn_solve = Button(ax_solve, 'Find Group!', color='lightyellow')
        self.btn_solve.on_clicked(self.show_solution)

        # Initialize
        self.friendships = create_random_relationships(self.n_people)
        self.showing_solution = False
        self.draw()

        # Add Ramsey number reference
        self.fig.text(0.5, 0.96,
                     "Ramsey Numbers: R(3,3)=6  |  R(4,4)=18  |  R(5,5)=43-48 (unknown!)",
                     ha='center', fontsize=10, style='italic',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    def draw(self):
        if self.showing_solution:
            clique, clique_type = find_guaranteed_clique(self.n_people, self.friendships, self.clique_size)
            if clique is None:
                clique_type = "not_found"
            draw_party(self.n_people, self.friendships, clique, clique_type,
                      self.clique_size, self.ax)
        else:
            draw_party(self.n_people, self.friendships, clique_size=self.clique_size, ax=self.ax)
        self.fig.canvas.draw_idle()

    def update_people(self, val):
        self.n_people = int(val)
        self.friendships = create_random_relationships(self.n_people)
        self.showing_solution = False
        self.draw()

    def update_clique(self, val):
        self.clique_size = int(val)
        self.showing_solution = False
        self.draw()

    def randomize(self, event):
        self.friendships = create_random_relationships(self.n_people)
        self.showing_solution = False
        self.draw()

    def show_solution(self, event):
        self.showing_solution = True
        self.draw()

    def show(self):
        plt.show()


def run_experiment(n_people, clique_size, trials=1000):
    """Run many trials to see how often we find a clique."""
    found = 0
    for _ in range(trials):
        friendships = create_random_relationships(n_people)
        clique, _ = find_guaranteed_clique(n_people, friendships, clique_size)
        if clique:
            found += 1
    return found / trials


def main():
    parser = argparse.ArgumentParser(description="Visualize Erdős Party Problem (Ramsey Theory)")
    parser.add_argument("--people", "-p", type=int, default=6, help="Number of people (default: 6)")
    parser.add_argument("--clique", "-c", type=int, default=3, help="Clique size to find (default: 3)")
    parser.add_argument("--experiment", "-e", action="store_true",
                       help="Run experiment instead of visualization")
    args = parser.parse_args()

    print("=" * 70)
    print("THE PARTY PROBLEM (Erdős - Ramsey Theory)")
    print("=" * 70)
    print()
    print("RAMSEY NUMBERS R(k,k) - minimum people to GUARANTEE k mutual friends/strangers:")
    print("  • R(3,3) = 6   → 6 people always has 3 mutual friends or 3 strangers")
    print("  • R(4,4) = 18  → 18 people always has 4 mutual friends or 4 strangers")
    print("  • R(5,5) = ??  → Unknown! Somewhere between 43 and 48")
    print()

    if args.experiment:
        print(f"Running experiment: {args.people} people, looking for groups of {args.clique}")
        print()
        rate = run_experiment(args.people, args.clique, trials=1000)
        print(f"Found a group of {args.clique} in {rate*100:.1f}% of 1000 random trials")

        if args.clique in RAMSEY_NUMBERS:
            r = RAMSEY_NUMBERS[args.clique]
            if isinstance(r, int):
                if args.people >= r:
                    print(f"(This is expected: R({args.clique},{args.clique})={r}, so {args.people} people GUARANTEES finding one)")
                else:
                    print(f"(R({args.clique},{args.clique})={r}, so you need {r} people for a guarantee)")
    else:
        print("Use the sliders to change:")
        print("  • Number of people at the party")
        print("  • Size of group you're looking for (3, 4, or 5 mutual friends/strangers)")
        print()
        print("Click 'Randomize' for new relationships, 'Find Group!' to search")
        print()

        app = InteractiveParty(args.people, args.clique)
        app.show()


if __name__ == "__main__":
    main()
