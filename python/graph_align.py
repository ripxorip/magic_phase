#!/usr/bin/env python3
"""
Spanning Tree Alignment for Magic Phase.

Uses all-pairs correlation matrix to build a maximum spanning tree.
The tree determines which tracks align to which, cascading through
the strongest pairwise connections.

See docs/SPANNING_TREE_ALIGNMENT.md for the full spec.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque


@dataclass
class PairResult:
    """Result of pairwise cross-correlation."""
    i: int              # Track index A
    j: int              # Track index B
    correlation: float  # Absolute correlation coefficient
    delay: int          # Delay in samples (positive = j is late)
    polarity: int       # +1 or -1


@dataclass
class TreeEdge:
    """Edge in the spanning tree."""
    parent: int         # Parent track index
    child: int          # Child track index
    correlation: float  # Edge weight
    delay: int          # Delay from parent to child
    polarity: int       # Polarity flip needed


@dataclass
class TrackCorrection:
    """Accumulated correction for a track."""
    track_idx: int
    cumulative_delay: float     # Total delay from root
    cumulative_polarity: int    # +1 or -1
    parent_idx: Optional[int]   # None for root
    is_orphan: bool             # No edges above threshold


def compute_correlation_matrix(
    audios: List[np.ndarray],
    sr: int,
    max_delay_ms: float = 50.0,
    detect_delay_fn=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all-pairs cross-correlation matrix.

    Returns:
        corr_matrix: N×N correlation coefficients (absolute value)
        delay_matrix: N×N delays in samples (delay[i,j] = how much j lags i)
        polarity_matrix: N×N polarity values (+1 or -1)
    """
    from align_files import detect_delay_xcorr

    if detect_delay_fn is None:
        detect_delay_fn = detect_delay_xcorr

    N = len(audios)
    corr_matrix = np.zeros((N, N))
    delay_matrix = np.zeros((N, N))
    polarity_matrix = np.ones((N, N), dtype=int)

    # Compute upper triangle, mirror to lower
    for i in range(N):
        for j in range(i + 1, N):
            delay, corr, pol = detect_delay_fn(audios[i], audios[j], sr, max_delay_ms)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            delay_matrix[i, j] = delay
            delay_matrix[j, i] = -delay  # Opposite direction
            polarity_matrix[i, j] = pol
            polarity_matrix[j, i] = pol

    return corr_matrix, delay_matrix, polarity_matrix


def max_spanning_tree(
    corr_matrix: np.ndarray,
    threshold: float = 0.15
) -> List[TreeEdge]:
    """
    Build maximum spanning tree using Kruskal's algorithm.

    Edges below threshold are dropped. Tracks with no edges above
    threshold become orphans (handled separately).

    Returns:
        List of TreeEdge (parent, child, weight) - note: parent/child
        assignment happens later in root selection.
    """
    N = corr_matrix.shape[0]

    # Collect all edges above threshold
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if corr_matrix[i, j] >= threshold:
                edges.append((corr_matrix[i, j], i, j))

    # Sort by correlation descending (max spanning tree)
    edges.sort(reverse=True)

    # Union-Find for cycle detection
    parent = list(range(N))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Kruskal's: add edges that don't create cycles
    tree_edges = []
    for corr, i, j in edges:
        if union(i, j):
            # Edge added (direction determined later)
            tree_edges.append((i, j, corr))

    return tree_edges


def select_root(
    corr_matrix: np.ndarray,
    threshold: float = 0.15
) -> int:
    """
    Select root as track with highest correlation row-sum.

    Only counts correlations above threshold.
    """
    N = corr_matrix.shape[0]
    row_sums = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i != j and corr_matrix[i, j] >= threshold:
                row_sums[i] += corr_matrix[i, j]

    return int(np.argmax(row_sums))


def build_directed_tree(
    tree_edges: List[Tuple[int, int, float]],
    root: int,
    delay_matrix: np.ndarray,
    polarity_matrix: np.ndarray
) -> Dict[int, TreeEdge]:
    """
    Convert undirected tree edges to directed (parent → child) with root.

    Returns:
        Dict mapping child_idx → TreeEdge (with parent, delay, polarity)
    """
    N = delay_matrix.shape[0]

    # Build adjacency list
    adj = {i: [] for i in range(N)}
    for i, j, corr in tree_edges:
        adj[i].append((j, corr))
        adj[j].append((i, corr))

    # BFS from root to assign directions
    directed = {}
    visited = {root}
    queue = deque([root])

    while queue:
        parent = queue.popleft()
        for child, corr in adj[parent]:
            if child not in visited:
                visited.add(child)
                queue.append(child)
                directed[child] = TreeEdge(
                    parent=parent,
                    child=child,
                    correlation=corr,
                    delay=int(delay_matrix[parent, child]),
                    polarity=int(polarity_matrix[parent, child])
                )

    return directed


def compute_corrections(
    directed_tree: Dict[int, TreeEdge],
    root: int,
    N: int
) -> Dict[int, TrackCorrection]:
    """
    Compute cumulative corrections for each track via tree walk.

    Returns:
        Dict mapping track_idx → TrackCorrection
    """
    corrections = {}

    # Root is untouched
    corrections[root] = TrackCorrection(
        track_idx=root,
        cumulative_delay=0.0,
        cumulative_polarity=1,
        parent_idx=None,
        is_orphan=False
    )

    # BFS to accumulate corrections (parent must be done before child)
    queue = deque([root])
    visited = {root}

    while queue:
        parent_idx = queue.popleft()
        parent_corr = corrections[parent_idx]

        # Find children of this parent
        for child_idx, edge in directed_tree.items():
            if edge.parent == parent_idx and child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

                # Accumulate corrections
                corrections[child_idx] = TrackCorrection(
                    track_idx=child_idx,
                    cumulative_delay=parent_corr.cumulative_delay + edge.delay,
                    cumulative_polarity=parent_corr.cumulative_polarity * edge.polarity,
                    parent_idx=parent_idx,
                    is_orphan=False
                )

    # Mark orphans (tracks not in tree)
    for i in range(N):
        if i not in corrections:
            corrections[i] = TrackCorrection(
                track_idx=i,
                cumulative_delay=0.0,
                cumulative_polarity=1,
                parent_idx=None,
                is_orphan=True
            )

    return corrections


def spanning_tree_align(
    audios: List[np.ndarray],
    sr: int,
    track_names: Optional[List[str]] = None,
    threshold: float = 0.15,
    max_delay_ms: float = 50.0,
    verbose: bool = True
) -> Tuple[Dict[int, TrackCorrection], np.ndarray, int, Dict[int, TreeEdge]]:
    """
    Full spanning tree alignment algorithm.

    Returns:
        corrections: Dict[track_idx] → TrackCorrection
        corr_matrix: N×N correlation matrix
        root: Index of root track
        directed_tree: Dict[child_idx] → TreeEdge
    """
    N = len(audios)
    if track_names is None:
        track_names = [f"Track {i}" for i in range(N)]

    if verbose:
        print(f"\n  Computing {N}×{N} correlation matrix...")

    corr_matrix, delay_matrix, polarity_matrix = compute_correlation_matrix(
        audios, sr, max_delay_ms
    )

    if verbose:
        print(f"  Building maximum spanning tree (threshold={threshold})...")

    tree_edges = max_spanning_tree(corr_matrix, threshold)
    root = select_root(corr_matrix, threshold)

    if verbose:
        print(f"  Root selected: {track_names[root]} (highest row-sum)")

    directed_tree = build_directed_tree(
        tree_edges, root, delay_matrix, polarity_matrix
    )

    corrections = compute_corrections(directed_tree, root, N)

    if verbose:
        print(f"\n  Tree structure:")
        _print_tree(root, directed_tree, corrections, track_names, corr_matrix)

    return corrections, corr_matrix, root, directed_tree


def _print_tree(
    root: int,
    directed_tree: Dict[int, TreeEdge],
    corrections: Dict[int, TrackCorrection],
    track_names: List[str],
    corr_matrix: np.ndarray,
    indent: int = 4
):
    """Print tree structure with corrections."""
    N = len(track_names)

    # Build children map
    children = {i: [] for i in range(N)}
    for child_idx, edge in directed_tree.items():
        children[edge.parent].append(child_idx)

    def print_node(idx, prefix=""):
        corr = corrections[idx]
        name = track_names[idx]

        if corr.parent_idx is None and not corr.is_orphan:
            # Root
            print(f"{' ' * indent}{prefix}{name} (ROOT, untouched)")
        elif corr.is_orphan:
            # Orphan
            print(f"{' ' * indent}{prefix}{name} (ORPHAN, untouched)")
        else:
            # Child
            edge = directed_tree[idx]
            pol_str = ", INV" if corr.cumulative_polarity < 0 else ""
            print(f"{' ' * indent}{prefix}{name} ({edge.correlation:.2f}) "
                  f"→ delay={corr.cumulative_delay:+.0f}{pol_str}")

        # Print children
        for i, child in enumerate(sorted(children[idx])):
            is_last = (i == len(children[idx]) - 1)
            child_prefix = "└── " if is_last else "├── "
            print_node(child, prefix + child_prefix)

    print_node(root)

    # Print orphans
    for i in range(N):
        if corrections[i].is_orphan:
            print(f"{' ' * indent}{track_names[i]} (ORPHAN, untouched)")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    track_names: List[str],
    threshold: float = 0.15,
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation Matrix"
) -> plt.Figure:
    """
    Plot correlation matrix as heatmap with annotations.

    Cells below threshold are grayed out.
    """
    N = corr_matrix.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Create masked array for cells below threshold
    masked = np.ma.masked_where(corr_matrix < threshold, corr_matrix)

    # Plot heatmap
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='#f0f0f0')  # Gray for below threshold

    im = ax.imshow(masked, cmap=cmap, vmin=threshold, vmax=1.0, aspect='equal')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=15)

    # Set ticks
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(track_names, rotation=45, ha='right')
    ax.set_yticklabels(track_names)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            if i != j:
                val = corr_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                if val < threshold:
                    color = '#999999'
                    text = f'{val:.2f}'
                else:
                    text = f'{val:.2f}'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold' if val >= threshold else 'normal')

    # Grid
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add threshold legend
    ax.text(0.02, -0.15, f'Gray cells: below threshold ({threshold})',
            transform=ax.transAxes, fontsize=9, color='#666666')

    plt.tight_layout()
    return fig


def plot_spanning_tree(
    corr_matrix: np.ndarray,
    directed_tree: Dict[int, TreeEdge],
    corrections: Dict[int, TrackCorrection],
    root: int,
    track_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "Maximum Spanning Tree"
) -> plt.Figure:
    """
    Visualize the spanning tree structure.

    Root at top, children arranged hierarchically below.
    Edge thickness proportional to correlation.
    """
    N = len(track_names)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Build children map
    children = {i: [] for i in range(N)}
    for child_idx, edge in directed_tree.items():
        children[edge.parent].append(child_idx)

    # Position nodes using hierarchical layout (top-down)
    positions = {}

    def layout_subtree(node, x_center, x_width, y_level):
        """Recursively position nodes in hierarchical layout."""
        positions[node] = (x_center, -y_level * 2.5)

        child_list = sorted(children[node])
        if not child_list:
            return

        n_children = len(child_list)
        child_width = x_width / max(n_children, 1)

        for i, child in enumerate(child_list):
            child_x = x_center - x_width/2 + child_width * (i + 0.5)
            layout_subtree(child, child_x, child_width * 0.9, y_level + 1)

    # Start layout from root
    layout_subtree(root, 0, 8, 0)

    # Position orphans on the right
    orphans = [i for i in range(N) if corrections[i].is_orphan]
    for idx, orph in enumerate(orphans):
        positions[orph] = (5, -idx * 2.0)

    # Draw edges
    for child_idx, edge in directed_tree.items():
        x1, y1 = positions[edge.parent]
        x2, y2 = positions[child_idx]

        # Edge width based on correlation
        width = 1.5 + edge.correlation * 3

        # Draw curved arrow
        ax.annotate('', xy=(x2, y2 + 0.4), xytext=(x1, y1 - 0.4),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='#2196F3',
                        lw=width,
                        connectionstyle='arc3,rad=0.1',
                        mutation_scale=15
                    ))

        # Edge label (correlation)
        mid_x = (x1 + x2) / 2 + 0.3
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{edge.correlation:.2f}',
                ha='left', va='center',
                fontsize=9, color='#1565C0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='#1565C0', lw=0.5))

    # Draw nodes
    node_radius = 0.35
    for i in range(N):
        x, y = positions[i]
        corr = corrections[i]

        if i == root:
            color = '#4CAF50'  # Green for root
        elif corr.is_orphan:
            color = '#9E9E9E'  # Gray for orphan
        else:
            color = '#2196F3'  # Blue for aligned

        # Node circle
        circle = plt.Circle((x, y), node_radius, color=color, ec='white', lw=2.5, zorder=10)
        ax.add_patch(circle)

        # Track name (shortened to fit)
        short_name = track_names[i][:10] if len(track_names[i]) > 10 else track_names[i]
        ax.text(x, y, short_name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=11)

        # Correction info below node
        if i == root:
            ax.text(x, y - 0.55, 'ROOT', ha='center', va='top',
                    fontsize=8, color='#2E7D32', fontweight='bold')
        elif corr.is_orphan:
            ax.text(x, y - 0.55, 'ORPHAN', ha='center', va='top',
                    fontsize=8, color='#616161', fontweight='bold')
        elif corr.parent_idx is not None:
            delay_info = f'{corr.cumulative_delay:+.0f} smp'
            if corr.cumulative_polarity < 0:
                delay_info += ' INV'
            ax.text(x, y - 0.55, delay_info, ha='center', va='top',
                    fontsize=7, color='#666666')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='Root (reference)'),
        mpatches.Patch(color='#2196F3', label='Aligned'),
        mpatches.Patch(color='#9E9E9E', label='Orphan (untouched)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Auto-scale axes
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_alignment_overview(
    corr_matrix: np.ndarray,
    directed_tree: Dict[int, TreeEdge],
    corrections: Dict[int, TrackCorrection],
    root: int,
    track_names: List[str],
    threshold: float = 0.15,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Combined overview plot: matrix + tree side by side.
    """
    fig = plt.figure(figsize=(16, 7))

    # Left: correlation matrix
    ax1 = fig.add_subplot(1, 2, 1)
    plot_correlation_matrix(corr_matrix, track_names, threshold, ax=ax1,
                            title="All-Pairs Correlation Matrix")

    # Right: spanning tree
    ax2 = fig.add_subplot(1, 2, 2)
    plot_spanning_tree(corr_matrix, directed_tree, corrections, root,
                       track_names, ax=ax2, title="Maximum Spanning Tree")

    # Add row-sums table below matrix
    N = len(track_names)
    row_sums = []
    for i in range(N):
        s = sum(corr_matrix[i, j] for j in range(N)
                if i != j and corr_matrix[i, j] >= threshold)
        row_sums.append(s)

    # Sort by row sum
    sorted_idx = np.argsort(row_sums)[::-1]
    table_text = "Row sums (≥{:.2f}): ".format(threshold)
    table_text += " | ".join(f"{track_names[i]}: {row_sums[i]:.2f}" for i in sorted_idx)
    fig.text(0.5, 0.02, table_text, ha='center', fontsize=9, color='#666666')

    plt.suptitle("Magic Phase - Spanning Tree Alignment", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing spanning tree with synthetic correlation matrix...")

    # From the spec example
    track_names = ["SnTop", "Kick", "SnBot", "OH_L", "OH_R"]
    corr_matrix = np.array([
        [0.00, 0.08, 0.62, 0.40, 0.35],
        [0.08, 0.00, 0.08, 0.09, 0.13],
        [0.62, 0.08, 0.00, 0.32, 0.35],
        [0.40, 0.09, 0.32, 0.00, 0.22],
        [0.35, 0.13, 0.35, 0.22, 0.00],
    ])
    delay_matrix = np.zeros((5, 5))
    polarity_matrix = np.ones((5, 5), dtype=int)

    tree_edges = max_spanning_tree(corr_matrix, threshold=0.15)
    root = select_root(corr_matrix, threshold=0.15)
    directed_tree = build_directed_tree(tree_edges, root, delay_matrix, polarity_matrix)
    corrections = compute_corrections(directed_tree, root, 5)

    print(f"\nRoot: {track_names[root]}")
    print(f"Tree edges: {tree_edges}")
    print(f"\nCorrections:")
    for i, c in corrections.items():
        print(f"  {track_names[i]}: orphan={c.is_orphan}, parent={c.parent_idx}")

    # Plot
    fig = plot_alignment_overview(
        corr_matrix, directed_tree, corrections, root, track_names,
        threshold=0.15
    )
    plt.show()
