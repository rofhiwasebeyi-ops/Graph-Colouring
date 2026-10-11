"""
DSATUR (Degree of Saturation) — classical greedy graph colouring heuristic.
Used as the deterministic control/baseline against GA and ACO.

Reference: Brelaz, D. (1979). "New methods to color the vertices of a graph."
Communications of the ACM, 22(4), 251-256.

Deterministic given a fixed tie-breaking rule, so it needs only ONE run per
graph instance (unlike GA/ACO, which are stochastic and need multiple runs).
"""

import numpy as np
import time


def dsatur(graph):
    """
    Returns:
        colouring : np.ndarray of length graph.n, colouring[v] = colour index (0-based)
        stats     : dict with n_colours, conflicts (always 0 for DSATUR), runtime_s
    """
    start = time.perf_counter()
    n = graph.n
    UNCOLOURED = -1
    colouring = np.full(n, UNCOLOURED, dtype=int)

    # saturation[v] = number of DISTINCT colours used by v's neighbours
    saturation = np.zeros(n, dtype=int)
    degrees = graph.degrees()
    neighbour_colours = [set() for _ in range(n)]

    uncoloured = set(range(n))

    while uncoloured:
        # Pick vertex with max saturation; break ties by max degree
        v = max(uncoloured, key=lambda x: (saturation[x], degrees[x]))

        used_by_neighbours = neighbour_colours[v]
        colour = 0
        while colour in used_by_neighbours:
            colour += 1
        colouring[v] = colour

        uncoloured.remove(v)
        for u in graph.adj_list[v]:
            if colouring[u] == UNCOLOURED:
                if colour not in neighbour_colours[u]:
                    neighbour_colours[u].add(colour)
                    saturation[u] += 1

    runtime = time.perf_counter() - start
    stats = {
        "n_colours": graph.n_colours_used(colouring),
        "conflicts": graph.conflicts(colouring),
        "runtime_s": runtime,
    }
    return colouring, stats


if __name__ == "__main__":
    from graph import random_graph

    g = random_graph(50, 0.3, seed=42)
    colouring, stats = dsatur(g)
    print(g)
    print("DSATUR result:", stats)
