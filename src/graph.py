"""
Graph representation for the Graph Colouring project.

Supports:
- Loading DIMACS .col format benchmark instances (e.g. myciel3.col, queen5_5.col)
- Generating random Erdos-Renyi graphs of specified size/density, for the
  "varying size and density" arm of the comparative study.

All algorithms (DSATUR, GA, ACO) operate on this Graph object so the
comparison is apples-to-apples across methods.
"""

import numpy as np


class Graph:
    def __init__(self, n_vertices, edges, name="graph"):
        """
        n_vertices : int, number of vertices (0-indexed internally)
        edges      : iterable of (u, v) tuples, 0-indexed
        name       : identifier used in logs/results
        """
        self.n = n_vertices
        self.name = name
        self.edges = set()
        for u, v in edges:
            if u != v:
                self.edges.add((min(u, v), max(u, v)))

        # Adjacency matrix (bool) - fast conflict checks, fine for the
        # instance sizes used in this project (up to a few hundred vertices)
        self.adj_matrix = np.zeros((self.n, self.n), dtype=bool)
        for u, v in self.edges:
            self.adj_matrix[u, v] = True
            self.adj_matrix[v, u] = True

        # Adjacency list - faster iteration for construction heuristics (DSATUR, ACO)
        self.adj_list = [np.nonzero(self.adj_matrix[i])[0] for i in range(self.n)]

    @property
    def m(self):
        return len(self.edges)

    @property
    def density(self):
        max_edges = self.n * (self.n - 1) / 2
        return self.m / max_edges if max_edges > 0 else 0.0

    def degree(self, v):
        return len(self.adj_list[v])

    def degrees(self):
        return np.array([self.degree(v) for v in range(self.n)])

    def conflicts(self, colouring):
        """
        Count the number of conflicting edges for a given colouring.
        colouring : array-like of length n, colouring[v] = colour index of vertex v
        """
        colouring = np.asarray(colouring)
        u_idx = np.array([u for u, v in self.edges])
        v_idx = np.array([v for u, v in self.edges])
        if len(u_idx) == 0:
            return 0
        return int(np.sum(colouring[u_idx] == colouring[v_idx]))

    def is_valid_colouring(self, colouring):
        return self.conflicts(colouring) == 0

    def n_colours_used(self, colouring):
        return len(set(np.asarray(colouring).tolist()))

    def __repr__(self):
        return f"Graph(name={self.name}, n={self.n}, m={self.m}, density={self.density:.3f})"


def load_dimacs(path, name=None):
    """
    Load a DIMACS .col format graph.
    Format:
      c <comment lines>
      p edge <n_vertices> <n_edges>
      e <u> <v>          (1-indexed in the file)
    """
    edges = []
    n_vertices = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                n_vertices = int(parts[2])
            elif parts[0] == "e":
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # to 0-indexed
                edges.append((u, v))
    return Graph(n_vertices, edges, name=name or path.split("/")[-1])


def random_graph(n_vertices, density, seed=None, name=None):
    """
    Generate an Erdos-Renyi G(n, p) random graph with the given target edge density.
    Used to systematically vary size/density for the comparative study.
    """
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n_vertices, k=1)
    n_possible = len(upper[0])
    mask = rng.random(n_possible) < density
    edges = list(zip(upper[0][mask].tolist(), upper[1][mask].tolist()))
    label = name or f"random_n{n_vertices}_d{density:.2f}"
    return Graph(n_vertices, edges, name=label)
