import numpy as np
import pytest
from graph import Graph, load_dimacs, random_graph


def test_basic_construction():
    g = Graph(4, [(0, 1), (1, 2), (2, 3)])
    assert g.n == 4
    assert g.m == 3


def test_duplicate_and_reversed_edges_deduplicated():
    # (0,1) and (1,0) are the same edge; should only be counted once
    g = Graph(3, [(0, 1), (1, 0), (0, 1)])
    assert g.m == 1


def test_self_loops_ignored():
    g = Graph(3, [(0, 0), (0, 1)])
    assert g.m == 1


def test_density_complete_graph():
    n = 5
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    g = Graph(n, edges)
    assert g.density == pytest.approx(1.0)


def test_density_empty_graph():
    g = Graph(5, [])
    assert g.density == 0.0
    assert g.m == 0


def test_degree_and_adjacency():
    g = Graph(4, [(0, 1), (0, 2), (0, 3)])
    assert g.degree(0) == 3
    assert g.degree(1) == 1
    assert set(g.adj_list[0].tolist()) == {1, 2, 3}


def test_conflicts_all_same_colour():
    g = Graph(3, [(0, 1), (1, 2)])
    colouring = np.array([1, 1, 1])
    assert g.conflicts(colouring) == 2  # both edges conflict


def test_conflicts_valid_colouring():
    g = Graph(3, [(0, 1), (1, 2)])
    colouring = np.array([0, 1, 0])
    assert g.conflicts(colouring) == 0
    assert g.is_valid_colouring(colouring)


def test_conflicts_no_edges():
    g = Graph(3, [])
    colouring = np.array([0, 0, 0])
    assert g.conflicts(colouring) == 0


def test_n_colours_used():
    g = Graph(4, [(0, 1)])
    assert g.n_colours_used(np.array([0, 1, 2, 2])) == 3


def test_load_dimacs_parses_correctly():
    g = load_dimacs("instances/myciel3.col")
    assert g.n == 11
    assert g.m == 20


def test_load_dimacs_is_1_indexed_converted_to_0_indexed():
    # myciel3.col's first edge line is "e 1 2" -> should become (0, 1)
    g = load_dimacs("instances/myciel3.col")
    assert (0, 1) in g.edges


def test_random_graph_reproducible_with_seed():
    g1 = random_graph(20, 0.3, seed=42)
    g2 = random_graph(20, 0.3, seed=42)
    assert g1.edges == g2.edges


def test_random_graph_density_approximately_matches_target():
    g = random_graph(200, 0.3, seed=1)
    # with 200 vertices the empirical density should be close to target
    assert abs(g.density - 0.3) < 0.02


def test_random_graph_different_seeds_differ():
    g1 = random_graph(20, 0.3, seed=1)
    g2 = random_graph(20, 0.3, seed=2)
    assert g1.edges != g2.edges
