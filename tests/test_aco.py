import numpy as np
from graph import Graph, load_dimacs
from aco import ACOColourer


def test_aco_finds_valid_2_colouring_on_bipartite_graph():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    aco = ACOColourer(g, n_ants=15, generations=50, seed=1)
    colouring, conflicts, _ = aco.run_for_k(2)
    assert conflicts == 0
    assert g.is_valid_colouring(colouring)


def test_aco_finds_valid_3_colouring_on_triangle():
    g = Graph(3, [(0, 1), (1, 2), (0, 2)])
    aco = ACOColourer(g, n_ants=15, generations=50, seed=1)
    colouring, conflicts, _ = aco.run_for_k(3)
    assert conflicts == 0


def test_aco_reports_nonzero_conflicts_when_k_too_small():
    g = Graph(3, [(0, 1), (1, 2), (0, 2)])
    aco = ACOColourer(g, n_ants=15, generations=30, seed=1)
    colouring, conflicts, _ = aco.run_for_k(2)
    assert conflicts > 0


def test_construct_solution_colours_every_vertex():
    g = load_dimacs("instances/myciel3.col")
    aco = ACOColourer(g, n_ants=10, generations=10, seed=1)
    pheromone = np.ones((g.n, 5))
    colouring = aco._construct_solution(pheromone, 5)
    assert len(colouring) == g.n
    assert np.all(colouring >= 0)
    assert np.all(colouring < 5)


def test_find_minimum_colouring_does_not_exceed_k_start():
    g = load_dimacs("instances/myciel3.col")
    aco = ACOColourer(g, n_ants=15, generations=60, seed=1)
    colouring, stats = aco.find_minimum_colouring(k_start=5)
    if stats["n_colours"] is not None:
        assert stats["n_colours"] <= 5
        assert g.is_valid_colouring(colouring)


def test_reproducible_with_same_seed():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    aco1 = ACOColourer(g, n_ants=10, generations=20, seed=42)
    aco2 = ACOColourer(g, n_ants=10, generations=20, seed=42)
    c1, conf1, _ = aco1.run_for_k(2)
    c2, conf2, _ = aco2.run_for_k(2)
    assert np.array_equal(c1, c2)
    assert conf1 == conf2
