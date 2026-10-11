import numpy as np
from graph import Graph, load_dimacs
from ga import GAColourer


def test_ga_finds_valid_2_colouring_on_bipartite_graph():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    ga = GAColourer(g, population_size=30, generations=50, seed=1)
    colouring, conflicts, _ = ga.run_for_k(2)
    assert conflicts == 0
    assert g.is_valid_colouring(colouring)


def test_ga_finds_valid_3_colouring_on_triangle():
    g = Graph(3, [(0, 1), (1, 2), (0, 2)])
    ga = GAColourer(g, population_size=20, generations=50, seed=1)
    colouring, conflicts, _ = ga.run_for_k(3)
    assert conflicts == 0


def test_ga_reports_nonzero_conflicts_when_k_too_small():
    # a triangle cannot be 2-coloured
    g = Graph(3, [(0, 1), (1, 2), (0, 2)])
    ga = GAColourer(g, population_size=20, generations=30, seed=1)
    colouring, conflicts, _ = ga.run_for_k(2)
    assert conflicts > 0


def test_fitness_vectorised_matches_manual_count():
    g = Graph(4, [(0, 1), (1, 2), (2, 3)])
    ga = GAColourer(g, population_size=5, generations=1, seed=1)
    population = np.array([
        [0, 0, 1, 1],   # conflicts: (0,1) same -> 1 conflict; (1,2) diff; (2,3) same -> 1 conflict = 2
        [0, 1, 0, 1],   # no conflicts
    ])
    fitness = ga._fitness(population)
    assert fitness[0] == 2
    assert fitness[1] == 0


def test_find_minimum_colouring_does_not_exceed_k_start():
    g = load_dimacs("instances/myciel3.col")
    ga = GAColourer(g, population_size=50, generations=100, seed=1)
    colouring, stats = ga.find_minimum_colouring(k_start=5)
    if stats["n_colours"] is not None:
        assert stats["n_colours"] <= 5
        assert g.is_valid_colouring(colouring)


def test_reproducible_with_same_seed():
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    ga1 = GAColourer(g, population_size=20, generations=30, seed=42)
    ga2 = GAColourer(g, population_size=20, generations=30, seed=42)
    c1, conf1, _ = ga1.run_for_k(2)
    c2, conf2, _ = ga2.run_for_k(2)
    assert np.array_equal(c1, c2)
    assert conf1 == conf2
