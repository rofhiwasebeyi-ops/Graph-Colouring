import numpy as np
from graph import Graph, load_dimacs
from dsatur import dsatur


def test_dsatur_produces_valid_colouring():
    g = load_dimacs("instances/myciel4.col")
    colouring, stats = dsatur(g)
    assert stats["conflicts"] == 0
    assert g.is_valid_colouring(colouring)


def test_dsatur_deterministic():
    g = load_dimacs("instances/myciel4.col")
    c1, _ = dsatur(g)
    c2, _ = dsatur(g)
    assert np.array_equal(c1, c2)


def test_dsatur_on_triangle_needs_3_colours():
    # A triangle (K3) needs exactly 3 colours
    g = Graph(3, [(0, 1), (1, 2), (0, 2)])
    colouring, stats = dsatur(g)
    assert stats["n_colours"] == 3
    assert stats["conflicts"] == 0


def test_dsatur_on_empty_graph_uses_1_colour():
    g = Graph(4, [])
    colouring, stats = dsatur(g)
    assert stats["n_colours"] == 1
    assert stats["conflicts"] == 0


def test_dsatur_on_bipartite_graph_uses_2_colours():
    # A 4-cycle (0-1-2-3-0) is bipartite, chromatic number 2
    g = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    colouring, stats = dsatur(g)
    assert stats["n_colours"] == 2
    assert stats["conflicts"] == 0


def test_dsatur_colours_every_vertex():
    g = load_dimacs("instances/queen5_5.col")
    colouring, _ = dsatur(g)
    assert len(colouring) == g.n
    assert np.all(colouring >= 0)  # no vertex left uncoloured
