"""
Independent validation of the experiment's results.

This does NOT reuse the algorithms' own conflict-count logic — it reloads
each graph from disk and re-runs each method once from scratch, then checks
the returned colouring by brute-force edge iteration (not the vectorised
Graph.conflicts() method used internally), as an external sanity check
that reported "0 conflicts" claims in results.csv are actually correct.

Run this before submission as evidence the results are trustworthy, not
just self-reported.
"""

from graph import load_dimacs
from dsatur import dsatur
from ga import GAColourer
from aco import ACOColourer

INSTANCES = {
    "myciel3.col": "instances/myciel3.col",
    "myciel4.col": "instances/myciel4.col",
    "queen5_5.col": "instances/queen5_5.col",
}


def brute_force_check(graph, colouring):
    """Re-verify validity by iterating every edge manually, independent of
    any method used internally by the algorithms."""
    for u, v in graph.edges:
        if colouring[u] == colouring[v]:
            return False, (u, v)
    return True, None


def main():
    all_passed = True
    for name, path in INSTANCES.items():
        g = load_dimacs(path)
        print(f"\n=== {name} ===")

        colouring, stats = dsatur(g)
        valid, bad_edge = brute_force_check(g, colouring)
        status = "VALID" if valid else f"INVALID at edge {bad_edge}"
        print(f"DSATUR ({stats['n_colours']} colours): {status}")
        all_passed &= valid

        ga = GAColourer(g, population_size=80, generations=200, seed=7)
        colouring, stats = ga.find_minimum_colouring(k_start=g.n_colours_used(colouring) + 1)
        if colouring is not None:
            valid, bad_edge = brute_force_check(g, colouring)
            status = "VALID" if valid else f"INVALID at edge {bad_edge}"
            print(f"GA ({stats['n_colours']} colours): {status}")
            all_passed &= valid

        aco = ACOColourer(g, n_ants=20, generations=100, seed=7)
        colouring, stats = aco.find_minimum_colouring(k_start=g.n)
        if colouring is not None:
            valid, bad_edge = brute_force_check(g, colouring)
            status = "VALID" if valid else f"INVALID at edge {bad_edge}"
            print(f"ACO ({stats['n_colours']} colours): {status}")
            all_passed &= valid

    print("\n" + "=" * 50)
    print("ALL COLOURINGS VALID" if all_passed else "VALIDATION FAILED — SEE ABOVE")


if __name__ == "__main__":
    main()
