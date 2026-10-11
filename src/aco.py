"""
Ant Colony Optimization for Graph Colouring — Swarm Intelligence arm.

Based on the ANTCOL approach (Costa & Hertz, 1997): each ant constructs a
full colouring by visiting vertices in DSATUR-style saturation order and
choosing a colour probabilistically, weighted by:
    - pheromone[v, c]        : learned desirability of colour c at vertex v
    - heuristic desirability : favours colours that create fewer conflicts
                                with already-coloured neighbours

After each generation (a batch of ants), pheromone evaporates and the best
ant(s) deposit pheromone proportional to solution quality. This gives ACO
the same "population of candidate solutions, iterative improvement" shape
as the GA, making the two directly comparable.
"""

import numpy as np
import time


class ACOColourer:
    def __init__(
        self,
        graph,
        n_ants=45,
        generations=200,
        alpha=1.0,      # pheromone influence
        beta=3.0,       # heuristic influence
        evaporation=0.1,
        q=1.0,           # pheromone deposit strength
        seed=None,
    ):
        self.g = graph
        self.n_ants = n_ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        self.rng = np.random.default_rng(seed)

    def _construct_solution(self, pheromone, k):
        """One ant builds a full colouring for k available colours."""
        n = self.g.n
        colouring = np.full(n, -1, dtype=int)
        uncoloured = set(range(n))
        neighbour_colour_count = np.zeros((n, k), dtype=int)  # conflicts each colour would cause

        while uncoloured:
            # dynamic saturation: vertex with most distinct neighbour colours used so far
            def saturation(v):
                return np.count_nonzero(neighbour_colour_count[v])

            v = max(uncoloured, key=lambda x: (saturation(x), self.g.degree(x)))

            # heuristic desirability: fewer expected conflicts -> higher desirability
            conflict_counts = neighbour_colour_count[v]
            heuristic = 1.0 / (1.0 + conflict_counts)

            tau = pheromone[v] ** self.alpha
            eta = heuristic ** self.beta
            weights = tau * eta
            probs = weights / weights.sum()

            colour = self.rng.choice(k, p=probs)
            colouring[v] = colour
            uncoloured.remove(v)

            for u in self.g.adj_list[v]:
                if colouring[u] == -1:
                    neighbour_colour_count[u, colour] += 1

        return colouring

    def run_for_k(self, k, track_history=False):
        pheromone = np.ones((self.g.n, k))
        history = []
        best_colouring, best_conflicts = None, np.inf

        for gen in range(self.generations):
            ants = [self._construct_solution(pheromone, k) for _ in range(self.n_ants)]
            conflicts = np.array([self.g.conflicts(a) for a in ants])

            gen_best_idx = np.argmin(conflicts)
            if conflicts[gen_best_idx] < best_conflicts:
                best_conflicts = conflicts[gen_best_idx]
                best_colouring = ants[gen_best_idx].copy()

            if track_history:
                history.append(int(conflicts.min()))

            if best_conflicts == 0:
                break

            # evaporation
            pheromone *= (1 - self.evaporation)

            # deposit: best ants reinforce the (vertex, colour) choices they made
            order = np.argsort(conflicts)[: max(1, self.n_ants // 5)]  # top 20%
            for idx in order:
                ant = ants[idx]
                deposit = self.q / (1 + conflicts[idx])
                for v in range(self.g.n):
                    pheromone[v, ant[v]] += deposit

        return best_colouring, int(best_conflicts), history

    def find_minimum_colouring(self, k_start, k_min=1, track_history=False):
        start = time.perf_counter()
        best_colouring, best_k = None, None
        all_history = {}

        k = k_start
        while k >= k_min:
            colouring, conflicts, history = self.run_for_k(k, track_history=track_history)
            all_history[k] = history
            if conflicts == 0:
                best_colouring, best_k = colouring, k
                k -= 1
            else:
                break

        runtime = time.perf_counter() - start
        stats = {
            "n_colours": best_k,
            "conflicts": 0 if best_colouring is not None else None,
            "runtime_s": runtime,
            "history": all_history,
        }
        return best_colouring, stats


if __name__ == "__main__":
    from graph import random_graph
    from dsatur import dsatur

    g = random_graph(50, 0.3, seed=42)
    _, dsatur_stats = dsatur(g)
    print(g, "| DSATUR upper bound:", dsatur_stats["n_colours"])

    aco = ACOColourer(g, n_ants=20, generations=100, seed=1)
    colouring, stats = aco.find_minimum_colouring(k_start=dsatur_stats["n_colours"], track_history=True)
    print("ACO result:", {k: v for k, v in stats.items() if k != "history"})
