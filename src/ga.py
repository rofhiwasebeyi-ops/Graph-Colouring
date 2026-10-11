"""
Genetic Algorithm for Graph Colouring — Evolutionary Intelligence arm.

Encoding: direct colour-vector, one gene per vertex, values in [0, k).
Fitness:  number of conflicting edges (to be minimised; 0 = valid k-colouring).

Strategy for finding the minimum number of colours:
    Run the GA for a fixed k. If it finds a 0-conflict colouring, try k-1.
    If it fails to reach 0 conflicts within the generation budget, k is
    reported as the best achievable and the search stops (mirrors how
    ACO/DSATUR results are compared: "fewest colours with zero conflicts").

Conflict-directed mutation (mutate vertices that are currently in conflict,
picking a colour that reduces local conflicts) converges much faster than
pure random mutation and is standard practice for GA-based colouring.
"""

import numpy as np
import time


class GAColourer:
    def __init__(
        self,
        graph,
        population_size=100,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.05,
        tournament_size=3,
        elitism=2,
        seed=None,
    ):
        self.g = graph
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.rng = np.random.default_rng(seed)

    # ---------- core GA operators ----------

    def _init_population(self, k):
        return self.rng.integers(0, k, size=(self.pop_size, self.g.n))

    def _fitness(self, population):
        """Vectorised conflict count for a whole population. Lower is better."""
        edges = np.array(list(self.g.edges))
        if len(edges) == 0:
            return np.zeros(len(population), dtype=int)
        u_idx, v_idx = edges[:, 0], edges[:, 1]
        # population: (pop_size, n) -> compare colours at u and v for every individual
        return np.sum(population[:, u_idx] == population[:, v_idx], axis=1)

    def _tournament_select(self, population, fitness):
        idx = self.rng.integers(0, len(population), size=self.tournament_size)
        best = idx[np.argmin(fitness[idx])]
        return population[best]

    def _crossover(self, p1, p2):
        if self.rng.random() > self.crossover_rate:
            return p1.copy()
        mask = self.rng.random(self.g.n) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _conflict_directed_mutate(self, individual, k):
        """For each vertex in conflict, reassign to the colour that minimises
        conflicts with its neighbours (greedy local repair), with some
        random exploration controlled by mutation_rate."""
        child = individual.copy()
        for v in range(self.g.n):
            neighbours = self.g.adj_list[v]
            if len(neighbours) == 0:
                continue
            in_conflict = np.any(child[neighbours] == child[v])
            if in_conflict and self.rng.random() < self.mutation_rate * 5:
                # try each colour, pick the one with fewest neighbour conflicts
                neighbour_colours = child[neighbours]
                counts = np.bincount(neighbour_colours, minlength=k)
                child[v] = np.argmin(counts)
            elif self.rng.random() < self.mutation_rate:
                child[v] = self.rng.integers(0, k)
        return child

    # ---------- public API ----------

    def run_for_k(self, k, track_history=True):
        """Run the GA targeting a k-colouring. Returns best colouring, conflicts, history."""
        population = self._init_population(k)
        history = []
        best_individual, best_fit = None, np.inf

        for gen in range(self.generations):
            fitness = self._fitness(population)
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            if track_history:
                history.append(int(fitness.min()))

            if best_fit == 0:
                break

            # elitism
            elite_idx = np.argsort(fitness)[: self.elitism]
            new_population = [population[i].copy() for i in elite_idx]

            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                child = self._crossover(p1, p2)
                child = self._conflict_directed_mutate(child, k)
                new_population.append(child)

            population = np.array(new_population)

        return best_individual, int(best_fit), history

    def find_minimum_colouring(self, k_start, k_min=1, track_history=True):
        """
        Decrement k from k_start, keeping the best valid (0-conflict) result.
        Stops as soon as a k fails to reach 0 conflicts within the budget.
        Returns dict with best colouring found, n_colours, runtime, history per k.
        """
        start = time.perf_counter()
        best_colouring, best_k = None, None
        all_history = {}

        k = k_start
        while k >= k_min:
            colouring, conflicts, history = self.run_for_k(k, track_history=track_history)
            all_history[k] = history
            if conflicts == 0:
                best_colouring, best_k = colouring, k
                k -= 1  # try to do better
            else:
                break  # this k failed within budget; stop here

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

    ga = GAColourer(g, population_size=80, generations=200, seed=1)
    colouring, stats = ga.find_minimum_colouring(k_start=dsatur_stats["n_colours"], track_history=True)
    print("GA result:", {k: v for k, v in stats.items() if k != "history"})
