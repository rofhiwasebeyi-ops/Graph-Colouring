"""
Formal hyperparameter tuning via grid search.

Tunes GA (population_size x mutation_rate) and ACO (n_ants x beta)
independently, on two moderate-difficulty validation instances
(myciel5.col, target k=6; queen6_6.col, target k=8 — the best colour
count both algorithms achieved on it in the main experiment), holding
target k fixed so operator quality is isolated from the k-search
strategy.

For each hyperparameter combination, 3 repeats are run and the metric
is: success rate (fraction reaching 0 conflicts within the generation
budget) and, among successful runs, the mean generation/iteration at
which 0 conflicts was first reached (lower = faster convergence).

Selection rule: highest success rate first, mean convergence speed as
tie-breaker. The full grids are written to tuning_ga.csv and tuning_aco.csv,
the winners to tuning_winners.csv, and the winning parameters to
best_params.json, which the experiment scripts load automatically. All of
these are written to the outputs/ folder.
"""

import csv
import itertools
import numpy as np

from graph import load_dimacs
from ga import GAColourer
from aco import ACOColourer
from params import save_best_params, output_path

VALIDATION_SET = [
    ("instances/myciel5.col", 6),
    ("instances/queen6_6.col", 8),
]

N_REPEATS = 3
GA_GENERATIONS = 200
ACO_GENERATIONS = 150

GA_GRID = {
    "population_size": [50, 100, 150],
    "mutation_rate": [0.05, 0.1, 0.2],
}

ACO_GRID = {
    "n_ants": [15, 30, 45],
    "beta": [1.0, 2.0, 3.0],
}


def convergence_gen(history):
    """First index at which conflicts hit 0, or None if never."""
    for i, c in enumerate(history):
        if c == 0:
            return i
    return None


def tune_ga():
    rows = []
    combos = list(itertools.product(GA_GRID["population_size"], GA_GRID["mutation_rate"]))
    for pop_size, mut_rate in combos:
        successes, conv_gens = 0, []
        for path, k in VALIDATION_SET:
            g = load_dimacs(path)
            for rep in range(N_REPEATS):
                ga = GAColourer(g, population_size=pop_size, generations=GA_GENERATIONS,
                                 mutation_rate=mut_rate, seed=2000 + rep)
                _, conflicts, history = ga.run_for_k(k, track_history=True)
                if conflicts == 0:
                    successes += 1
                    cg = convergence_gen(history)
                    if cg is not None:
                        conv_gens.append(cg)
        total_runs = len(VALIDATION_SET) * N_REPEATS
        row = {
            "population_size": pop_size, "mutation_rate": mut_rate,
            "success_rate": successes / total_runs,
            "mean_convergence_gen": float(np.mean(conv_gens)) if conv_gens else None,
        }
        rows.append(row)
        print("GA", row)
    return rows


def tune_aco():
    rows = []
    combos = list(itertools.product(ACO_GRID["n_ants"], ACO_GRID["beta"]))
    for n_ants, beta in combos:
        successes, conv_iters = 0, []
        for path, k in VALIDATION_SET:
            g = load_dimacs(path)
            for rep in range(N_REPEATS):
                aco = ACOColourer(g, n_ants=n_ants, generations=ACO_GENERATIONS,
                                    beta=beta, seed=2000 + rep)
                _, conflicts, history = aco.run_for_k(k, track_history=True)
                if conflicts == 0:
                    successes += 1
                    cg = convergence_gen(history)
                    if cg is not None:
                        conv_iters.append(cg)
        total_runs = len(VALIDATION_SET) * N_REPEATS
        row = {
            "n_ants": n_ants, "beta": beta,
            "success_rate": successes / total_runs,
            "mean_convergence_iter": float(np.mean(conv_iters)) if conv_iters else None,
        }
        rows.append(row)
        print("ACO", row)
    return rows


def pick_best(rows, conv_key):
    """Highest success rate, then fastest mean convergence as tie-break."""
    def sort_key(r):
        conv = r[conv_key] if r[conv_key] is not None else float("inf")
        return (-r["success_rate"], conv)
    return sorted(rows, key=sort_key)[0]


if __name__ == "__main__":
    print("=== Tuning GA (population_size x mutation_rate) ===")
    ga_rows = tune_ga()
    with open(output_path("tuning_ga.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["population_size", "mutation_rate", "success_rate", "mean_convergence_gen"])
        writer.writeheader()
        writer.writerows(ga_rows)

    print("\n=== Tuning ACO (n_ants x beta) ===")
    aco_rows = tune_aco()
    with open(output_path("tuning_aco.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_ants", "beta", "success_rate", "mean_convergence_iter"])
        writer.writeheader()
        writer.writerows(aco_rows)

    best_ga = pick_best(ga_rows, "mean_convergence_gen")
    best_aco = pick_best(aco_rows, "mean_convergence_iter")
    print("\nBest GA config:", best_ga)
    print("Best ACO config:", best_aco)

    with open(output_path("tuning_winners.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "parameters", "success_rate", "mean_convergence"])
        writer.writerow(["GA", f"population_size={best_ga['population_size']}, mutation_rate={best_ga['mutation_rate']}",
                          best_ga["success_rate"], best_ga["mean_convergence_gen"]])
        writer.writerow(["ACO", f"n_ants={best_aco['n_ants']}, beta={best_aco['beta']}",
                          best_aco["success_rate"], best_aco["mean_convergence_iter"]])

    saved = save_best_params(best_ga, best_aco)
    print("\nSaved outputs/best_params.json:", saved)
