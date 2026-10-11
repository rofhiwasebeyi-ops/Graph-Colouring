"""
Full experiment runner for the comparative study:

    "How do evolutionary (GA) and swarm intelligence (ACO) approaches compare
    with a classical heuristic (DSATUR) in terms of solution quality (colours
    used), convergence behaviour, and computational cost, across graph
    colouring instances of varying size and density?"

Uses real DIMACS benchmark instances (Mycielski + Queen graphs), which
naturally span a range of size (11-95 vertices) and density (0.16-0.53),
with published chromatic numbers for external validation.

Results are written incrementally to results.csv (so a partial run is
still usable), and convergence history for one representative run per
graph/method is saved to convergence_history.json for plotting.

Run: python run_experiment.py
"""

import csv
import json
import time

from graph import load_dimacs
from dsatur import dsatur
from ga import GAColourer
from aco import ACOColourer
from params import load_best_params, output_path, EXPERIMENT_INSTANCES

# Tuned hyperparameters: read from best_params.json if tuning has been run,
# otherwise the built-in defaults (printed so the run records what it used).
PARAMS = load_best_params()

# ---------------- experiment configuration ----------------
# The instance table (generation budgets and run counts) lives in params.py
# so the demo can use the same budgets.
INSTANCES = EXPERIMENT_INSTANCES

SEED_BASE = 1000
OUTPUT_CSV = output_path("results.csv")
HISTORY_JSON = output_path("convergence_history.json")
# ------------------------------------------------------------

FIELDNAMES = ["graph", "n", "m", "density", "chi_known", "method", "run",
              "n_colours", "conflicts", "runtime_s"]


def run_all():
    convergence_history = {}

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for path, chi_known, ga_gens, aco_gens, n_runs in INSTANCES:
            g = load_dimacs(path)
            print(f"\n=== {g} (known chi={chi_known}) ===")
            convergence_history[g.name] = {}

            # --- DSATUR: deterministic, single run ---
            colouring, stats = dsatur(g)
            row = {
                "graph": g.name, "n": g.n, "m": g.m, "density": round(g.density, 4),
                "chi_known": chi_known, "method": "DSATUR", "run": 0,
                "n_colours": stats["n_colours"], "conflicts": stats["conflicts"],
                "runtime_s": round(stats["runtime_s"], 4),
            }
            writer.writerow(row)
            f.flush()
            print(f"DSATUR: {stats['n_colours']} colours, {stats['runtime_s']:.4f}s")
            dsatur_k = stats["n_colours"]

            # --- GA: stochastic, n_runs repeats ---
            for run in range(n_runs):
                ga = GAColourer(g, **PARAMS["GA"], generations=ga_gens,
                                 seed=SEED_BASE + run)
                if run == 0:
                    colouring, stats = ga.find_minimum_colouring(k_start=dsatur_k, track_history=True)
                    best_k_hist = stats["history"].get(min(stats["history"].keys()), [])
                    convergence_history[g.name]["GA"] = best_k_hist
                else:
                    colouring, stats = ga.find_minimum_colouring(k_start=dsatur_k)
                row = {
                    "graph": g.name, "n": g.n, "m": g.m, "density": round(g.density, 4),
                    "chi_known": chi_known, "method": "GA", "run": run,
                    "n_colours": stats["n_colours"], "conflicts": stats["conflicts"],
                    "runtime_s": round(stats["runtime_s"], 4),
                }
                writer.writerow(row)
                f.flush()
                print(f"GA run {run}: {stats['n_colours']} colours, {stats['runtime_s']:.2f}s")

            # --- ACO: stochastic, n_runs repeats ---
            for run in range(n_runs):
                aco = ACOColourer(g, **PARAMS["ACO"], generations=aco_gens,
                                   seed=SEED_BASE + run)
                if run == 0:
                    colouring, stats = aco.find_minimum_colouring(k_start=dsatur_k, track_history=True)
                    best_k_hist = stats["history"].get(min(stats["history"].keys()), [])
                    convergence_history[g.name]["ACO"] = best_k_hist
                else:
                    colouring, stats = aco.find_minimum_colouring(k_start=dsatur_k)
                row = {
                    "graph": g.name, "n": g.n, "m": g.m, "density": round(g.density, 4),
                    "chi_known": chi_known, "method": "ACO", "run": run,
                    "n_colours": stats["n_colours"], "conflicts": stats["conflicts"],
                    "runtime_s": round(stats["runtime_s"], 4),
                }
                writer.writerow(row)
                f.flush()
                print(f"ACO run {run}: {stats['n_colours']} colours, {stats['runtime_s']:.2f}s")

    with open(HISTORY_JSON, "w") as f:
        json.dump(convergence_history, f, indent=2)

    print(f"\nResults written to {OUTPUT_CSV}")
    print(f"Convergence history written to {HISTORY_JSON}")


if __name__ == "__main__":
    start = time.time()
    run_all()
    print(f"\nTotal experiment time: {time.time() - start:.1f}s")
