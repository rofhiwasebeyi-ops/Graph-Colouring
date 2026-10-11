"""
Regenerates convergence_history.json using the actual best-achieved k per
graph/method (from results.csv), rather than the deepest-searched k (which
may have been a failed attempt). One representative run per graph/method,
run directly at the known-good k with track_history=True.
"""

import json
import pandas as pd
from graph import load_dimacs
from ga import GAColourer
from aco import ACOColourer
from params import load_best_params, output_path

PARAMS = load_best_params()

df = pd.read_csv(output_path("results.csv"))
best_k = (
    df[df.method.isin(["GA", "ACO"])]
    .dropna(subset=["n_colours"])
    .groupby(["graph", "method"])["n_colours"].min()
)

INSTANCE_FILES = {
    "myciel3.col": "instances/myciel3.col",
    "myciel4.col": "instances/myciel4.col",
    "myciel5.col": "instances/myciel5.col",
    "myciel6.col": "instances/myciel6.col",
    "queen5_5.col": "instances/queen5_5.col",
    "queen6_6.col": "instances/queen6_6.col",
    "queen8_8.col": "instances/queen8_8.col",
}

history = {}
for graph_name, path in INSTANCE_FILES.items():
    g = load_dimacs(path)
    history[graph_name] = {}
    print(f"\n=== {graph_name} ===")

    k_ga = int(best_k.get((graph_name, "GA"), None))
    ga = GAColourer(g, **PARAMS["GA"], generations=500, seed=1000)
    _, conflicts, hist = ga.run_for_k(k_ga, track_history=True)
    history[graph_name]["GA"] = hist
    print(f"GA at k={k_ga}: final conflicts={conflicts}, {len(hist)} generations logged")

    k_aco = int(best_k.get((graph_name, "ACO"), None))
    aco = ACOColourer(g, **PARAMS["ACO"], generations=350, seed=1000)
    _, conflicts, hist = aco.run_for_k(k_aco, track_history=True)
    history[graph_name]["ACO"] = hist
    print(f"ACO at k={k_aco}: final conflicts={conflicts}, {len(hist)} iterations logged")

with open(output_path("convergence_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print("\nSaved corrected convergence_history.json")
