"""
Generates the figures used in the report from results.csv and
convergence_history.json:

  1. colours_vs_graph.png     - grouped bar chart: colours used (mean +/- std
                                 for GA/ACO) vs DSATUR, per graph, annotated
                                 with the known chromatic number.
  2. convergence_curves.png   - GA vs ACO conflict-count convergence, one
                                 panel per graph.
  3. runtime_comparison.png   - runtime (log scale) vs graph size, per method.

Run after run_experiment.py has produced results.csv / convergence_history.json.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import output_path

df = pd.read_csv(output_path("results.csv"))
with open(output_path("convergence_history.json")) as f:
    history = json.load(f)

GRAPH_ORDER = df["graph"].unique().tolist()

# ---------------- 1. colours used vs graph ----------------
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(GRAPH_ORDER))
width = 0.25

dsatur_vals, ga_means, ga_stds, aco_means, aco_stds, chi_vals = [], [], [], [], [], []
for graph in GRAPH_ORDER:
    sub = df[df["graph"] == graph]
    dsatur_vals.append(sub[sub.method == "DSATUR"]["n_colours"].iloc[0])
    ga_vals = sub[sub.method == "GA"]["n_colours"].dropna()
    aco_vals = sub[sub.method == "ACO"]["n_colours"].dropna()
    ga_means.append(ga_vals.mean() if len(ga_vals) else np.nan)
    ga_stds.append(ga_vals.std() if len(ga_vals) else 0)
    aco_means.append(aco_vals.mean() if len(aco_vals) else np.nan)
    aco_stds.append(aco_vals.std() if len(aco_vals) else 0)
    chi_vals.append(sub["chi_known"].iloc[0])

ax.bar(x - width, dsatur_vals, width, label="DSATUR (baseline)", color="#888888")
ax.bar(x, ga_means, width, yerr=ga_stds, capsize=3, label="GA (evolutionary)", color="#4C72B0")
ax.bar(x + width, aco_means, width, yerr=aco_stds, capsize=3, label="ACO (swarm)", color="#DD8452")
ax.scatter(x, chi_vals, marker="_", s=800, color="black", linewidths=2, label="Known chromatic number (χ)")

ax.set_xticks(x)
ax.set_xticklabels(GRAPH_ORDER, rotation=30, ha="right")
ax.set_ylabel("Colours used")
ax.set_title("Colours used per method vs. known chromatic number, by benchmark graph")
ax.legend()
fig.tight_layout()
fig.savefig(output_path("colours_vs_graph.png"), dpi=150)
plt.close(fig)

# ---------------- 2. convergence curves ----------------
n_graphs = len(GRAPH_ORDER)
cols = 3
rows = int(np.ceil(n_graphs / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
axes = np.array(axes).reshape(-1)

for i, graph in enumerate(GRAPH_ORDER):
    ax = axes[i]
    hist = history.get(graph, {})
    if "GA" in hist and len(hist["GA"]) > 0:
        ax.plot(hist["GA"], label="GA", color="#4C72B0")
    if "ACO" in hist and len(hist["ACO"]) > 0:
        ax.plot(hist["ACO"], label="ACO", color="#DD8452")
    ax.set_title(graph, fontsize=10)
    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Conflicts")
    ax.legend(fontsize=8)

for j in range(n_graphs, len(axes)):
    axes[j].axis("off")

fig.suptitle("Convergence behaviour: conflicts vs. generation (best-found k per graph)")
fig.tight_layout()
fig.savefig(output_path("convergence_curves.png"), dpi=150)
plt.close(fig)

# ---------------- 3. runtime comparison ----------------
fig, ax = plt.subplots(figsize=(9, 6))
summary = df.groupby(["graph", "method"]).agg(
    n=("n", "first"), runtime_mean=("runtime_s", "mean")
).reset_index()

for method, color, marker in [("DSATUR", "#888888", "s"), ("GA", "#4C72B0", "o"), ("ACO", "#DD8452", "^")]:
    sub = summary[summary.method == method].sort_values("n")
    ax.plot(sub["n"], sub["runtime_mean"], marker=marker, color=color, label=method)

ax.set_xlabel("Graph size (vertices)")
ax.set_ylabel("Mean runtime (s)")
ax.set_yscale("log")
ax.set_title("Runtime vs. graph size, by method")
ax.legend()
fig.tight_layout()
fig.savefig(output_path("runtime_comparison.png"), dpi=150)
plt.close(fig)

print("Saved to outputs/: colours_vs_graph.png, convergence_curves.png, runtime_comparison.png")
