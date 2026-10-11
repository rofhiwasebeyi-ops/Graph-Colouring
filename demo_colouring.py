"""
Visual demonstration for the Graph Colouring project.

Run from the project root:
    python demo_colouring.py
    python demo_colouring.py --instance 5
    python demo_colouring.py --instance queen5_5.col
    python demo_colouring.py --instance 6 --quick     # smaller budget, faster

Output:
    outputs/demo_<instance>.png
"""

from pathlib import Path
import sys
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Make the project's src/ modules importable.
ROOT = Path(__file__).resolve().parent
# Also support keeping this script in a separate folder beside the project.
if not (ROOT / "instances").exists() and (ROOT / "Graph colouring" / "instances").exists():
    ROOT = ROOT / "Graph colouring"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from graph import load_dimacs
from dsatur import dsatur
from ga import GAColourer
from aco import ACOColourer
from params import load_best_params, output_path, OUTPUT_DIR, EXPERIMENT_INSTANCES


# ---------------- configuration ----------------

INSTANCES = {
    "1": ("myciel3.col", 4),
    "2": ("myciel4.col", 5),
    "3": ("myciel5.col", 6),
    "4": ("myciel6.col", 7),
    "5": ("queen5_5.col", 5),
    "6": ("queen6_6.col", 7),
    "7": ("queen8_8.col", 9),
}

SEED = 7

# Hyperparameters come from best_params.json (or the tuned defaults), and the
# generation budgets from the same table as the full experiment, so the demo
# matches the reported results. Use --quick for a fast, reduced-budget run.
QUICK_GENERATIONS = 100   # budget used by --quick (fast, but weaker on hard graphs)

OUTPUT = None  # set in main(): outputs/demo_<instance>.png

# Neutral, presentation-friendly palette.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD", "#1B9E77", "#7F3C8D",
    "#E6AB02", "#2F4858", "#A6761D", "#66C2A5",
]

UNCOLOURED = "#D9D9D9"

# ------------------------------------------------


def circular_layout(n):
    """Place vertices evenly around a circle."""
    if n == 1:
        return np.array([[0.0, 0.0]])

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def draw_graph(ax, graph, colouring, title, stats, highlight=None, reference=None):
    """Draw one coloured graph using only matplotlib."""
    pos = circular_layout(graph.n)

    # Draw edges first.
    for u, v in sorted(graph.edges):
        x = [pos[u, 0], pos[v, 0]]
        y = [pos[u, 1], pos[v, 1]]

        # Highlight an edge if it is currently conflicting.
        conflict = colouring is not None and colouring[u] == colouring[v]
        ax.plot(
            x,
            y,
            linewidth=2.4 if conflict else 1.0,
            color="#C0392B" if conflict else "#B0B0B0",
            zorder=1,
        )

    # Draw vertices.
    for v in range(graph.n):
        c = int(colouring[v])
        face = PALETTE[c % len(PALETTE)]

        if highlight == v:
            edge = "#111111"
            linewidth = 3.5
            radius = 0.085
        else:
            edge = "#333333"
            linewidth = 1.2
            radius = 0.075

        ax.add_patch(
            Circle(
                pos[v],
                radius,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                zorder=3,
            )
        )

        ax.text(
            pos[v, 0],
            pos[v, 1],
            str(v + 1),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=26)
    ax.text(
        0.5,
        1.01,
        (
            f"{stats['n_colours']} colours  |  "
            f"{stats['conflicts']} conflicts  |  "
            f"{stats['runtime_s']:.3f} s"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    if reference:
        ax.text(
            0.5,
            -0.02,
            reference,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="#444444",
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")


def load_experiment_summary(graph_name):
    """
    Summarise this instance's rows from results.csv (the full experiment).
    Returns {"DSATUR": {...}, "GA": {...}, "ACO": {...}} with mean/std colours,
    number of runs and mean runtime, or None if results.csv is missing or has
    no rows for this instance. The file holds numbers only, not colourings,
    so it can annotate the demo but not draw the graphs.
    """
    path = OUTPUT_DIR / "results.csv"
    if not path.exists():
        return None

    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["graph"] != graph_name or not r["n_colours"]:
                continue
            rows.setdefault(r["method"], []).append(
                (float(r["n_colours"]), float(r["runtime_s"]))
            )
    if not rows:
        return None

    summary = {}
    for method, vals in rows.items():
        colours = np.array([v[0] for v in vals])
        times = np.array([v[1] for v in vals])
        summary[method] = {
            "mean": colours.mean(),
            "std": colours.std(),
            "runs": len(vals),
            "time": times.mean(),
        }
    return summary


def reference_text(summary, method):
    """One line describing the full-experiment result for a method."""
    if not summary or method not in summary:
        return None
    r = summary[method]
    if r["runs"] == 1:
        return f"Full experiment: {r['mean']:.0f} colours"
    return (
        f"Full experiment ({r['runs']} runs): "
        f"{r['mean']:.2f} \u00b1 {r['std']:.2f} colours, {r['time']:.1f} s avg"
    )


def get_successful_history(history_dict):
    """
    Return the convergence history for the smallest successful k.

    GA/ACO try progressively smaller k values. The final successful
    k is the useful one to show in the demonstration.
    """
    successful = [
        k for k, history in history_dict.items()
        if history and history[-1] == 0
    ]

    if not successful:
        # Fallback: show the history with the fewest final conflicts.
        k = min(history_dict, key=lambda x: history_dict[x][-1])
    else:
        k = min(successful)

    return k, history_dict[k]


def choose_instance():
    parser = argparse.ArgumentParser(
        description="Visually compare DSATUR, GA and ACO on a graph-colouring instance."
    )
    parser.add_argument(
        "--instance",
        help="Instance number (1-7) or filename, e.g. queen5_5.col",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small generation budget (fast, but may not improve on DSATUR for hard graphs)",
    )
    args = parser.parse_args()
    choice = args.instance

    if choice is None:
        print("\nChoose a graph-colouring instance:")
        for number, (filename, chi) in INSTANCES.items():
            print(f"  {number}. {filename} (known chromatic number: {chi})")
        choice = input("Enter 1-7 or a filename: ").strip()

    if choice in INSTANCES:
        filename, chi = INSTANCES[choice]
    else:
        filename = Path(choice).name
        matches = [item for item in INSTANCES.values() if item[0] == filename]
        if not matches:
            valid = ", ".join(item[0] for item in INSTANCES.values())
            raise SystemExit(f"Unknown instance '{choice}'. Choose one of: {valid}")
        _, chi = matches[0]

    instance_path = ROOT / "instances" / filename
    if not instance_path.exists():
        raise FileNotFoundError(f"Could not find {instance_path}")
    return instance_path, chi, args.quick


def main():
    global OUTPUT
    instance_path, known_chromatic_number, quick = choose_instance()
    OUTPUT = output_path(f"demo_{instance_path.stem}.png")
    graph = load_dimacs(str(instance_path))
    PARAMS = load_best_params()

    # Generation budgets: same as the full experiment unless --quick.
    budgets = {Path(f).name: (g, a) for f, _, g, a, _ in EXPERIMENT_INSTANCES}
    if quick or instance_path.name not in budgets:
        ga_generations = aco_generations = QUICK_GENERATIONS
    else:
        ga_generations, aco_generations = budgets[instance_path.name]
    print(f"Generation budget: GA {ga_generations}, ACO {aco_generations}"
          f"{' (--quick)' if quick else ''}")
    summary = load_experiment_summary(graph.name)
    if summary is None:
        print("(No results.csv rows for this instance - showing the live run only.)")

    print(f"\nGraph: {graph.name}")
    print(f"Vertices: {graph.n}")
    print(f"Edges: {graph.m}")
    print(f"Density: {graph.density:.3f}")
    print(f"Known chromatic number: {known_chromatic_number}")

    # -------------------------------------------------
    # 1. Run the actual DSATUR implementation.
    # -------------------------------------------------
    dsatur_colouring, dsatur_stats = dsatur(graph)

    # Use DSATUR's result as the upper bound, exactly as in the
    # main experimental pipeline.
    k_start = dsatur_stats["n_colours"]

    # -------------------------------------------------
    # 2. Run the actual GA implementation.
    # -------------------------------------------------
    ga = GAColourer(
        graph,
        **PARAMS["GA"],
        generations=ga_generations,
        seed=SEED,
    )

    ga_colouring, ga_stats = ga.find_minimum_colouring(
        k_start=k_start,
        track_history=True,
    )

    # -------------------------------------------------
    # 3. Run the actual ACO implementation.
    # -------------------------------------------------
    aco = ACOColourer(
        graph,
        **PARAMS["ACO"],
        generations=aco_generations,
        seed=SEED,
    )

    aco_colouring, aco_stats = aco.find_minimum_colouring(
        k_start=k_start,
        track_history=True,
    )

    # With the reduced demo budget, GA/ACO can fail even at DSATUR's k on
    # the hardest graphs. Show DSATUR's colouring in that case, and say so.
    ga_note = aco_note = ""
    if ga_colouring is None:
        ga_colouring = dsatur_colouring.copy()
        ga_stats = dict(ga_stats, n_colours=k_start, conflicts=0)
        ga_note = " (no improvement - DSATUR colouring shown)"
    if aco_colouring is None:
        aco_colouring = dsatur_colouring.copy()
        aco_stats = dict(aco_stats, n_colours=k_start, conflicts=0)
        aco_note = " (no improvement - DSATUR colouring shown)"

    # -------------------------------------------------
    # Print results for the live demonstration.
    # -------------------------------------------------
    print("\nResults")
    print("-" * 60)
    print(
        f"DSATUR : {dsatur_stats['n_colours']} colours | "
        f"{dsatur_stats['conflicts']} conflicts | "
        f"{dsatur_stats['runtime_s']:.4f} s"
    )
    print(
        f"GA     : {ga_stats['n_colours']} colours | "
        f"{ga_stats['conflicts']} conflicts | "
        f"{ga_stats['runtime_s']:.4f} s{ga_note}"
    )
    print(
        f"ACO    : {aco_stats['n_colours']} colours | "
        f"{aco_stats['conflicts']} conflicts | "
        f"{aco_stats['runtime_s']:.4f} s{aco_note}"
    )
    print(f"Known chromatic number: {known_chromatic_number}")
    if summary:
        print("\nFull experiment (results.csv)")
        print("-" * 60)
        for method in ("DSATUR", "GA", "ACO"):
            line = reference_text(summary, method)
            if line:
                print(f"{method:7}: {line.replace('Full experiment ', '').replace('Full experiment: ', '').lstrip('(').replace('runs): ', 'runs: ')}")

    # -------------------------------------------------
    # Build the visual demonstration.
    # -------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.25, 0.85],
        hspace=0.45,
        wspace=0.12,
    )

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, :])

    draw_graph(
        ax1,
        graph,
        dsatur_colouring,
        "DSATUR — classical baseline",
        dsatur_stats,
        reference=reference_text(summary, "DSATUR"),
    )

    draw_graph(
        ax2,
        graph,
        ga_colouring,
        "GA — evolutionary intelligence",
        ga_stats,
        reference=reference_text(summary, "GA"),
    )

    draw_graph(
        ax3,
        graph,
        aco_colouring,
        "ACO — swarm intelligence",
        aco_stats,
        reference=reference_text(summary, "ACO"),
    )

    # Convergence history.
    ga_k, ga_history = get_successful_history(ga_stats["history"])
    aco_k, aco_history = get_successful_history(aco_stats["history"])

    ax4.plot(
        np.arange(1, len(ga_history) + 1),
        ga_history,
        linewidth=2.2,
        marker="o",
        markersize=5,
        label=f"GA (k={ga_k})",
    )
    ax4.plot(
        np.arange(1, len(aco_history) + 1),
        aco_history,
        linewidth=2.2,
        marker="s",
        markersize=5,
        label=f"ACO (k={aco_k})",
    )

    ax4.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        color="#555555",
        label="Valid colouring (0 conflicts)",
    )

    ax4.set_title(
        "Search convergence on the same graph",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Generation (iteration for ACO)")
    ax4.set_ylabel("Best conflicts")
    ax4.grid(alpha=0.25)
    ax4.legend()
    ax4.set_ylim(bottom=-0.8)

    # Overall title and explanation.
    fig.suptitle(
        "Graph Colouring: Same Instance, Three Computational Intelligence Approaches",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    fig.text(
        0.5,
        0.962,
        (
            f"{graph.name}  •  {graph.n} vertices  •  {graph.m} edges  •  "
            f"known χ = {known_chromatic_number}"
        ),
        ha="center",
        fontsize=10,
    )

    # Small colour legend.
    max_colour = max(
        int(np.max(dsatur_colouring)),
        int(np.max(ga_colouring)),
        int(np.max(aco_colouring)),
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PALETTE[c % len(PALETTE)],
            markeredgecolor="#333333",
            markersize=9,
            label=f"Colour {c + 1}",
        )
        for c in range(max_colour + 1)
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(max_colour + 1, 16),
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"\nSaved demonstration figure to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
