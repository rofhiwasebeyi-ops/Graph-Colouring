"""
Produces summary_table.csv (mean/std of colours and runtime per graph/method)
and prints Mann-Whitney U test results comparing GA vs ACO colour-count
distributions per graph, for use in the report's results/discussion section.
"""

import pandas as pd
from params import output_path
from stats_utils import summarise, compare_methods

df = pd.read_csv(output_path("results.csv"))

rows = []
for graph in df["graph"].unique():
    sub = df[df.graph == graph]
    chi = sub["chi_known"].iloc[0]
    n = sub["n"].iloc[0]
    density = sub["density"].iloc[0]

    dsatur_row = sub[sub.method == "DSATUR"].iloc[0]

    ga_colours = sub[sub.method == "GA"]["n_colours"].dropna()
    aco_colours = sub[sub.method == "ACO"]["n_colours"].dropna()
    ga_time = sub[sub.method == "GA"]["runtime_s"]
    aco_time = sub[sub.method == "ACO"]["runtime_s"]

    ga_stats = summarise(ga_colours) if len(ga_colours) else {"mean": None, "std": None}
    aco_stats = summarise(aco_colours) if len(aco_colours) else {"mean": None, "std": None}

    _, p_value = (None, None)
    if len(ga_colours) and len(aco_colours):
        _, p_value = compare_methods(ga_colours, aco_colours)

    rows.append({
        "graph": graph, "n": n, "density": round(density, 3), "chi_known": chi,
        "dsatur_colours": dsatur_row["n_colours"],
        "ga_mean_colours": round(ga_stats["mean"], 2) if ga_stats["mean"] is not None else None,
        "ga_std_colours": round(ga_stats["std"], 2) if ga_stats["std"] is not None else None,
        "aco_mean_colours": round(aco_stats["mean"], 2) if aco_stats["mean"] is not None else None,
        "aco_std_colours": round(aco_stats["std"], 2) if aco_stats["std"] is not None else None,
        "ga_mean_runtime_s": round(ga_time.mean(), 2),
        "aco_mean_runtime_s": round(aco_time.mean(), 2),
        "dsatur_runtime_s": round(dsatur_row["runtime_s"], 5),
        "ga_vs_aco_pvalue": round(p_value, 4) if p_value is not None else None,
        "ga_success_rate": f"{ga_colours.notna().sum() if hasattr(ga_colours,'notna') else len(ga_colours)}/{len(sub[sub.method=='GA'])}",
        "aco_success_rate": f"{len(aco_colours)}/{len(sub[sub.method=='ACO'])}",
    })

summary = pd.DataFrame(rows)
summary.to_csv(output_path("summary_table.csv"), index=False)
print(summary.to_string(index=False))
