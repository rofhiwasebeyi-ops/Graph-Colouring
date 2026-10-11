# Comparative Analysis of Evolutionary and Swarm Intelligence Algorithms for the Graph Colouring Problem

Computational Intelligence — Optimization Project

## Folder structure

```
.
├── requirements.txt
├── conftest.py                   # lets pytest find src/ modules — do not delete
├── src/
│   ├── graph.py                  # Graph class, DIMACS loader, random graph generator
│   ├── dsatur.py                 # DSATUR classical heuristic (baseline)
│   ├── ga.py                     # Genetic Algorithm (Evolutionary Intelligence)
│   ├── aco.py                    # Ant Colony Optimization (Swarm Intelligence)
│   ├── stats_utils.py            # Summary statistics + significance testing
│   ├── tune_hyperparameters.py   # Formal grid search for GA/ACO hyperparameters
│   ├── run_experiment.py         # Full experiment: all 3 methods on all 7 instances
│   ├── regenerate_history.py     # Regenerates convergence curves at the best-achieved k
│   ├── summarize_results.py      # Aggregated stats + Mann-Whitney U significance table
│   ├── make_plots.py             # Generates the report's 3 figures
│   ├── verify_results.py         # Independent, external validation of colouring correctness
├── tests/                        # 45 pytest unit tests (Graph, DSATUR, GA, ACO, stats_utils)
└── instances/                    # 7 real DIMACS benchmark graphs
    ├── myciel3.col   myciel4.col   myciel5.col   myciel6.col
    └── queen5_5.col  queen6_6.col  queen8_8.col
```

## Setup

```bash
pip install -r requirements.txt
```

## Where files are saved

Every generated file (CSVs, JSON, figures, demo images, the Word report) is saved in
the `outputs/` folder, which is created automatically. Scripts read their inputs
from there too, so run the pipeline steps in order. Deleting `outputs/` is safe:
the scripts fall back to the built-in tuned hyperparameters and the folder is
recreated on the next run. Benchmark graphs stay in `instances/`.

## Quick visual demo

```bash
python demo_colouring.py --instance 6     # queen6_6; or run without arguments for a menu
```

Colours the chosen graph with DSATUR, GA and ACO side by side and plots GA/ACO
convergence. Saves `outputs/demo_<instance>.png`. Uses the tuned hyperparameters
(`best_params.json` if present, otherwise the built-in tuned defaults) with a
reduced generation budget so it finishes in seconds. If `outputs/results.csv` (from
`run_experiment.py`) exists, each panel also shows that method's
full-experiment mean ± std for the same graph, so the live run can be compared
with the reported results (the CSV holds numbers only, so the coloured graphs are
always from the live run); the hardest graph
(queen8_8) may show no improvement over DSATUR at that budget.

## Running the full pipeline

Run these in order from the project root:

```bash
# 0. Unit tests — confirms the implementation is correct before trusting any results
python -m pytest tests/ -v

# 1. Formal hyperparameter tuning 
#    Writes outputs/tuning_ga.csv, tuning_aco.csv, tuning_winners.csv, best_params.json
#    Steps 2-3 and the demo load best_params.json automatically; if it does not
#    exist (e.g. you skip this step) they use the built-in tuned defaults.
python src/tune_hyperparameters.py

# 2. Full experiment across all 7 DIMACS instances (the two
#    largest instances, myciel6 and queen8_8, take most of that time)
#    Writes outputs/results.csv
python src/run_experiment.py

# 3. Regenerate convergence curves at the actual best-achieved k (~3 minutes)
#    Writes outputs/convergence_history.json
python src/regenerate_history.py

# 4. Compute summary statistics + significance tests
#    Writes outputs/summary_table.csv
python src/summarize_results.py

# 5. Generate the three report figures
#    Writes outputs/colours_vs_graph.png, convergence_curves.png, runtime_comparison.png
python src/make_plots.py

# 6. Independently verify colouring correctness (external check, separate from
#    the algorithms' own internal conflict-counting)
python src/verify_results.py

```

Steps 1-5 must run in that order — each one reads a file written by the step
before it. Step 0 and step 6 can be run any time.

## What to expect

- Step 1 (tuning) will print the winning GA/ACO configuration at the end.
- Step 2 (main experiment) prints progress per graph/method/run as it goes —
  it's normal for this to take a while, especially on `myciel6.col` and
  `queen8_8.col`.
- If you interrupt step 2 partway through, `outputs/results.csv` will contain whatever
  rows were written before the interruption (it writes incrementally), but
  steps 3-5 expect it to be complete — re-run step 2 fully before continuing.

## Notes on the implementation

Everything is implemented from scratch in NumPy (no off-the-shelf EA/ACO
libraries). GA uses a direct colour-vector encoding with conflict-directed
mutation; ACO follows the ANTCOL construction approach (Costa & Hertz, 1997)
with a dynamic DSATUR-style saturation ordering. Both use an identical
k-decrementing search strategy so the comparison against DSATUR's colour
count is apples-to-apples, and both were tuned via an identical grid-search
protocol (`tune_hyperparameters.py`) before the main experiment was run.
