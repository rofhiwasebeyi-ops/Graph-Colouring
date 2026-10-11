"""
Loading and saving the tuned hyperparameters.

`tune_hyperparameters.py` writes outputs/best_params.json.
The experiment scripts read it through load_best_params(), so re-running
tuning changes what the experiments use, with no code edits.

If best_params.json does not exist (e.g. a fresh unzip, or you have not
run tuning), the built-in defaults below are used. These are the values
the tuning step selected for the reported results.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"   # every generated file goes here
PARAMS_FILE = OUTPUT_DIR / "best_params.json"

DEFAULTS = {
    "GA": {"population_size": 100, "mutation_rate": 0.05},
    "ACO": {"n_ants": 45, "beta": 3.0},
}

# Experiment table, shared by run_experiment.py and demo_colouring.py so the
# demo uses the same search budgets as the reported results.
# (instance file, known chromatic number chi, GA generations, ACO generations, N_RUNS)
EXPERIMENT_INSTANCES = [
    ("instances/myciel3.col",  4, 200, 150, 8),
    ("instances/myciel4.col",  5, 250, 180, 8),
    ("instances/myciel5.col",  6, 300, 200, 8),
    ("instances/queen5_5.col", 5, 250, 180, 8),
    ("instances/queen6_6.col", 7, 300, 200, 8),
    ("instances/myciel6.col",  7, 400, 250, 5),
    ("instances/queen8_8.col", 9, 500, 300, 5),
]


def output_path(name):
    """Path of a generated file inside outputs/ (creates the folder if needed)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name


def load_best_params(path=None, verbose=True):
    """Return {"GA": {...}, "ACO": {...}}, from the JSON file if present."""
    path = Path(path) if path else PARAMS_FILE
    params = {alg: dict(vals) for alg, vals in DEFAULTS.items()}
    source = "built-in defaults (no best_params.json found)"

    if path.exists():
        with open(path) as f:
            loaded = json.load(f)
        for alg in params:
            for key in params[alg]:
                if key in loaded.get(alg, {}):
                    params[alg][key] = loaded[alg][key]
        source = str(path)

    if verbose:
        print(f"Hyperparameters ({source}):")
        for alg, vals in params.items():
            print(f"  {alg}: {vals}")
    return params


def save_best_params(best_ga, best_aco, path=None):
    """Write the tuning winners (dicts of grid-search rows) to JSON."""
    path = Path(path) if path else PARAMS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "GA": {
            "population_size": int(best_ga["population_size"]),
            "mutation_rate": float(best_ga["mutation_rate"]),
        },
        "ACO": {
            "n_ants": int(best_aco["n_ants"]),
            "beta": float(best_aco["beta"]),
        },
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return params
