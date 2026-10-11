"""Tests for loading/saving tuned hyperparameters."""

import json

from params import DEFAULTS, load_best_params, save_best_params


def test_missing_file_falls_back_to_defaults(tmp_path):
    params = load_best_params(tmp_path / "nope.json", verbose=False)
    assert params == DEFAULTS


def test_defaults_are_not_mutated_by_callers(tmp_path):
    params = load_best_params(tmp_path / "nope.json", verbose=False)
    params["GA"]["population_size"] = 1
    assert DEFAULTS["GA"]["population_size"] == 100


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "best.json"
    save_best_params(
        {"population_size": 80, "mutation_rate": 0.02},
        {"n_ants": 30, "beta": 2.0},
        path,
    )
    params = load_best_params(path, verbose=False)
    assert params["GA"] == {"population_size": 80, "mutation_rate": 0.02}
    assert params["ACO"] == {"n_ants": 30, "beta": 2.0}


def test_partial_file_keeps_defaults_for_missing_keys(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"GA": {"mutation_rate": 0.01}}))
    params = load_best_params(path, verbose=False)
    assert params["GA"]["mutation_rate"] == 0.01
    assert params["GA"]["population_size"] == DEFAULTS["GA"]["population_size"]
    assert params["ACO"] == DEFAULTS["ACO"]


def test_loaded_params_construct_the_algorithms(tmp_path):
    from aco import ACOColourer
    from ga import GAColourer
    from graph import random_graph

    g = random_graph(10, 0.3, seed=1)
    params = load_best_params(tmp_path / "nope.json", verbose=False)
    GAColourer(g, **params["GA"], generations=5, seed=0)
    ACOColourer(g, **params["ACO"], generations=5, seed=0)


def test_output_path_creates_folder_and_points_inside_it(tmp_path, monkeypatch):
    import params

    monkeypatch.setattr(params, "OUTPUT_DIR", tmp_path / "outputs")
    path = params.output_path("results.csv")
    assert path == tmp_path / "outputs" / "results.csv"
    assert (tmp_path / "outputs").is_dir()


def test_save_creates_missing_output_folder(tmp_path):
    target = tmp_path / "does" / "not" / "exist" / "best.json"
    save_best_params(
        {"population_size": 100, "mutation_rate": 0.05},
        {"n_ants": 45, "beta": 3.0},
        target,
    )
    assert target.exists()
