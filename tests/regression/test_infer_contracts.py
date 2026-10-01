"""Stage 5 contracts. The Model Pulse dashboard depends on these.

Golden scoring: a LogisticRegression with fixed coefficients (no fit, so the
result doesn't depend on the scikit-learn version) scores the golden features
fixture, and the output must match tests/fixtures/golden_predictions_seed2027.csv
byte for byte. If a change is deliberate, regenerate and review the diff:

    .venv/bin/python -m tests.regression.test_infer_contracts
"""
from __future__ import annotations

import ast
import fnmatch
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline import infer, preprocess, train
from pipeline.config import load_config

pytestmark = pytest.mark.regression

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_FEATURES = FIXTURES / "golden_features_seed2027.csv"
GOLDEN_PREDICTIONS = FIXTURES / "golden_predictions_seed2027.csv"
FEATURES_NAME = "features_20260928T140203Z_0001.csv"


def fixed_model() -> LogisticRegression:
    model = LogisticRegression()
    model.coef_ = np.array([[0.6, 0.25]])
    model.intercept_ = np.array([-7.0])
    model.classes_ = np.array([0, 1])
    return model


def publish_fixed(models_dir: Path, version: int = 1) -> None:
    """A checkpoint in train's on-disk format, built without running train."""
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model_v{version:04d}"
    columns = ["distance_km", "prep_minutes"]
    joblib.dump({"model": fixed_model(), "feature_columns": columns, "version": version},
                models_dir / f"{stem}.joblib")
    (models_dir / f"{stem}.json").write_text(json.dumps({
        "version": version, "model_file": f"{stem}.joblib", "feature_columns": columns,
        "rows_total": 8, "accuracy": 0.9,
    }))


def score_golden(work_dir: Path) -> Path:
    cfg = load_config({"DASHBITE_DATA_DIR": str(work_dir)})
    publish_fixed(work_dir / "models")
    (work_dir / "features").mkdir(parents=True)
    features = work_dir / "features" / FEATURES_NAME
    shutil.copyfile(GOLDEN_FEATURES, features)
    loaded = infer.load_checkpoint(infer.newest_checkpoint(cfg))
    return infer.score_file(features, loaded, cfg).predictions_path


def test_golden_predictions_byte_for_byte(tmp_path):
    assert score_golden(tmp_path).read_bytes() == GOLDEN_PREDICTIONS.read_bytes()


def test_prediction_contract_pinned():
    assert infer.PREDICTION_COLUMNS == (
        "order_id", "late_probability", "predicted_late", "checkpoint_id",
    )
    assert infer.PREDICT_THRESHOLD == 0.5


def test_output_file_name_pinned(tmp_path):
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    path = infer.predictions_path_for(tmp_path / FEATURES_NAME, cfg)
    assert path == tmp_path / "predictions" / "predictions_20260928T140203Z_0001.csv"


def test_input_contracts_match_producers():
    assert infer.FEATURES_GLOB == train.FEATURES_GLOB
    assert fnmatch.fnmatch(FEATURES_NAME, infer.FEATURES_GLOB)
    assert set(preprocess.FEATURE_COLUMNS) >= {"order_id", *train.FEATURE_COLUMNS}
    for name in ("model_v0001.json", "model_v0042.json", "model_v12345.json"):
        match = train.MODEL_PATTERN.fullmatch(name)
        assert match and match.group(2) == "json"
        assert infer.MODEL_SIDECAR_PATTERN.fullmatch(name)
    for name in ("model_v0001.joblib", ".model_v0001.json.tmp", "model_v1.json"):
        assert not infer.MODEL_SIDECAR_PATTERN.fullmatch(name)


def _model_dir_cases(root: Path) -> dict[str, tuple[Path, int]]:
    """name -> (data dir, expected newest version). Each case has its own data dir."""
    cases = {}

    def case(name, expected):
        data = root / name
        (data / "models").mkdir(parents=True)
        cases[name] = (data, expected)
        return data / "models"

    models = case("clean", 3)
    for v in (1, 2, 3):
        publish_fixed(models, v)
    models = case("orphan", 2)
    for v in (1, 2):
        publish_fixed(models, v)
    joblib.dump({"model": fixed_model()}, models / "model_v0003.joblib")  # no sidecar
    models = case("damaged", 2)
    for v in (1, 2, 3):
        publish_fixed(models, v)
    (models / "model_v0003.json").write_text("{truncated")
    case("empty", 0)
    return cases


def test_newest_means_the_same_to_train_and_infer(tmp_path):
    # Tests may import both stages; the stages themselves never import each other.
    for name, (data, expected) in _model_dir_cases(tmp_path).items():
        cfg = load_config({"DASHBITE_DATA_DIR": str(data)})
        newest = infer.newest_checkpoint(cfg)
        assert (newest.version if newest else 0) == train.latest_published(cfg).version == (
            expected), name


def test_infer_imports_no_other_stage():
    tree = ast.parse(inspect.getsource(infer))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for a in node.names}
    assert {m for m in imported if m and m.startswith("pipeline")} == {
        "pipeline.config", "pipeline.paths",
    }
    assert not any("train" in m for m in imported if m)


def test_run_is_read_only_on_models(tmp_path, capsys):
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    publish_fixed(tmp_path / "models", 1)
    publish_fixed(tmp_path / "models", 2)
    (tmp_path / "features").mkdir()
    shutil.copyfile(GOLDEN_FEATURES, tmp_path / "features" / FEATURES_NAME)

    def snapshot():
        return {p.name: (p.read_bytes(), os.stat(p).st_mtime_ns)
                for p in (tmp_path / "models").iterdir()}

    before = snapshot()
    assert infer.run(cfg, poll_seconds=0.01, once=True)[0] == 1
    assert snapshot() == before


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(score_golden(Path(tmp)), GOLDEN_PREDICTIONS)
    print(f"wrote {GOLDEN_PREDICTIONS}", file=sys.stderr)


def test_infer_cannot_start_signal_or_feed_training(tmp_path, capsys):
    # "Inference never triggers training": it has no way to start or signal a
    # process, and it leaves everything train reads (features, models) untouched.
    source = inspect.getsource(infer)
    for forbidden in ("subprocess", "os.kill", "Popen", "pipeline.runner", "TRAIN_",
                      "train_every_n_events", "pipeline.train"):
        assert forbidden not in source, forbidden

    from tests.regression.test_golden_preprocess import preprocess_golden

    data = tmp_path / "data"
    preprocess_golden(data)
    cfg = load_config({"DASHBITE_DATA_DIR": str(data), "TRAIN_EVERY_N_EVENTS": "1"})
    assert train.run(cfg, poll_seconds=0.01, once=True) == 0  # too few rows: guard says wait
    # Publish a checkpoint from outside train so infer has something to score with.
    publish_fixed(data / "models")
    before = {p: p.read_bytes() for p in (data / "features").iterdir()}
    seen_by_train = (train.rows_available(cfg), train.latest_published(cfg))

    assert infer.run(cfg, poll_seconds=0.01, once=True)[0] == 1
    assert {p: p.read_bytes() for p in (data / "features").iterdir()} == before
    assert (train.rows_available(cfg), train.latest_published(cfg)) == seen_by_train
