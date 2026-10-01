"""Stage 4 contracts. Inference and the Model Pulse dashboard depend on these."""
from __future__ import annotations

import ast
import fnmatch
import inspect
import random
from datetime import datetime, timezone

import pytest

from pipeline import preprocess, simulator, train
from pipeline.config import load_config

pytestmark = pytest.mark.regression


def test_feature_and_label_columns_pinned():
    assert train.FEATURE_COLUMNS == ("distance_km", "prep_minutes")
    assert train.LABEL_COLUMN == "was_late"


def test_input_contract_matches_preprocess_output():
    assert set(train.FEATURE_COLUMNS) <= set(preprocess.FEATURE_COLUMNS)
    assert train.LABEL_COLUMN == preprocess.FEATURE_COLUMNS[-1]
    assert train.TIMESTAMP_COLUMN in preprocess.FEATURE_COLUMNS
    raw = datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc)
    key = simulator.batch_filename(raw, 1)[len("orders_"):]
    assert fnmatch.fnmatch(f"features_{key}", train.FEATURES_GLOB)
    assert not fnmatch.fnmatch(f".features_{key}.tmp", train.FEATURES_GLOB)


def test_model_file_pattern_pinned(tmp_path):
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    joblib_path, json_path = train.model_paths(cfg, 7)
    assert (joblib_path.name, json_path.name) == ("model_v0007.joblib", "model_v0007.json")
    assert train.MODEL_PATTERN.fullmatch("model_v0007.json")


def test_sidecar_keys_pinned():
    assert train.SIDECAR_KEYS == (
        "version", "model_file", "model_type", "feature_columns", "label_column",
        "trained_at", "rows_total", "rows_new", "n_train", "n_test", "data_through",
        "late_rate_test", "baseline_accuracy", "accuracy", "precision", "recall", "f1",
        "roc_auc", "coefficients", "intercept", "train_every_n_events",
    )


def test_training_constants_pinned():
    assert train.HOLDOUT_FRACTION == 0.2
    assert train.MIN_TRAIN_ROWS == 50


def test_train_imports_no_other_stage():
    tree = ast.parse(inspect.getsource(train))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for a in node.names}
    pipeline_modules = {m for m in imported if m and m.startswith("pipeline")}
    assert pipeline_modules == {"pipeline.config", "pipeline.paths"}
    # Train only publishes: it must never reach for inference.
    assert not any("infer" in m for m in imported if m)


def test_model_learns_the_signal(tmp_path, capsys):
    # 5,000 clean simulated orders, through the real preprocess, then train.
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    now = datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc)
    rows = simulator.generate_batch(random.Random(2026), 5000, now, messy_rate=0)
    (tmp_path / "raw").mkdir()
    raw_path = simulator.write_batch(rows, tmp_path / "raw", now, 1)
    preprocess.process_file(raw_path, cfg)

    _, metrics = train.train_model(train.load_rows(cfg))
    assert metrics["rows_total"] == 5000
    assert metrics["accuracy"] >= metrics["baseline_accuracy"] + 0.10
    assert metrics["roc_auc"] >= 0.75
    assert metrics["coefficients"]["distance_km"] > 0
    assert metrics["coefficients"]["prep_minutes"] > 0


def test_train_only_uses_the_features_and_models_folders():
    # Hard isolation: train may locate data/features and data/models, and nothing
    # else. In particular it can't find data/predictions, which inference owns.
    tree = ast.parse(inspect.getsource(train))
    from_paths = {a.name for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and node.module == "pipeline.paths"
                  for a in node.names}
    assert from_paths <= {"display_path", "ensure_data_dirs", "features_dir", "models_dir"}
    source = inspect.getsource(train)
    for other in ("raw_dir", "predictions_dir", "quality_dir", '"predictions"', "'predictions'"):
        assert other not in source
