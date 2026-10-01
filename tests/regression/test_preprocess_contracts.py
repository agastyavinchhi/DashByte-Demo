"""Stage 3 contracts. Training and the dashboard depend on these."""
from __future__ import annotations

import ast
import csv
import inspect
import random
from datetime import datetime, timezone

import pytest

from pipeline import preprocess, runner, simulator
from pipeline.config import load_config

pytestmark = pytest.mark.regression


def test_feature_columns_pinned_with_label_last():
    assert preprocess.FEATURE_COLUMNS == (
        "order_id", "timestamp", "hour", "is_peak",
        "distance_km", "prep_minutes", "order_value", "was_late",
    )
    assert preprocess.FEATURE_COLUMNS[-1] == "was_late"


def test_peak_hours_pinned():
    assert preprocess.PEAK_HOURS == frozenset({11, 12, 13, 17, 18, 19, 20})


def test_reject_reasons_pinned():
    # wrong_field_count added deliberately after review: extra fields used to be dropped silently.
    assert preprocess.REJECT_REASONS == (
        "duplicate_id", "wrong_field_count", "blank", "bad_timestamp", "not_a_number",
        "out_of_range", "bad_label",
    )


def test_valid_ranges_pinned():
    assert preprocess.VALID_RANGES == {
        "distance_km": (0.0, 50.0),
        "prep_minutes": (0, 120),
        "order_value": (0.0, 1000.0),
    }


def test_raw_contract_matches_simulator():
    # The only place a drift in the raw hand-off would be caught.
    assert preprocess.RAW_GLOB == simulator.FILE_GLOB
    assert preprocess.RAW_COLUMNS == simulator.COLUMNS


def test_peak_hours_agree_between_stages():
    assert simulator.SIM_PEAK_HOURS == preprocess.PEAK_HOURS


def test_preprocess_imports_only_shared_modules():
    # Folders are contracts, imports are not.
    tree = ast.parse(inspect.getsource(preprocess))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for a in node.names}
    assert {m for m in imported if m and m.startswith("pipeline")} == {
        "pipeline.config", "pipeline.paths",
    }


def test_stage3_simulator_defaults_pinned():
    cfg = load_config({})
    assert cfg.sim_clock_speed == 300
    assert cfg.sim_peak_delay_minutes == 8


def test_runner_processes_pinned():
    # Stage 4 added train, deliberately.
    # Stage 4 added train and Stage 5 added infer, deliberately.
    assert list(runner.PROCESSES) == ["simulator", "preprocess", "train", "infer", "dashboard"]
    assert runner.PROCESSES["infer"] == ["-m", "pipeline.infer"]
    assert runner.PROCESSES["preprocess"] == ["-m", "pipeline.preprocess"]


def _late_share(hour):
    now = datetime(2026, 9, 28, hour, 30, tzinfo=timezone.utc)
    rows = simulator.generate_batch(random.Random(2026), 5000, now, messy_rate=0)
    return sum(r["was_late"] == "1" for r in rows) / len(rows)


def test_rush_hour_signal_is_learnable():
    peak, off_peak = _late_share(18), _late_share(14)  # ~75% vs ~44% today
    assert peak - off_peak >= 0.10
    assert 0.30 <= off_peak <= 0.60  # Stage 2's band


def test_rush_hour_survives_preprocessing(tmp_path):
    # End to end: generate a day of raw batches, preprocess them, compare by is_peak.
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path), "SIM_MESSY_RATE": "0"})
    rng = random.Random(2026)
    (tmp_path / "raw").mkdir()
    for seq, hour in enumerate(range(24), start=1):
        now = datetime(2026, 9, 28, hour, 30, tzinfo=timezone.utc)
        rows = simulator.generate_batch(rng, 200, now, 0)
        simulator.write_batch(rows, tmp_path / "raw", now, seq)
    assert preprocess.run(cfg, poll_seconds=0.01, once=True)[0] == 24

    late = {"0": [0, 0], "1": [0, 0]}
    for path in (tmp_path / "features").glob("features_*.csv"):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                late[row["is_peak"]][0] += row["was_late"] == "1"
                late[row["is_peak"]][1] += 1
    assert late["1"][0] / late["1"][1] > late["0"][0] / late["0"][1]


def test_reject_columns_pinned():
    # raw_line added deliberately (after review) and kept last, so `reason` stays
    # the 7th column for anything already reading rejects files positionally.
    assert preprocess.REJECT_COLUMNS == (
        "order_id", "timestamp", "distance_km", "prep_minutes", "order_value", "was_late",
        "reason", "raw_line",
    )
