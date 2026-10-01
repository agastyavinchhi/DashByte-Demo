"""Stage 2 contracts. Preprocess, train and the dashboard depend on these."""
from __future__ import annotations

import fnmatch
import random
from datetime import datetime, timezone

import pytest

from pipeline import runner
from pipeline.config import load_config
from pipeline.simulator import (
    COLUMNS,
    COPY_EARLIER_ID,
    FILE_GLOB,
    MESSY_DEFECTS,
    batch_filename,
    generate_batch,
)

pytestmark = pytest.mark.regression


def test_columns_pinned():
    assert COLUMNS == (
        "order_id", "timestamp", "distance_km", "prep_minutes", "order_value", "was_late",
    )


def test_messy_defects_pinned():
    # Preprocessing cleans against these exact values.
    assert MESSY_DEFECTS == {
        "blank": ("order_value", ""),
        "negative_distance": ("distance_km", "-1.0"),
        "missing_prep": ("prep_minutes", "n/a"),
        "text_label": ("was_late", "yes"),
        "outlier_distance": ("distance_km", "999.0"),
        "duplicate_id": ("order_id", COPY_EARLIER_ID),
    }


def test_file_pattern_pinned():
    assert FILE_GLOB == "orders_*.csv"
    name = batch_filename(datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc), 3)
    assert name == "orders_20260928T140203Z_0003.csv"
    assert fnmatch.fnmatch(name, FILE_GLOB)


def test_runner_starts_simulator():
    assert runner.PROCESSES["simulator"] == ["-m", "pipeline.simulator"]


def test_default_late_threshold_pinned():
    # Changed deliberately from the planned 35 to 30 minutes.
    assert load_config({}).sim_late_threshold_minutes == 30.0


def _late_share(rows):
    return sum(r["was_late"] == "1" for r in rows) / len(rows)


def test_order_mix_unchanged_at_original_threshold():
    # Guards the distance/prep/noise mix on its own, whatever the default threshold is.
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    rows = generate_batch(random.Random(2026), 5000, now, messy_rate=0, late_threshold=35)
    assert 0.15 <= _late_share(rows) <= 0.45  # ~28% today


def test_lateness_signal_is_learnable():
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    rows = generate_batch(random.Random(2026), 5000, now, messy_rate=0)
    # Band centred on the default 30-minute threshold (~44% today), with room either side.
    assert 0.30 <= _late_share(rows) <= 0.60

    def late_rate(keep):
        picked = [r["was_late"] == "1" for r in rows if keep(float(r["distance_km"]))]
        assert picked, "bucket is empty"
        return sum(picked) / len(picked)

    assert late_rate(lambda d: d > 8) > late_rate(lambda d: d < 3)


def test_simulator_imports_only_shared_modules():
    # Folders are contracts, imports are not: the simulator may use config and
    # paths, never another stage (or the runner).
    import ast
    import inspect

    from pipeline import simulator

    tree = ast.parse(inspect.getsource(simulator))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for a in node.names}
    assert {m for m in imported if m and m.startswith("pipeline")} == {
        "pipeline.config", "pipeline.paths",
    }
