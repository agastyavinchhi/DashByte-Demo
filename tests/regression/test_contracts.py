"""Contracts later stages depend on. Changing one should need a deliberate test update."""
from __future__ import annotations

import pytest

from pipeline import paths
from pipeline.config import ENV_VARS, load_config

pytestmark = pytest.mark.regression


def test_default_train_every_n_events_pinned():
    # Stage 1 pinned 2000; Stage 4 lowered it deliberately (Architect, 2026-09-29).
    assert load_config({}).train_every_n_events == 400


def test_default_batch_size_pinned():
    # Stage 1 pinned 50; Stage 2 deliberately lowered it to 20 so the live log is easy to follow.
    assert load_config({}).batch_size == 20


def test_data_dir_names_pinned():
    assert set(paths.DATA_DIR_NAMES) == {"raw", "features", "models", "predictions", "quality"}


def test_ensure_data_dirs_keys_match_pinned_names(tmp_data_dir):
    result = paths.ensure_data_dirs(load_config())
    assert set(result) == {"raw", "features", "models", "predictions", "quality"}


def test_env_var_names_pinned():
    # `VAR=... make <target>` in class and in smoke tests relies on these exact names.
    assert ENV_VARS == (
        "TRAIN_EVERY_N_EVENTS", "BATCH_SIZE", "DASHBITE_DATA_DIR", "DASHBITE_RUN_DIR",
        "SIM_INTERVAL_SECONDS", "SIM_MAX_BATCHES", "SIM_SEED", "SIM_MESSY_RATE",
        "SIM_LATE_THRESHOLD_MINUTES",
        # Stage 3, appended deliberately in this order.
        "SIM_CLOCK_SPEED", "SIM_PEAK_DELAY_MINUTES", "PREPROCESS_POLL_SECONDS", "PREPROCESS_ONCE",
        # Stage 4, appended deliberately.
        "TRAIN_POLL_SECONDS", "TRAIN_ONCE",
        # Stage 5, appended deliberately.
        "INFER_POLL_SECONDS", "INFER_ONCE",
        # Stage 6, appended deliberately.
        "DASHBOARD_WINDOW_MINUTES", "DASHBOARD_BUCKET_MINUTES", "DASHBOARD_REFRESH_SECONDS",
        "DASHBOARD_PORT",
        # Full stack, appended deliberately: one classroom poll cadence.
        "POLL_INTERVAL_SECONDS",
    )
