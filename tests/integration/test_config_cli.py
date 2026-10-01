from __future__ import annotations

import subprocess
import sys

import pytest

from pipeline.config import PROJECT_ROOT
from pipeline.paths import DATA_DIR_NAMES
from tests.conftest import clean_env

pytestmark = pytest.mark.integration


def _run_config(env_overrides, cwd):
    env = clean_env(PYTHONPATH=str(PROJECT_ROOT), **env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "pipeline.config"],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
    )


def _dir_status(stdout):
    """Map dir name -> 'created'/'exists' from the CLI output."""
    status = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("created", "exists"):
            status[parts[1]] = parts[0]
    return status


def test_cli_prints_config_and_creates_dirs_idempotently(tmp_path):
    data_dir = tmp_path / "data"
    env = {"DASHBITE_DATA_DIR": str(data_dir), "BATCH_SIZE": "25"}

    # Run from an unrelated cwd to prove nothing depends on the working directory.
    first = _run_config(env, cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    assert "batch_size = 25" in first.stdout
    assert "train_every_n_events = 400" in first.stdout
    for name in DATA_DIR_NAMES:
        assert (data_dir / name).is_dir()
    assert _dir_status(first.stdout) == {n: "created" for n in DATA_DIR_NAMES}

    second = _run_config(env, cwd=tmp_path)
    assert second.returncode == 0, second.stderr
    assert _dir_status(second.stdout) == {n: "exists" for n in DATA_DIR_NAMES}


@pytest.mark.parametrize("var", ["BATCH_SIZE", "TRAIN_EVERY_N_EVENTS"])
def test_cli_bad_value_exits_nonzero_naming_variable(tmp_path, var):
    data_dir = tmp_path / "data"
    result = _run_config({"DASHBITE_DATA_DIR": str(data_dir), var: "oops"},
                         cwd=tmp_path)
    assert result.returncode != 0
    assert var in result.stderr
    assert "Traceback" not in result.stderr
    assert not data_dir.exists()


@pytest.mark.parametrize("var", ["TRAIN_EVERY_N_EVENTS", "BATCH_SIZE"])
@pytest.mark.parametrize("bad", ["0", "-3"])
def test_cli_zero_or_negative_count_fails_clearly(tmp_path, var, bad):
    data_dir = tmp_path / "data"
    result = _run_config({"DASHBITE_DATA_DIR": str(data_dir), var: bad}, cwd=tmp_path)
    assert result.returncode == 1
    assert result.stderr.strip() == f"config error: {var} must be a positive integer, got '{bad}'"
    assert result.stdout == ""  # fails before printing any config
    assert not data_dir.exists()  # and before creating any dirs
