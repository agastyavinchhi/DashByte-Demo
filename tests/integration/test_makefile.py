"""Drive the real Makefile, since it is the interface the room uses.

clean-data is only ever exercised with ``make -n`` (dry run), so these tests can
never delete the developer's real data/.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from pipeline.config import (
    ENV_BATCH_SIZE,
    ENV_DATA_DIR,
    ENV_SIM_CLOCK_SPEED,
    ENV_TRAIN_EVERY_N_EVENTS,
    PROJECT_ROOT,
)
from pipeline.paths import DATA_DIR_NAMES
from tests.conftest import clean_env

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("make") is None, reason="make not installed"),
]

MAKEFILE = PROJECT_ROOT / "Makefile"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
needs_venv = pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")


def _make(args, cwd, env_overrides=None, makefile=MAKEFILE):
    env = clean_env(**(env_overrides or {}))
    return subprocess.run(
        ["make", "-f", str(makefile), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )


@needs_venv
def test_make_config_honours_env_overrides_from_any_cwd(tmp_path):
    data_dir = tmp_path / "data"
    result = _make(["config"], cwd=tmp_path, env_overrides={
        ENV_DATA_DIR: str(data_dir), ENV_BATCH_SIZE: "10", ENV_TRAIN_EVERY_N_EVENTS: "500",
    })
    assert result.returncode == 0, result.stderr
    assert "train_every_n_events = 500" in result.stdout
    assert "batch_size = 10" in result.stdout
    assert f"data_dir = {data_dir}" in result.stdout
    assert all((data_dir / name).is_dir() for name in DATA_DIR_NAMES)


@needs_venv
def test_make_config_falls_back_to_real_time_when_sim_clock_speed_is_removed(tmp_path):
    # Stage 3's rush-hour demo has the room run SIM_CLOCK_SPEED=300, and a shell
    # that still exports it would quietly put every later demo on the fast clock.
    # Once the variable is removed, `make config` must report the real-time
    # default (1x, since Stage 6), which Model Pulse's per-minute charts rely on.
    data_dir = tmp_path / "data"
    leftover = clean_env(**{ENV_DATA_DIR: str(data_dir), ENV_SIM_CLOCK_SPEED: "300"})

    # First prove `make config` really reads the variable, so the check below means something.
    with_leftover = subprocess.run(["make", "-f", str(MAKEFILE), "config"], cwd=tmp_path,
                                   env=leftover, capture_output=True, text=True, timeout=60)
    assert with_leftover.returncode == 0, with_leftover.stderr
    assert "sim_clock_speed = 300.0" in with_leftover.stdout.splitlines()

    removed = dict(leftover)
    del removed[ENV_SIM_CLOCK_SPEED]
    assert ENV_SIM_CLOCK_SPEED not in removed
    result = subprocess.run(["make", "-f", str(MAKEFILE), "config"], cwd=tmp_path,
                            env=removed, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "sim_clock_speed = 1.0" in lines  # the real-time default, exactly once
    assert sum(line.startswith("sim_clock_speed = ") for line in lines) == 1
    assert all((data_dir / name).is_dir() for name in DATA_DIR_NAMES)


@needs_venv
def test_make_config_bad_value_fails_make(tmp_path):
    data_dir = tmp_path / "data"
    result = _make(["config"], cwd=tmp_path,
                   env_overrides={ENV_DATA_DIR: str(data_dir), ENV_BATCH_SIZE: "oops"})
    assert result.returncode != 0
    assert "BATCH_SIZE" in result.stderr
    assert not data_dir.exists()


def test_clean_data_targets_only_project_data_dir(tmp_path):
    result = _make(["-n", "clean-data"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    rm_lines = [line for line in result.stdout.splitlines() if line.startswith("rm ")]
    assert rm_lines == [f'rm -rf "{PROJECT_ROOT / "data"}"']


def test_clean_data_refuses_when_custom_data_dir_set(tmp_path):
    result = _make(["-n", "clean-data"], cwd=tmp_path,
                   env_overrides={ENV_DATA_DIR: str(tmp_path / "custom")})
    assert result.returncode != 0
    assert ENV_DATA_DIR in result.stderr
    assert "rm " not in result.stdout


@pytest.mark.parametrize("from_inside", [True, False], ids=["cwd", "make-f"])
def test_refuses_to_run_from_path_with_spaces(tmp_path, from_inside):
    # Regression: with a space in the path, ROOT was split and clean-data's
    # `rm -rf` expanded to the *parent* folder.
    project = tmp_path / "My Folder"
    (project / "pipeline").mkdir(parents=True)
    (project / "pipeline" / "config.py").touch()
    shutil.copy(MAKEFILE, project / "Makefile")

    if from_inside:
        result = _make(["-n", "clean-data"], cwd=project, makefile="Makefile")
    else:
        result = _make(["-n", "clean-data"], cwd=tmp_path, makefile=project / "Makefile")
    assert result.returncode != 0
    assert "rm " not in result.stdout
