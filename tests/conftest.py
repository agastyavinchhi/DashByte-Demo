from __future__ import annotations

import os

import pytest

from pipeline.config import ENV_DATA_DIR, ENV_RUN_DIR, ENV_VARS


def clean_env(**overrides):
    """os.environ minus every DashBite variable, plus ``overrides``. For subprocess tests."""
    drop = set(ENV_VARS) | {"MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(overrides)
    return env


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Point DASHBITE_DATA_DIR at a temp dir so tests never write to the real data/.

    Also clears the other overrides so a developer's shell env can't leak in.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    data_dir = tmp_path / "data"
    monkeypatch.setenv(ENV_DATA_DIR, str(data_dir))
    return data_dir


@pytest.fixture
def tmp_run_dir(tmp_path, tmp_data_dir, monkeypatch):
    """Point DASHBITE_RUN_DIR at a temp dir so tests never touch the real .run/ or logs/."""
    run_dir = tmp_path / "run"
    monkeypatch.setenv(ENV_RUN_DIR, str(run_dir))
    return run_dir
