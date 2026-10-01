from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pipeline.config import PROJECT_ROOT, Config, load_config

pytestmark = pytest.mark.unit


def test_defaults_when_env_empty():
    cfg = load_config({})
    assert cfg.train_every_n_events == 400  # was 2000 until Stage 4; changed deliberately
    assert cfg.batch_size == 20  # was 50 in Stage 1; changed deliberately in Stage 2
    assert cfg.data_dir == PROJECT_ROOT / "data"


def test_train_every_n_events_override_is_int():
    cfg = load_config({"TRAIN_EVERY_N_EVENTS": "500"})
    assert cfg.train_every_n_events == 500
    assert isinstance(cfg.train_every_n_events, int)
    assert cfg.batch_size == 20


def test_batch_size_override_is_int():
    cfg = load_config({"BATCH_SIZE": "500"})
    assert cfg.batch_size == 500
    assert isinstance(cfg.batch_size, int)
    assert cfg.train_every_n_events == 400


def test_data_dir_override_absolute(tmp_path):
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    assert cfg.data_dir == tmp_path


def test_data_dir_override_relative_resolves_from_project_root():
    cfg = load_config({"DASHBITE_DATA_DIR": "scratch/data"})
    assert cfg.data_dir == PROJECT_ROOT / "scratch" / "data"


def test_reads_os_environ_by_default(tmp_data_dir, monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "7")
    cfg = load_config()
    assert cfg.batch_size == 7
    assert cfg.data_dir == tmp_data_dir


@pytest.mark.parametrize("var", ["TRAIN_EVERY_N_EVENTS", "BATCH_SIZE"])
@pytest.mark.parametrize("bad", ["abc", "0", "-5", "", "2.5"])
def test_bad_int_raises_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})


def test_empty_data_dir_raises_naming_variable():
    with pytest.raises(ValueError, match="DASHBITE_DATA_DIR"):
        load_config({"DASHBITE_DATA_DIR": "  "})


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        Config().batch_size = 1  # type: ignore[misc]


def test_int_override_tolerates_surrounding_whitespace():
    cfg = load_config({"TRAIN_EVERY_N_EVENTS": " 500 ", "BATCH_SIZE": "10\n"})
    assert cfg.train_every_n_events == 500
    assert cfg.batch_size == 10


def test_data_dir_override_strips_surrounding_whitespace(tmp_path):
    # Regression: " /abs/path" used to be treated as relative -> PROJECT_ROOT/" /abs/path".
    cfg = load_config({"DASHBITE_DATA_DIR": f"  {tmp_path}\n"})
    assert cfg.data_dir == tmp_path


def test_data_dir_override_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config({"DASHBITE_DATA_DIR": "~/dashbite"})
    assert cfg.data_dir == tmp_path / "dashbite"


def test_unset_vars_keep_defaults_independently(tmp_path):
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    assert (cfg.train_every_n_events, cfg.batch_size) == (400, 20)


@pytest.mark.parametrize("var", ["TRAIN_EVERY_N_EVENTS", "BATCH_SIZE"])
@pytest.mark.parametrize("bad", ["0", "-0", "-1", "-2000", " -5 "])
def test_zero_or_negative_count_gives_clear_error(var, bad):
    # The message must say which variable, what was expected and what was given.
    with pytest.raises(ValueError) as exc:
        load_config({var: bad})
    assert str(exc.value) == f"{var} must be a positive integer, got {bad!r}"
