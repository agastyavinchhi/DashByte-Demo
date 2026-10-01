from __future__ import annotations

import pytest

from pipeline.config import PROJECT_ROOT, load_config, load_run_dir

pytestmark = pytest.mark.unit


def test_simulator_defaults():
    cfg = load_config({})
    assert cfg.sim_interval_seconds == 3.0
    assert cfg.sim_max_batches is None
    assert cfg.sim_seed is None
    assert cfg.sim_messy_rate == 0.05
    assert cfg.run_dir == PROJECT_ROOT
    assert cfg.sim_late_threshold_minutes == 30.0


def test_simulator_overrides():
    cfg = load_config({
        "SIM_INTERVAL_SECONDS": "0.5",
        "SIM_MAX_BATCHES": "3",
        "SIM_SEED": "-7",
        "SIM_MESSY_RATE": " 0.2 ",
        "SIM_LATE_THRESHOLD_MINUTES": "37.5",
    })
    assert cfg.sim_interval_seconds == 0.5
    assert cfg.sim_max_batches == 3
    assert cfg.sim_seed == -7
    assert cfg.sim_messy_rate == 0.2
    assert cfg.sim_late_threshold_minutes == 37.5


@pytest.mark.parametrize("rate", ["0", "1", "0.0", "1.0"])
def test_messy_rate_bounds_are_inclusive(rate):
    assert load_config({"SIM_MESSY_RATE": rate}).sim_messy_rate == float(rate)


@pytest.mark.parametrize("var,bad", [
    ("SIM_INTERVAL_SECONDS", "0"),
    ("SIM_INTERVAL_SECONDS", "-1"),
    ("SIM_INTERVAL_SECONDS", "abc"),
    ("SIM_INTERVAL_SECONDS", "nan"),
    ("SIM_INTERVAL_SECONDS", "inf"),
    ("SIM_MESSY_RATE", "1.5"),
    ("SIM_MESSY_RATE", "-0.1"),
    ("SIM_MESSY_RATE", "lots"),
    ("SIM_MAX_BATCHES", "abc"),
    ("SIM_MAX_BATCHES", "0"),
    ("SIM_MAX_BATCHES", "2.5"),
    ("SIM_SEED", "abc"),
    ("SIM_SEED", "1.5"),
    ("DASHBITE_RUN_DIR", "  "),
    ("SIM_LATE_THRESHOLD_MINUTES", "0"),
    ("SIM_LATE_THRESHOLD_MINUTES", "-5"),
    ("SIM_LATE_THRESHOLD_MINUTES", "soon"),
    ("SIM_LATE_THRESHOLD_MINUTES", "nan"),
])
def test_bad_values_raise_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})


def test_run_dir_resolves_like_data_dir(tmp_path):
    assert load_config({"DASHBITE_RUN_DIR": str(tmp_path)}).run_dir == tmp_path
    assert load_config({"DASHBITE_RUN_DIR": "rundir"}).run_dir == PROJECT_ROOT / "rundir"


def test_load_run_dir_ignores_other_bad_vars(tmp_path):
    # `make stop` must still work if, say, SIM_INTERVAL_SECONDS is exported wrong.
    env = {"DASHBITE_RUN_DIR": str(tmp_path), "SIM_INTERVAL_SECONDS": "0", "BATCH_SIZE": "x"}
    assert load_run_dir(env) == tmp_path
    assert load_run_dir({}) == PROJECT_ROOT


def test_stage3_simulator_defaults_and_overrides():
    cfg = load_config({})
    assert (cfg.sim_clock_speed, cfg.sim_peak_delay_minutes) == (300.0, 8.0)
    cfg = load_config({"SIM_CLOCK_SPEED": "1", "SIM_PEAK_DELAY_MINUTES": "0"})
    assert (cfg.sim_clock_speed, cfg.sim_peak_delay_minutes) == (1.0, 0.0)
    assert load_config({"SIM_PEAK_DELAY_MINUTES": "12.5"}).sim_peak_delay_minutes == 12.5


@pytest.mark.parametrize("var,bad", [
    ("SIM_CLOCK_SPEED", "0"),
    ("SIM_CLOCK_SPEED", "-1"),
    ("SIM_CLOCK_SPEED", "nan"),
    ("SIM_CLOCK_SPEED", "abc"),
    ("SIM_PEAK_DELAY_MINUTES", "-1"),
    ("SIM_PEAK_DELAY_MINUTES", "nan"),
    ("SIM_PEAK_DELAY_MINUTES", "abc"),
])
def test_stage3_simulator_bad_values_raise_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})
