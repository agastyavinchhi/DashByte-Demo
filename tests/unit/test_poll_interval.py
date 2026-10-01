"""POLL_INTERVAL_SECONDS: one classroom cadence for every poller and the dashboard."""
from __future__ import annotations

import pytest

from pipeline.config import Config, load_config

pytestmark = pytest.mark.unit

POLLERS = ("preprocess_poll_seconds", "train_poll_seconds", "infer_poll_seconds",
           "dashboard_refresh_seconds")


def test_default_cadence_is_15_seconds_everywhere():
    cfg = load_config({})
    assert cfg.poll_interval_seconds == 15.0
    assert {getattr(cfg, f) for f in POLLERS} == {15.0}
    assert cfg == Config()


def test_poll_interval_sets_every_poller():
    cfg = load_config({"POLL_INTERVAL_SECONDS": "4"})
    assert {getattr(cfg, f) for f in POLLERS} == {4.0}


def test_stage_specific_setting_still_wins():
    cfg = load_config({"POLL_INTERVAL_SECONDS": "4", "TRAIN_POLL_SECONDS": "1",
                       "DASHBOARD_REFRESH_SECONDS": "30"})
    assert (cfg.preprocess_poll_seconds, cfg.train_poll_seconds, cfg.infer_poll_seconds,
            cfg.dashboard_refresh_seconds) == (4.0, 1.0, 4.0, 30.0)


def test_simulator_interval_is_not_a_poll():
    assert load_config({"POLL_INTERVAL_SECONDS": "4"}).sim_interval_seconds == 3.0


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "abc", ""])
def test_bad_poll_interval_names_the_variable(bad):
    with pytest.raises(ValueError, match="POLL_INTERVAL_SECONDS"):
        load_config({"POLL_INTERVAL_SECONDS": bad})
