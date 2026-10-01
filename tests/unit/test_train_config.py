from __future__ import annotations

import pytest

from pipeline.config import load_config

pytestmark = pytest.mark.unit


def test_train_defaults():
    cfg = load_config({})
    assert cfg.train_poll_seconds == 15.0
    assert cfg.train_once is False
    assert cfg.train_every_n_events == 400


def test_train_overrides():
    cfg = load_config({"TRAIN_POLL_SECONDS": "0.2", "TRAIN_ONCE": "1",
                       "TRAIN_EVERY_N_EVENTS": "50"})
    assert (cfg.train_poll_seconds, cfg.train_once, cfg.train_every_n_events) == (0.2, True, 50)


@pytest.mark.parametrize("var,bad", [
    ("TRAIN_POLL_SECONDS", "0"),
    ("TRAIN_POLL_SECONDS", "-1"),
    ("TRAIN_POLL_SECONDS", "nan"),
    ("TRAIN_POLL_SECONDS", "abc"),
    ("TRAIN_ONCE", "2"),
    ("TRAIN_ONCE", "yes"),
])
def test_bad_values_raise_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})
