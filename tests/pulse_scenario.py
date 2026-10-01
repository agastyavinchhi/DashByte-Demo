"""A fixed, seeded data/ tree for Model Pulse tests, built by the real stages.

The simulator runs on a real-time clock (SIM_CLOCK_SPEED=1, set explicitly so
the golden numbers don't move if the default ever does) from a fixed start, with
a fake clock swapped in for its ``time`` module only, so 130 minutes of orders
take a few seconds and come out identical every run. Preprocess, one train
publish and one infer pass then run as they normally do.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from pipeline import infer, preprocess, simulator, train
from pipeline.config import load_config

START = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)
INTERVAL_SECONDS = 20  # 3 batches of 20 orders a minute
# 130 minutes: the 60-minute window *and* a fully covered previous hour, so the
# KPI deltas are real like-for-like comparisons (they're hidden until then).
BATCHES = 390
SEED = 7
# The batch the scenario ends on; the newest order is stamped at this time.
END = START + timedelta(seconds=INTERVAL_SECONDS * (BATCHES - 1))  # 16:09:40Z


class _FakeTime:
    """Stands in for the simulator's ``time`` module: sleeping advances the clock instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def build(data_dir: Path, batches: int = BATCHES, start: datetime = START) -> None:
    env = {"DASHBITE_DATA_DIR": str(data_dir), "TRAIN_EVERY_N_EVENTS": "100",
           "SIM_CLOCK_SPEED": "1"}
    cfg = load_config(env)
    real_time = simulator.time
    simulator.time = _FakeTime()
    try:
        with redirect_stdout(StringIO()):
            simulator.run(cfg, interval=INTERVAL_SECONDS, max_batches=batches, seed=SEED,
                          messy_rate=0.05, start=start)
            preprocess.run(cfg, poll_seconds=0.01, once=True)
            train.run(cfg, poll_seconds=0.01, once=True)
            infer.run(cfg, poll_seconds=0.01, once=True)
    finally:
        simulator.time = real_time
