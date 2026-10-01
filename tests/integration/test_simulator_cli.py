from __future__ import annotations

import csv
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pytest

from pipeline.config import PROJECT_ROOT
from tests.conftest import clean_env

pytestmark = pytest.mark.integration

SIM_ENV = {"BATCH_SIZE": "10", "SIM_MAX_BATCHES": "3", "SIM_INTERVAL_SECONDS": "0.1"}
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _env(data_dir, **extra):
    return clean_env(PYTHONPATH=str(PROJECT_ROOT), DASHBITE_DATA_DIR=str(data_dir), **extra)


def _assert_three_batches_of_ten(raw):
    files = sorted(raw.glob("orders_*.csv"))
    assert len(files) == 3
    for f in files:
        with open(f, newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 10
    assert list(raw.glob(".*.tmp")) == []


def _wait_for_first_batch(raw, proc, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"simulator exited early: {proc.communicate()}")
        if raw.is_dir() and any(raw.glob("orders_*.csv")):
            return
        time.sleep(0.05)
    pytest.fail("no batch appeared")


def test_simulator_writes_max_batches_and_exits(tmp_path):
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.simulator"],
        cwd=tmp_path, env=_env(data_dir, **SIM_ENV),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert ("batch size 10, every 0.1s, messy rate 5%, late after 30 min, "
            "clock 1× (real time), up to 3 batches →") in result.stdout
    assert result.stdout.count("new orders arrived: 10 (") == 3
    assert "stopped after 3 batches, 30 orders" in result.stdout
    _assert_three_batches_of_ten(data_dir / "raw")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["ctrl-c", "sigterm"])
def test_simulator_stops_cleanly_on_signal(tmp_path, sig):
    data_dir = tmp_path / "data"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pipeline.simulator"],
        cwd=tmp_path, env=_env(data_dir, SIM_INTERVAL_SECONDS="0.2"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        _wait_for_first_batch(data_dir / "raw", proc)
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "stopped after" in out
    assert "Traceback" not in err
    assert list((data_dir / "raw").glob(".*.tmp")) == []


def test_banner_shows_threshold_override(tmp_path):
    result = subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.simulator"],
        cwd=tmp_path,
        env=_env(tmp_path / "data", SIM_LATE_THRESHOLD_MINUTES="37.5", **SIM_ENV),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "late after 37.5 min" in result.stdout.splitlines()[0]


def test_simulator_bad_config_exits_1_without_traceback(tmp_path):
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.simulator"],
        cwd=tmp_path, env=_env(data_dir, SIM_MESSY_RATE="1.5"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "config error" in result.stderr and "SIM_MESSY_RATE" in result.stderr
    assert "Traceback" not in result.stderr
    assert not data_dir.exists()


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")
def test_make_simulator_wiring(tmp_path):
    data_dir = tmp_path / "data"
    result = subprocess.run(
        ["make", "-f", str(PROJECT_ROOT / "Makefile"), "simulator"],
        cwd=tmp_path, env=clean_env(DASHBITE_DATA_DIR=str(data_dir), **SIM_ENV),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("new orders arrived") == 3
    _assert_three_batches_of_ten(data_dir / "raw")


def _one_tick(data_dir, cwd):
    """One real simulator process, one batch, seeded like the golden fixture."""
    from tests.regression.test_golden_batch import MESSY_RATE, N, SEED

    return subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.simulator"],
        cwd=cwd,
        env=_env(data_dir, BATCH_SIZE=str(N), SIM_SEED=str(SEED), SIM_MESSY_RATE=str(MESSY_RATE),
                 SIM_MAX_BATCHES="1", SIM_INTERVAL_SECONDS="3",
                 # Stage 2 behaviour: real-time clock, no rush-hour delay, so the
                 # labels can't depend on the hour the test happens to run.
                 SIM_CLOCK_SPEED="1", SIM_PEAK_DELAY_MINUTES="0"),
        capture_output=True, text=True, timeout=30,
    )


def test_one_tick_writes_golden_rows_to_raw_only(tmp_path):
    from pipeline.paths import DATA_DIR_NAMES
    from pipeline.simulator import COLUMNS
    from tests.regression.test_golden_batch import GOLDEN, read_csv

    data_dir = tmp_path / "data"
    result = _one_tick(data_dir, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "seed 2027" in result.stdout.splitlines()[0]
    assert "stopped after 1 batches, 20 orders" in result.stdout

    # Exactly one file, in raw/ only. The other hand-off dirs exist but stay empty.
    written = [p for p in data_dir.rglob("*") if p.is_file()]
    assert len(written) == 1, written
    (path,) = written
    assert path.parent == data_dir / "raw"
    assert re.fullmatch(r"orders_\d{8}T\d{6}Z_0001\.csv", path.name)
    assert sorted(p.name for p in data_dir.iterdir()) == sorted(DATA_DIR_NAMES)

    # Same seed, same rows: everything but the wall-clock timestamp matches the golden batch.
    header, rows = read_csv(path)
    _, golden = read_csv(GOLDEN)
    assert tuple(header) == COLUMNS
    strip_ts = lambda rs: [{k: v for k, v in r.items() if k != "timestamp"} for r in rs]  # noqa: E731
    assert strip_ts(rows) == strip_ts(golden)

    # The file name's time is the last row's time; rows span the 3s interval before it.
    stamp = datetime.strptime(path.name.split("_")[1], "%Y%m%dT%H%M%SZ")
    times = [datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ") for r in rows]
    assert times == sorted(times) and times[-1] == stamp
    assert stamp - times[0] <= timedelta(seconds=3)


def test_back_to_back_runs_both_succeed(tmp_path):
    # Regression: the second run usually lands in the same second as the first and
    # used to crash with FileExistsError because both numbered their batch 0001.
    data_dir = tmp_path / "data"
    for _ in range(2):
        result = _one_tick(data_dir, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr
    files = sorted((data_dir / "raw").glob("orders_*.csv"))
    assert len(files) == 2
