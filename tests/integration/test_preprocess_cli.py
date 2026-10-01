"""Simulator and preprocess as separate real processes, handing off only through data/."""
from __future__ import annotations

import csv
import shutil
import signal
import subprocess
import sys
import time

import pytest

from pipeline.config import PROJECT_ROOT
from tests.conftest import clean_env

pytestmark = pytest.mark.integration

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _env(data_dir, **extra):
    return clean_env(PYTHONPATH=str(PROJECT_ROOT), DASHBITE_DATA_DIR=str(data_dir), **extra)


def _run(module, data_dir, cwd, **extra):
    return subprocess.run(
        [sys.executable, "-u", "-m", module],
        cwd=cwd, env=_env(data_dir, **extra), capture_output=True, text=True, timeout=60,
    )


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _keys(folder, prefix):
    return sorted(p.name[len(prefix):] for p in folder.glob(f"{prefix}*.csv"))


def test_simulator_then_preprocess_end_to_end(tmp_path):
    data = tmp_path / "data"
    sim = _run("pipeline.simulator", data, tmp_path, SIM_MAX_BATCHES="3", BATCH_SIZE="10",
               SIM_SEED="7", SIM_MESSY_RATE="0.5", SIM_INTERVAL_SECONDS="0.1")
    assert sim.returncode == 0, sim.stderr

    pre = _run("pipeline.preprocess", data, tmp_path, PREPROCESS_ONCE="1")
    assert pre.returncode == 0, pre.stderr
    batch_lines = [l for l in pre.stdout.splitlines() if " → features_" in l]
    assert len(batch_lines) == 3
    assert all(": kept " in l and " rejected " in l for l in batch_lines)

    raw_keys = _keys(data / "raw", "orders_")
    assert len(raw_keys) == 3
    assert _keys(data / "features", "features_") == raw_keys
    assert _keys(data / "quality", "rejects_") == raw_keys
    for key in raw_keys:
        kept = _read(data / "features" / f"features_{key}")
        rejected = _read(data / "quality" / f"rejects_{key}")
        assert len(kept) + len(rejected) == 10
        for row in kept:
            assert 0 <= int(row["hour"]) <= 23
            assert row["is_peak"] in {"0", "1"}
            assert row["was_late"] in {"0", "1"}
    assert not list(data.rglob("*.tmp"))


def test_simulated_day_shows_rush_hour_after_preprocessing(tmp_path):
    data = tmp_path / "data"
    # 0.1s x 18000 = 30 simulated minutes per batch, so 48 batches cover 24 hours.
    # (The plan's 3600 would give 6 minutes per batch: only 4.8 hours.)
    sim = _run("pipeline.simulator", data, tmp_path, SIM_CLOCK_SPEED="18000",
               SIM_INTERVAL_SECONDS="0.1", SIM_MAX_BATCHES="48", BATCH_SIZE="20",
               SIM_MESSY_RATE="0", SIM_SEED="11")
    assert sim.returncode == 0, sim.stderr
    pre = _run("pipeline.preprocess", data, tmp_path, PREPROCESS_ONCE="1")
    assert pre.returncode == 0, pre.stderr

    rows = [r for p in (data / "features").glob("features_*.csv") for r in _read(p)]
    assert len({r["hour"] for r in rows}) >= 20
    assert {r["is_peak"] for r in rows} == {"0", "1"}

    def late_rate(flag):
        picked = [r["was_late"] == "1" for r in rows if r["is_peak"] == flag]
        return sum(picked) / len(picked)

    assert late_rate("1") > late_rate("0")


def test_live_loop_picks_up_new_batch_and_stops_on_sigterm(tmp_path):
    data = tmp_path / "data"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pipeline.preprocess"],
        cwd=tmp_path, env=_env(data, PREPROCESS_POLL_SECONDS="0.2"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # Preprocess is already watching before any raw data exists.
        deadline = time.monotonic() + 10
        while not (data / "raw").is_dir():
            assert time.monotonic() < deadline and proc.poll() is None, "preprocess didn't start"
            time.sleep(0.05)

        sim = _run("pipeline.simulator", data, tmp_path, SIM_MAX_BATCHES="1",
                   SIM_INTERVAL_SECONDS="0.1")
        assert sim.returncode == 0, sim.stderr
        (raw_file,) = (data / "raw").glob("orders_*.csv")
        features = data / "features" / f"features_{raw_file.name[len('orders_'):]}"

        deadline = time.monotonic() + 5
        while not features.exists():
            assert time.monotonic() < deadline, "features file didn't appear within 5s"
            time.sleep(0.05)

        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "stopped after 1 batches" in out
    assert "Traceback" not in err
    assert not list(data.rglob("*.tmp"))


def test_restart_resumes_and_changes_nothing(tmp_path):
    data = tmp_path / "data"
    assert _run("pipeline.simulator", data, tmp_path, SIM_MAX_BATCHES="2",
                SIM_INTERVAL_SECONDS="0.1").returncode == 0
    first = _run("pipeline.preprocess", data, tmp_path, PREPROCESS_ONCE="1")
    assert "found 2 pending batches" in first.stdout
    snapshot = {p: p.stat().st_mtime_ns for p in data.rglob("*.csv")}

    second = _run("pipeline.preprocess", data, tmp_path, PREPROCESS_ONCE="1")
    assert second.returncode == 0, second.stderr
    assert "stopped after 0 batches" in second.stdout
    assert "pending" not in second.stdout
    assert {p: p.stat().st_mtime_ns for p in data.rglob("*.csv")} == snapshot


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")
def test_make_preprocess_wiring(tmp_path):
    data = tmp_path / "data"
    assert _run("pipeline.simulator", data, tmp_path, SIM_MAX_BATCHES="1",
                SIM_INTERVAL_SECONDS="0.1").returncode == 0

    def make(**extra):
        return subprocess.run(
            ["make", "-f", str(PROJECT_ROOT / "Makefile"), "preprocess"],
            cwd=tmp_path, env=clean_env(DASHBITE_DATA_DIR=str(data), **extra),
            capture_output=True, text=True, timeout=60,
        )

    ok = make(PREPROCESS_ONCE="1")
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[0].startswith("[preprocess] watching ")
    assert len(list((data / "features").glob("features_*.csv"))) == 1

    bad = make(PREPROCESS_POLL_SECONDS="0")
    assert bad.returncode != 0
    assert "PREPROCESS_POLL_SECONDS" in bad.stderr


def test_unreadable_raw_file_is_skipped_not_fatal(tmp_path):
    # Regression: a non-UTF-8 raw file crashed preprocess with a traceback, and
    # it crashed again on every restart because the file stayed pending.
    data_dir = tmp_path / "data"
    assert _run("pipeline.simulator", data_dir, tmp_path,
                SIM_MAX_BATCHES="1", SIM_INTERVAL_SECONDS="0.1").returncode == 0
    bad = data_dir / "raw" / "orders_19990101T000000Z_0001.csv"
    bad.write_bytes(b"order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
                    b"ORD-\xe9,1999-01-01T00:00:00Z,1.0,10,20.00,0\n")

    result = _run("pipeline.preprocess", data_dir, tmp_path, PREPROCESS_ONCE="1")
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert f"skipped {bad.name}: not UTF-8 text" in result.stdout
    assert len(_keys(data_dir / "features", "features_")) == 1


def test_extra_field_row_is_rejected_end_to_end(tmp_path):
    # Review fix: a row with more fields than the header used to be kept, extras dropped.
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    name = "orders_20260928T140203Z_0001.csv"
    (raw / name).write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "ORD-0000000001,2026-09-28T14:02:01Z,6.4,18,20.00,0\n"
        "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,EXTRA\n"
        "ORD-0000000003,2026-09-28T14:02:03Z,6.4,18,20.00\n",
        encoding="utf-8",
    )

    result = _run("pipeline.preprocess", data_dir, tmp_path, PREPROCESS_ONCE="1")
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "kept 1, rejected 2 (wrong_field_count 2)" in result.stdout

    kept = _read(data_dir / "features" / "features_20260928T140203Z_0001.csv")
    assert [r["order_id"] for r in kept] == ["ORD-0000000001"]
    rejected = _read(data_dir / "quality" / "rejects_20260928T140203Z_0001.csv")
    assert [(r["order_id"], r["reason"]) for r in rejected] == [
        ("ORD-0000000002", "wrong_field_count"),
        ("ORD-0000000003", "wrong_field_count"),
    ]
    # The rejects file keeps each bad row exactly as it arrived, extras and all.
    assert [r["raw_line"] for r in rejected] == [
        "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,EXTRA",
        "ORD-0000000003,2026-09-28T14:02:03Z,6.4,18,20.00",
    ]
