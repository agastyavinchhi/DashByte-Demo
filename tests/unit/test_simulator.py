from __future__ import annotations

import csv
import random
import re
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import simulator
from pipeline.config import load_config
from pipeline.simulator import (
    COLUMNS,
    COPY_EARLIER_ID,
    MESSY_DEFECTS,
    count_messy,
    generate_batch,
    is_late,
    write_batch,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc)
FILE_RE = re.compile(r"^orders_\d{8}T\d{6}Z_\d{4}\.csv$")


def _batch(n=200, messy_rate=0.0, seed=1, **kw):
    return generate_batch(random.Random(seed), n, NOW, messy_rate, **kw)


def _defects_present(row, earlier_ids):
    """Every defect signature in the row, so 'exactly one' can be checked."""
    found = []
    for name, (column, value) in MESSY_DEFECTS.items():
        if value is COPY_EARLIER_ID:
            if row[column] in earlier_ids:
                found.append(name)
        elif row[column] == value:
            found.append(name)
    return found


def test_returns_n_rows_with_exact_columns_in_order():
    rows = _batch(n=17)
    assert len(rows) == 17
    assert all(tuple(row) == COLUMNS for row in rows)


def test_same_seed_gives_identical_rows():
    assert _batch(seed=42, messy_rate=0.3) == _batch(seed=42, messy_rate=0.3)
    assert _batch(seed=42) != _batch(seed=43)


def test_clean_rows_are_in_range_unique_and_well_formed():
    rows = _batch(n=2000)
    for row in rows:
        assert re.fullmatch(r"ORD-[0-9a-f]{10}", row["order_id"])
        assert 0.5 <= float(row["distance_km"]) <= 15
        assert re.fullmatch(r"\d+\.\d", row["distance_km"])
        assert 5 <= int(row["prep_minutes"]) <= 30
        assert 8 <= float(row["order_value"]) <= 120
        assert re.fullmatch(r"\d+\.\d\d", row["order_value"])
        assert row["was_late"] in {"0", "1"}
    assert len({row["order_id"] for row in rows}) == len(rows)
    assert count_messy(rows) == 0


def test_labels_follow_delivery_rule_when_noise_is_zero():
    for row in _batch(n=2000, noise_sd=0):
        expected = is_late(int(row["prep_minutes"]), float(row["distance_km"]))
        assert row["was_late"] == ("1" if expected else "0")


def test_is_late_rule():
    # Default threshold is 30 minutes.
    assert is_late(prep_minutes=15, distance_km=5.1) is True  # 30.3 > 30
    assert is_late(prep_minutes=15, distance_km=5.0) is False  # exactly 30 is on time
    assert is_late(prep_minutes=15, distance_km=5.0, noise=0.5) is True


def test_is_late_honours_custom_threshold():
    assert is_late(prep_minutes=20, distance_km=5.0, threshold=35) is False  # 35 is on time
    assert is_late(prep_minutes=20, distance_km=5.1, threshold=35) is True


@pytest.mark.parametrize("threshold", [20, 30, 45])
def test_labels_follow_custom_threshold_when_noise_is_zero(threshold):
    rows = _batch(n=2000, noise_sd=0, late_threshold=threshold)
    for row in rows:
        estimate = int(row["prep_minutes"]) + 3 * float(row["distance_km"])
        assert row["was_late"] == ("1" if estimate > threshold else "0")


def test_higher_threshold_means_fewer_late_orders():
    def late_rate(threshold):
        rows = _batch(n=3000, seed=5, late_threshold=threshold)
        return sum(r["was_late"] == "1" for r in rows) / len(rows)

    assert late_rate(25) > late_rate(30) > late_rate(40)


def test_run_uses_threshold_from_config(tmp_data_dir, monkeypatch):
    # With a huge threshold nothing can be late; with a tiny one everything is.
    for value, expected in (("1000", {"0"}), ("0.1", {"1"})):
        monkeypatch.setenv("SIM_LATE_THRESHOLD_MINUTES", value)
        monkeypatch.setenv("SIM_MESSY_RATE", "0")
        raw = tmp_data_dir / "raw"
        for f in raw.glob("*.csv"):
            f.unlink()
        cfg = load_config()
        simulator.run(cfg, interval=0.01, max_batches=1, seed=1, messy_rate=0)
        (path,) = raw.glob("orders_*.csv")
        with open(path, newline="") as fh:
            assert {r["was_late"] for r in csv.DictReader(fh)} == expected


def test_timestamps_are_utc_ascending_and_end_at_now():
    rows = _batch(n=20, span_seconds=10)
    stamps = [datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ") for r in rows]
    assert all(r["timestamp"].endswith("Z") for r in rows)
    assert stamps == sorted(stamps)
    naive_now = NOW.replace(tzinfo=None)
    assert stamps[-1] == naive_now
    assert stamps[0] >= naive_now - timedelta(seconds=10)


def test_messy_rate_one_gives_exactly_one_exact_defect_per_row():
    rows = _batch(n=3000, messy_rate=1.0)
    assert len(rows) == 3000  # duplicate_id reuses an id, never adds a row

    seen_ids, seen_defects = set(), set()
    for row in rows:
        found = _defects_present(row, seen_ids)
        assert len(found) == 1, (row, found)
        name = found[0]
        column, value = MESSY_DEFECTS[name]
        if value is not COPY_EARLIER_ID:
            assert row[column] == value
        seen_defects.add(name)
        seen_ids.add(row["order_id"])

    assert seen_defects == set(MESSY_DEFECTS)
    # Only the exact defect values ever appear, never a variant.
    distances = {float(r["distance_km"]) for r in rows if r["distance_km"]}
    assert {d for d in distances if d < 0} <= {-1.0}
    assert {d for d in distances if d > 15} <= {999.0}
    assert count_messy(rows) == 3000


def test_first_row_never_gets_duplicate_id():
    for seed in range(200):
        rows = generate_batch(random.Random(seed), 1, NOW, messy_rate=1.0)
        assert count_messy(rows) == 1


def test_messy_rate_roughly_respected():
    rows = _batch(n=5000, messy_rate=0.05, seed=3)
    assert 0.03 < count_messy(rows) / len(rows) < 0.07


def test_write_batch_round_trips_and_names_file(tmp_path):
    rows = _batch(n=20, messy_rate=0.5)
    path = write_batch(rows, tmp_path, NOW, seq=3)

    assert path.name == "orders_20260928T140203Z_0003.csv"
    assert FILE_RE.match(path.name)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == COLUMNS
        assert list(reader) == rows
    assert list(tmp_path.glob("*.tmp")) == list(tmp_path.glob(".*.tmp")) == []


def test_write_batch_refuses_to_overwrite(tmp_path):
    first = write_batch(_batch(n=5), tmp_path, NOW, seq=1)
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        write_batch(_batch(n=5, seed=9), tmp_path, NOW, seq=1)
    assert first.read_bytes() == before


def test_write_batch_leaves_no_tmp_file_on_failure(tmp_path):
    with pytest.raises(ValueError):
        write_batch([{"not_a_column": "x"}], tmp_path, NOW, seq=1)
    assert list(tmp_path.iterdir()) == []


def test_run_writes_max_batches_then_stops(tmp_data_dir, capsys):
    cfg = load_config()
    batches, orders = simulator.run(cfg, interval=0.01, max_batches=3, seed=1)
    assert (batches, orders) == (3, 3 * cfg.batch_size)
    files = sorted((tmp_data_dir / "raw").glob("orders_*.csv"))
    assert [f.name[-8:] for f in files] == ["0001.csv", "0002.csv", "0003.csv"]
    out = capsys.readouterr().out
    assert out.count("new orders arrived: 20 (") == 3
    assert "| total 60" in out
    assert "stopped after 3 batches, 60 orders" in out


def test_run_ctrl_c_stops_cleanly(tmp_data_dir, monkeypatch, capsys):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(simulator.time, "sleep", interrupt)
    batches, orders = simulator.run(load_config(), interval=5, seed=1)
    assert (batches, orders) == (1, 20)
    assert "stopped after 1 batches, 20 orders" in capsys.readouterr().out
    assert list((tmp_data_dir / "raw").glob(".*.tmp")) == []


# The same instant, written three ways: naive (treated as UTC), UTC, and UTC-4.
SAME_INSTANT = [
    datetime(2026, 9, 28, 14, 2, 3),
    datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc),
    datetime(2026, 9, 28, 10, 2, 3, tzinfo=timezone(timedelta(hours=-4))),
]


@pytest.mark.parametrize("moment", SAME_INSTANT, ids=["naive", "utc", "utc-4"])
def test_to_utc_normalises_every_input_the_same_way(moment):
    assert simulator.to_utc(moment) == NOW
    assert simulator.to_utc(moment).tzinfo == timezone.utc


@pytest.mark.parametrize("moment", SAME_INSTANT, ids=["naive", "utc", "utc-4"])
def test_filename_and_row_timestamps_agree(tmp_path, moment):
    # Regression: batch_filename used to read a naive time as local time while
    # write_batch and generate_batch read it as UTC, so they could disagree.
    rows = generate_batch(random.Random(1), 5, moment, 0)
    assert simulator.batch_filename(moment, 1) == "orders_20260928T140203Z_0001.csv"
    path = write_batch(rows, tmp_path, moment, seq=1)
    assert path.name == "orders_20260928T140203Z_0001.csv"
    assert rows[-1]["timestamp"] == "2026-09-28T14:02:03Z"


def test_filename_time_matches_last_row_in_live_run(tmp_data_dir):
    simulator.run(load_config(), interval=0.01, max_batches=2, seed=1)
    for path in (tmp_data_dir / "raw").glob("orders_*.csv"):
        with open(path, newline="") as fh:
            last = list(csv.DictReader(fh))[-1]["timestamp"]
        stamp = path.name.split("_")[1]  # 20260928T140203Z
        assert datetime.strptime(stamp, "%Y%m%dT%H%M%SZ") == datetime.strptime(
            last, "%Y-%m-%dT%H:%M:%SZ"
        )


class _FrozenDatetime(datetime):
    """datetime whose now() is always NOW, so two runs land in the same second."""

    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def test_restart_in_same_second_does_not_crash_or_overwrite(tmp_data_dir, monkeypatch):
    # Regression: seq restarted at 1 on every run, so a second run in the same
    # second hit write_batch's overwrite guard and died with a traceback.
    monkeypatch.setattr(simulator, "datetime", _FrozenDatetime)
    monkeypatch.setenv("SIM_CLOCK_SPEED", "1")  # real time, so both runs share one second
    cfg = load_config()
    raw = tmp_data_dir / "raw"

    simulator.run(cfg, interval=0.01, max_batches=1, seed=1)
    first = raw / "orders_20260928T140203Z_0001.csv"
    before = first.read_bytes()

    assert simulator.run(cfg, interval=0.01, max_batches=2, seed=2) == (2, 40)
    names = sorted(p.name for p in raw.glob("orders_*.csv"))
    assert names == [f"orders_20260928T140203Z_{i:04d}.csv" for i in (1, 2, 3)]
    assert first.read_bytes() == before


# ---------- Stage 3: rush-hour delay ----------

def _at(hh, mm=0):
    return datetime(2026, 9, 28, hh, mm, tzinfo=timezone.utc)


def test_is_late_adds_peak_delay_only_at_peak_hours():
    # 12 + 3*5.0 = 27 minutes: on time, unless 8 rush-hour minutes push it past 30.
    assert is_late(12, 5.0, peak_delay=8, at=_at(10, 59)) is False
    assert is_late(12, 5.0, peak_delay=8, at=_at(11, 0)) is True
    assert is_late(12, 5.0, peak_delay=8, at=_at(14, 0)) is False
    assert is_late(12, 5.0, peak_delay=8, at=_at(20, 59)) is True
    assert is_late(12, 5.0, peak_delay=8, at=_at(21, 0)) is False
    # Exactly peak_delay minutes: 27 + 3 = 30 is still on time, + 3.1 is late.
    assert is_late(12, 5.0, peak_delay=3, at=_at(18)) is False
    assert is_late(12, 5.0, peak_delay=3.1, at=_at(18)) is True
    # No time given, no delay (the Stage 2 call shape).
    assert is_late(12, 5.0, peak_delay=8) is False


@pytest.mark.parametrize("hh", range(24))
def test_zero_peak_delay_gives_stage2_labels_at_every_hour(hh):
    # Stage 2 labels never depended on the hour. With the delay off, a batch at
    # any hour must match the same seed at 03:30 in everything but the timestamp.
    strip = lambda rows: [{k: v for k, v in r.items() if k != "timestamp"} for r in rows]  # noqa: E731
    at_hour = generate_batch(random.Random(4), 300, _at(hh, 30), 0.1, peak_delay=0)
    at_night = generate_batch(random.Random(4), 300, _at(3, 30), 0.1, peak_delay=0)
    assert strip(at_hour) == strip(at_night)


def test_peak_delay_does_not_change_the_random_sequence():
    base = generate_batch(random.Random(9), 500, _at(18, 30), 0.2, peak_delay=0)
    delayed = generate_batch(random.Random(9), 500, _at(18, 30), 0.2, peak_delay=8)
    keep = ("order_id", "timestamp", "distance_km", "prep_minutes", "order_value")
    assert [{k: r[k] for k in keep} for r in base] == [{k: r[k] for k in keep} for r in delayed]
    late = lambda rows: sum(r["was_late"] == "1" for r in rows)  # noqa: E731
    assert late(delayed) > late(base)


# ---------- Stage 3: simulated clock ----------

class _FakeClock:
    """Stands in for time.monotonic/time.sleep: sleeping advances the clock instantly."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _run_with_fake_clock(monkeypatch, speed, batches=3, interval=3.0):
    clock = _FakeClock()
    monkeypatch.setattr(simulator.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(simulator.time, "sleep", clock.sleep)
    monkeypatch.setenv("SIM_CLOCK_SPEED", str(speed))
    simulator.run(load_config(), interval=interval, max_batches=batches, seed=1,
                  start=_at(10, 0))


def _batches(raw):
    out = []
    for path in sorted(raw.glob("orders_*.csv")):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        stamp = datetime.strptime(path.name.split("_")[1], "%Y%m%dT%H%M%SZ")
        times = [datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ") for r in rows]
        out.append((stamp, times))
    return out


def test_clock_speed_300_advances_15_simulated_minutes_per_3s_batch(tmp_data_dir, monkeypatch, capsys):
    _run_with_fake_clock(monkeypatch, speed=300)
    batches = _batches(tmp_data_dir / "raw")
    stamps = [stamp for stamp, _ in batches]
    assert stamps == [datetime(2026, 9, 28, 10, 0), datetime(2026, 9, 28, 10, 15),
                      datetime(2026, 9, 28, 10, 30)]
    for stamp, times in batches:
        assert times[-1] == stamp  # file name agrees with the last row
        assert times == sorted(times)
        assert stamp - timedelta(minutes=15) <= times[0] < stamp - timedelta(minutes=14)
    out = capsys.readouterr().out
    assert "| sim time 2026-09-28 10:15Z" in out


def test_clock_speed_1_is_stage2_real_time(tmp_data_dir, monkeypatch):
    _run_with_fake_clock(monkeypatch, speed=1)
    batches = _batches(tmp_data_dir / "raw")
    assert [stamp for stamp, _ in batches] == [
        datetime(2026, 9, 28, 10, 0, 0), datetime(2026, 9, 28, 10, 0, 3),
        datetime(2026, 9, 28, 10, 0, 6),
    ]
    for stamp, times in batches:
        assert times[-1] == stamp and stamp - times[0] <= timedelta(seconds=3)


def test_describe_clock():
    assert simulator.describe_clock(300) == "clock 300× (1 simulated day ≈ 4.8 min)"
    assert simulator.describe_clock(1) == "clock 1× (real time)"
    assert simulator.describe_clock(3600) == "clock 3600× (1 simulated day ≈ 0.4 min)"
    assert simulator.describe_clock(10) == "clock 10× (1 simulated day ≈ 2.4 h)"
