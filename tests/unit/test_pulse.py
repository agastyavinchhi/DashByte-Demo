from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from pipeline import pulse
from pipeline.config import load_config
from pipeline.pulse import (
    clock_check,
    drop_rate,
    failure_counts,
    failures_headline,
    flag_rate_headline,
    late_flag_rate_over_time,
    load_artifacts,
    sample_volume,
    score_summary,
    volume_headline,
    volume_over_time,
)

pytestmark = pytest.mark.unit

T0 = pd.Timestamp("2026-09-28T14:00:00Z")
MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(minutes=60)


def ts(minutes, seconds=0):
    return T0 + pd.Timedelta(minutes=minutes, seconds=seconds)


def features(rows):
    """rows: (batch, order_id, minute offset or Timestamp)."""
    frame = pd.DataFrame({
        "order_id": [r[1] for r in rows],
        "timestamp": pd.to_datetime([r[2] if isinstance(r[2], pd.Timestamp) else ts(r[2])
                                     for r in rows], utc=True),
        "batch": [r[0] for r in rows],
    })
    # Short labels like "b1" stand in for batch keys where batch time doesn't matter.
    frame["batch_time"] = [pulse.batch_time(b) if re.fullmatch(pulse.BATCH_KEY, b) else T0
                           for b in frame["batch"]]
    return frame.astype(pulse.FEATURES_SCHEMA) if len(rows) else pulse._empty(pulse.FEATURES_SCHEMA)


def predictions(rows):
    """rows: (batch, order_id, probability, flag, checkpoint)."""
    if not rows:
        return pulse._empty(pulse.PREDICTIONS_SCHEMA)
    return pd.DataFrame(rows, columns=["batch", "order_id", "late_probability",
                                       "predicted_late", "checkpoint_id"])


def rejects(rows):
    """rows: (batch, reason)."""
    if not rows:
        return pulse._empty(pulse.REJECTS_SCHEMA)
    frame = pd.DataFrame(rows, columns=["batch", "reason"])
    frame["batch_time"] = frame["batch"].map(pulse.batch_time)
    return frame


def key(minute, seq=1):
    return f"{ts(minute):%Y%m%dT%H%M%SZ}_{seq:04d}"


# ---------- loading ----------

FEATURE_HEADER = "order_id,timestamp,hour,is_peak,distance_km,prep_minutes,order_value,was_late"


def write(path, text, crlf=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\n", "\r\n") if crlf else text
    path.write_bytes(data.encode())
    return path


def feature_file(data, minute, seq, ids):
    k = key(minute, seq)
    body = "".join(f"{i},{ts(minute):%Y-%m-%dT%H:%M:%SZ},14,0,3.0,12,20.00,0\n" for i in ids)
    return write(data / "features" / f"features_{k}.csv", FEATURE_HEADER + "\n" + body)


@pytest.fixture
def cfg(tmp_data_dir):
    return load_config()


def test_missing_folders_give_empty_frames(cfg):
    art = load_artifacts(cfg, pulse.Cache())
    assert art.features.empty and art.predictions.empty and art.rejects.empty
    assert art.newest_model is None
    assert art.counts == {"features": 0, "predictions": 0, "rejects": 0}


def test_tmp_and_header_only_files_add_nothing(cfg, tmp_data_dir):
    feature_file(tmp_data_dir, 0, 1, ["ORD-1", "ORD-2"])
    write(tmp_data_dir / "features" / f".features_{key(1)}.csv.tmp", FEATURE_HEADER + "\nx\n")
    write(tmp_data_dir / "features" / f"features_{key(2, 2)}.csv", FEATURE_HEADER + "\n")
    art = load_artifacts(cfg, pulse.Cache())
    assert art.features["order_id"].tolist() == ["ORD-1", "ORD-2"]
    assert art.counts["features"] == 2  # the header-only file is readable, just empty


def test_unreadable_file_is_skipped_and_logged_once(cfg, tmp_data_dir, capsys):
    feature_file(tmp_data_dir, 0, 1, ["ORD-1"])
    bad = write(tmp_data_dir / "features" / f"features_{key(1, 2)}.csv", "order_id,when\nx,y\n")
    cache = pulse.Cache()
    for _ in range(3):
        art = load_artifacts(cfg, cache)
    assert len(art.features) == 1
    assert capsys.readouterr().out.count(f"skipped {bad.name}: missing column timestamp") == 1


def test_crlf_prediction_files_parse(cfg, tmp_data_dir):
    write(tmp_data_dir / "predictions" / f"predictions_{key(0)}.csv",
          "order_id,late_probability,predicted_late,checkpoint_id\n"
          "ORD-1,0.812,1,model_v0001\nORD-2,0.100,0,model_v0001\n", crlf=True)
    frame = load_artifacts(cfg, pulse.Cache()).predictions
    assert frame["checkpoint_id"].tolist() == ["model_v0001", "model_v0001"]
    assert frame["predicted_late"].tolist() == [1, 0]
    assert frame["late_probability"].tolist() == [0.812, 0.1]


def test_cache_parses_each_file_once_and_picks_up_new_files(cfg, tmp_data_dir, monkeypatch):
    feature_file(tmp_data_dir, 0, 1, ["ORD-1"])
    parsed = []
    real = pulse._parse_features

    def spy(path, key):
        parsed.append(path.name)
        return real(path, key)

    monkeypatch.setattr(pulse, "_parse_features", spy)
    cache = pulse.Cache()
    first = load_artifacts(cfg, cache)
    second = load_artifacts(cfg, cache)
    assert (first.new_files, second.new_files) == (1, 0)
    feature_file(tmp_data_dir, 1, 2, ["ORD-2", "ORD-3"])
    third = load_artifacts(cfg, cache)
    assert third.new_files == 1 and len(third.features) == 3
    assert len(parsed) == 2


def test_batch_and_batch_time_come_from_file_names(cfg, tmp_data_dir):
    feature_file(tmp_data_dir, 5, 3, ["ORD-1"])
    write(tmp_data_dir / "quality" / f"rejects_{key(5, 3)}.csv",
          "order_id,timestamp,distance_km,prep_minutes,order_value,was_late,reason,raw_line\n"
          "ORD-9,garbage,1,1,1,1,bad_timestamp,ORD-9;garbage\n")
    art = load_artifacts(cfg, pulse.Cache())
    assert art.features["batch"].tolist() == [key(5, 3)]
    assert art.rejects["batch_time"].tolist() == [ts(5)]
    assert art.rejects["reason"].tolist() == ["bad_timestamp"]


def test_newest_model_is_highest_readable_sidecar(cfg, tmp_data_dir):
    models = tmp_data_dir / "models"
    models.mkdir(parents=True)
    for v in (1, 2):
        (models / f"model_v{v:04d}.json").write_text(
            json.dumps({"rows_total": 10, "trained_at": f"t{v}"}))
    (models / "model_v0003.json").write_text("{broken")
    (models / "model_v0004.joblib").write_bytes(b"orphan")
    assert pulse.newest_model(cfg) == ("model_v0002", 2, "t2")


# ---------- volume ----------

def test_sample_volume_is_half_open():
    f = features([("b", "a", ts(0)), ("b", "b", ts(59, 59)), ("b", "c", ts(60))])
    assert sample_volume(f, T0, T0 + HOUR) == 2


def test_volume_over_time_buckets_and_zero_fill():
    f = features([("b", "a", 0), ("b", "b", 0.5), ("b", "c", 2), ("b", "d", 59.99)])
    series = volume_over_time(f, T0 + HOUR, HOUR, MIN)
    assert len(series) == 60 and series.index[0] == T0
    assert series.iloc[:4].tolist() == [2, 0, 1, 0]
    assert series.iloc[-1] == 1 and series.sum() == 4


def test_window_is_anchored_to_newest_and_bucket_counts():
    newest = ts(42, 30)
    start, end = pulse.window_bounds(newest, HOUR, MIN)
    assert end == ts(43) and start == ts(-17)
    f = features([("b", "a", newest)])
    assert len(volume_over_time(f, end, HOUR, MIN)) == 60
    start15, end15 = pulse.window_bounds(newest, pd.Timedelta(minutes=1440),
                                         pd.Timedelta(minutes=15))
    assert end15 == ts(45)
    assert len(volume_over_time(f, end15, pd.Timedelta(minutes=1440),
                                pd.Timedelta(minutes=15))) == 96


# ---------- flag rate ----------

def test_flag_rate_join_and_percentages():
    f = features([("b1", "a", 0), ("b1", "b", 0.5), ("b1", "c", 1), ("b1", "d", 1.2)])
    p = predictions([("b1", "a", .9, 1, "m"), ("b1", "b", .1, 0, "m"),
                     ("b1", "c", .8, 1, "m"), ("b1", "d", .7, 1, "m")])
    series = late_flag_rate_over_time(p, f, ts(3), pd.Timedelta(minutes=3), MIN)
    assert series.iloc[0] == 50 and series.iloc[1] == 100
    assert math.isnan(series.iloc[2])  # no predictions is a gap, not 0%
    assert series.attrs["orphans"] == 0


def test_orphan_predictions_are_dropped_and_counted():
    f = features([("b1", "a", 0)])
    p = predictions([("b1", "a", .9, 1, "m"), ("b1", "ghost", .9, 1, "m")])
    series = late_flag_rate_over_time(p, f, ts(1), MIN, MIN)
    assert series.iloc[0] == 100 and series.attrs["orphans"] == 1


def test_same_order_id_in_two_batches_does_not_cross_join():
    f = features([("b1", "same", 0), ("b2", "same", 1)])
    p = predictions([("b1", "same", .9, 1, "m"), ("b2", "same", .1, 0, "m")])
    joined, orphans = pulse.join_predictions(p, f)
    assert len(joined) == 2 and orphans == 0
    series = late_flag_rate_over_time(p, f, ts(2), pd.Timedelta(minutes=2), MIN)
    assert series.tolist() == [100.0, 0.0]


# ---------- summaries ----------

def test_score_summary_on_known_frame():
    p = predictions([("b", str(i), prob, int(prob >= .5), cp) for i, (prob, cp) in enumerate(
        [(.1, "model_v0001"), (.2, "model_v0001"), (.6, "model_v0002"), (.9, "model_v0002"),
         (.95, "model_v0010")])])
    s = score_summary(p)
    assert s["n"] == 5 and s["flag_rate"] == 60.0
    assert s["mean_probability"] == pytest.approx(.55)
    assert s["p50"] == pytest.approx(.6)
    assert s["p10"] == pytest.approx(np.percentile([.1, .2, .6, .9, .95], 10))
    assert s["checkpoints"] == {"model_v0001": 2, "model_v0002": 2, "model_v0010": 1}
    assert s["current_checkpoint"] == "model_v0010"  # by number, not string order


def test_score_summary_empty():
    s = score_summary(predictions([]))
    assert s["n"] == 0 and s["flag_rate"] is None and s["checkpoints"] == {}


def test_drop_rate_windowed_by_batch_time():
    f = features([("x", str(i), 0) for i in range(3)])
    f["batch"] = key(0)
    f["batch_time"] = ts(0)
    r = rejects([(key(0), "blank"), (key(90), "blank")])  # second one is outside the window
    assert drop_rate(f, r, T0, T0 + HOUR) == 25.0
    assert drop_rate(features([]), rejects([]), T0, T0 + HOUR) is None


def test_failure_counts_sorted_labelled_and_windowed():
    reasons = (["out_of_range"] * 3 + ["blank"] * 2 + ["mystery_code"]
               + ["duplicate_id", "not_a_number", "bad_label", "wrong_field_count",
                  "bad_timestamp"])
    r = rejects([(key(0), reason) for reason in reasons] + [(key(120), "blank")] * 9)
    counts = failure_counts(r, T0, T0 + HOUR)
    assert counts.index[:2].tolist() == ["Value out of range", "Missing value"]
    assert counts.tolist() == sorted(counts.tolist(), reverse=True)
    assert set(counts.index) == set(pulse.REASON_LABELS.values()) | {"mystery_code"}
    assert counts["Missing value"] == 2  # the nine outside the window don't count


# ---------- headlines ----------

def series(values, minutes=1):
    index = pd.date_range(T0, periods=len(values), freq=f"{minutes}min")
    return pd.Series(values, index=index, dtype="float64")


@pytest.mark.parametrize("recent,expected", [
    (115, "Orders are steady at ~120 per minute"),   # +15% exactly: still steady (115 rounds to ~120)
    (85, "Orders are steady at ~85 per minute"),     # -15% exactly
    (116, "rose"), (84, "fell"),                     # just outside the band
])
def test_volume_headline_band(recent, expected):
    # Baseline 100/min for 20 buckets; the comparison span is the last 10 complete
    # buckets; the final bucket is still filling and is ignored.
    values = [100] * 10 + [recent] * 10 + [5]
    s = series(values)
    # Choose the earlier buckets so the window mean is exactly 100.
    s.iloc[:10] = 200 - recent
    assert expected in volume_headline(s)


def test_volume_headline_change_wording():
    s = series([100] * 50 + [130] * 10 + [1])
    assert volume_headline(s) == "Orders rose 24% in the last 10 minutes"


def test_volume_headline_stalls():
    s = series([100] * 57 + [0, 0, 0])
    assert volume_headline(s) == "No orders in the last 3 minutes — is the feed running?"
    assert volume_headline(series([100] * 60), idle_minutes=4.2) == (
        "No orders in the last 4 minutes — is the feed running?")
    assert "is the feed running" not in volume_headline(series([100] * 60), idle_minutes=1.9)
    assert "30 hours" in volume_headline(series([100] * 60), idle_minutes=30 * 60 + 5)
    assert "3 days" in volume_headline(series([100] * 60), idle_minutes=3 * 24 * 60)


def test_volume_headline_no_data_and_first_minute():
    assert volume_headline(None) == "No orders yet — start the pipeline with make run"
    assert volume_headline(series([0] * 60)) == volume_headline(None)
    assert volume_headline(series([0] * 59 + [40])) == "Orders started arriving in the last few minutes"
    # Only the half-minute the feed started in is complete so far: still too early to call a rate.
    assert volume_headline(series([0] * 57 + [190, 400])) == "Orders started arriving in the last few minutes"


def test_volume_headline_ignores_empty_stretch_before_the_feed():
    assert volume_headline(series([0] * 45 + [400] * 14 + [100])) == (
        "Orders are steady at ~400 per minute")


def test_volume_headline_names_the_bucket():
    assert volume_headline(series([60] * 12, minutes=15)).endswith("per 15 min")


@pytest.mark.parametrize("recent,expected", [
    (49.9, "The model is flagging a steady 47% of orders late"),  # 4.9 points: steady (window mean)
    (50.0, "The model is flagging more orders late (50% vs 45%)"),  # 5.0 points: called out
    (40.0, "The model is flagging fewer orders late (40% vs 45%)"),
])
def test_flag_rate_headline(recent, expected):
    assert flag_rate_headline(series([45.0] * 20 + [recent] * 10)) == expected


def test_flag_rate_headline_no_data_and_gaps():
    assert flag_rate_headline(None) == "No predictions yet — waiting for the first model"
    assert flag_rate_headline(series([np.nan] * 5)) == flag_rate_headline(None)
    gappy = series([40.0, np.nan] * 15)
    assert flag_rate_headline(gappy) == "The model is flagging a steady 40% of orders late"


def test_failures_headline():
    counts = pd.Series([41, 30, 29], index=["Value out of range", "Missing value", "x"])
    assert failures_headline(counts) == "Out-of-range values cause most dropped rows (41%)"
    assert failures_headline(pd.Series(dtype="int64")) == "No rows dropped in the last 60 minutes"
    assert failures_headline(None, 1440) == "No rows dropped in the last 1440 minutes"
    assert failures_headline(pd.Series([3], index=["new_code"])) == (
        "new_code cause most dropped rows (100%)")


# ---------- clock ----------

@pytest.mark.parametrize("lead,warns", [
    (None, False), (pd.Timedelta(0), False), (pd.Timedelta(minutes=4, seconds=59), False),
    (pd.Timedelta(minutes=5, seconds=1), True), (-pd.Timedelta(hours=6), False),
])
def test_clock_check(lead, warns):
    newest = None if lead is None else T0 + lead
    result = clock_check(newest, T0)
    assert (result is not None) is warns
    if warns:
        # The default clock is 300x, so the fix has to name the real-time setting.
        assert "fast simulated clock" in result and "then `SIM_CLOCK_SPEED=1 make run`" in result


# ---------- the assembled view ----------

def test_build_view_with_no_data(cfg):
    view = pulse.build_view(load_artifacts(cfg, pulse.Cache()), cfg, now=T0)
    assert [k.value for k in view.kpis] == ["—", "—", "—"]
    assert view.volume is view.flag_rate is view.failures is None
    assert view.volume_title == "No orders yet — start the pipeline with make run"
    assert view.flag_title == "No predictions yet — waiting for the first model"
    assert view.failures_title == "No rows dropped in the last 60 minutes"
    assert view.warning is None
    assert "(+0 new)" in view.log_line


def test_steady_flag_headline_quotes_the_kpi_share_when_given():
    # Per-minute rates average to 38.4%, but the KPI counts every order: 39%.
    s = series([38.4] * 30)
    assert flag_rate_headline(s) == "The model is flagging a steady 38% of orders late"
    assert flag_rate_headline(s, window_rate=39.2) == "The model is flagging a steady 39% of orders late"


# ---------- review: KPI deltas only compare like with like ----------

def _steady_feed(minutes, per_minute=10, start_second=0, late_every=2, reject_every=10):
    """A perfectly steady feed: every minute looks the same."""
    f_rows, p_rows, r_rows = [], [], []
    for m in range(minutes):
        for i in range(per_minute):
            seconds = start_second + i * (60 - start_second) // per_minute
            order = f"o{m}-{i}"
            f_rows.append((key(m), order, ts(m, seconds)))
            p_rows.append((key(m), order, 0.5, int(i % late_every == 0), "model_v0001"))
        r_rows += [(key(m), "out_of_range")] * (per_minute // reject_every)
    return pulse.Artifacts(features(f_rows), predictions(p_rows), rejects(r_rows), None,
                           {"features": minutes, "predictions": minutes, "rejects": minutes}, 0)


@pytest.mark.parametrize("minutes", [61, 90, 119])
def test_no_deltas_until_the_previous_window_is_fully_covered(minutes):
    # Regression: at minute 61 a steady feed showed "+23,600 orders" (at real
    # rates) against a previous hour that held one minute of data.
    cfg = load_config({})
    view = pulse.build_view(_steady_feed(minutes), cfg, now=ts(minutes))
    assert [k.delta for k in view.kpis] == [None, None, None]
    assert view.kpis[0].value == "600"  # the KPI itself is still shown


def test_steady_feed_shows_zero_deltas_once_two_windows_exist():
    cfg = load_config({})
    view = pulse.build_view(_steady_feed(121), cfg, now=ts(121))
    assert [k.delta for k in view.kpis] == ["+0", "+0.0 pp", "+0 pp"]


def test_partial_first_minute_still_counts_as_covered():
    # The feed usually starts mid-minute; that shouldn't delay the deltas a whole window.
    cfg = load_config({})
    view = pulse.build_view(_steady_feed(120, start_second=30), cfg, now=ts(120))
    assert view.kpis[0].delta is not None


def test_deltas_report_a_real_change_with_sign_and_colour():
    cfg = load_config({})
    older = _steady_feed(60, reject_every=10)          # 1 of 10 dropped, 50% flagged
    newer = _steady_feed(60, per_minute=20, reject_every=4, late_every=4)
    shifted_f = newer.features.assign(timestamp=newer.features["timestamp"] + HOUR,
                                      batch_time=newer.features["batch_time"] + HOUR)
    shifted_r = newer.rejects.assign(batch_time=newer.rejects["batch_time"] + HOUR)
    shifted_p = newer.predictions.copy()
    for frame in (shifted_f, shifted_r, shifted_p):  # keep batch keys unique across hours
        frame["batch"] = "h2-" + frame["batch"]
    art = pulse.Artifacts(pd.concat([older.features, shifted_f], ignore_index=True),
                          pd.concat([older.predictions, shifted_p], ignore_index=True),
                          pd.concat([older.rejects, shifted_r], ignore_index=True),
                          None, older.counts, 0)
    view = pulse.build_view(art, cfg, now=ts(120))
    orders, dropped, flagged = view.kpis
    assert (orders.value, orders.delta, orders.delta_color) == ("1,200", "+600", "normal")
    # Drop rate per minute: 5 of 25 rows (20.0%) now vs 1 of 11 (9.1%) before.
    assert (dropped.value, dropped.delta, dropped.delta_color) == ("20.0%", "+10.9 pp", "inverse")
    assert (flagged.value, flagged.delta, flagged.delta_color) == ("25%", "-25 pp", "off")
