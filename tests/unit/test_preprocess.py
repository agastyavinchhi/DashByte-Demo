from __future__ import annotations

import csv
import os
import re

import pytest

from pipeline import preprocess
from pipeline.config import load_config
from pipeline.preprocess import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    clean_row,
    pending_files,
    process_file,
)
from pipeline.simulator import COPY_EARLIER_ID, MESSY_DEFECTS

pytestmark = pytest.mark.unit

CLEAN = {
    "order_id": "ORD-3f9a1c02be",
    "timestamp": "2026-09-28T14:02:01Z",
    "distance_km": "6.4",
    "prep_minutes": "18",
    "order_value": "42.50",
    "was_late": "1",
}

# Which reason each simulator defect must be rejected with (see the plan's table).
DEFECT_TO_REASON = {
    "blank": "blank",
    "negative_distance": "out_of_range",
    "missing_prep": "not_a_number",
    "text_label": "bad_label",
    "outlier_distance": "out_of_range",
    "duplicate_id": "duplicate_id",
}


def _clean(**changes):
    return clean_row({**CLEAN, **changes}, set())


def _write_raw(raw_dir, name, rows, header=RAW_COLUMNS):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in header])
    return path


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return tuple(reader.fieldnames), list(reader)


# ---------- clean_row ----------

def test_clean_row_keeps_values_and_label():
    features, reason = _clean()
    assert reason is None
    assert tuple(features) == FEATURE_COLUMNS
    assert features == {
        "order_id": "ORD-3f9a1c02be",
        "timestamp": "2026-09-28T14:02:01Z",
        "hour": "14",
        "is_peak": "0",
        "distance_km": "6.4",
        "prep_minutes": "18",
        "order_value": "42.50",
        "was_late": "1",
    }


def test_clean_row_writes_fixed_number_format():
    features, _ = _clean(distance_km=" 6.40 ", prep_minutes="18.0", order_value="42.5")
    assert (features["distance_km"], features["prep_minutes"], features["order_value"]) == (
        "6.4", "18", "42.50",
    )


@pytest.mark.parametrize("defect", sorted(MESSY_DEFECTS))
def test_every_simulator_defect_is_rejected_with_its_reason(defect):
    column, value = MESSY_DEFECTS[defect]
    seen = set()
    if value is COPY_EARLIER_ID:
        clean_row(dict(CLEAN), seen)  # the earlier row claims the id
        value = CLEAN["order_id"]
    features, reason = clean_row({**CLEAN, column: value}, seen)
    assert features is None
    assert reason == DEFECT_TO_REASON[defect]


@pytest.mark.parametrize("changes,reason", [
    ({"timestamp": "2026-09-28 14:02:01"}, "bad_timestamp"),
    ({"timestamp": "2026-13-28T14:02:01Z"}, "bad_timestamp"),
    ({"timestamp": "yesterday"}, "bad_timestamp"),
    ({"distance_km": "   "}, "blank"),
    ({"order_id": ""}, "blank"),
    ({"prep_minutes": "12.5"}, "not_a_number"),
    ({"distance_km": "far"}, "not_a_number"),
    ({"order_value": "nan"}, "not_a_number"),
    ({"was_late": "2"}, "bad_label"),
    ({"was_late": "true"}, "bad_label"),
    ({"distance_km": "0"}, "out_of_range"),
    ({"distance_km": "50.1"}, "out_of_range"),
    ({"prep_minutes": "0"}, "out_of_range"),
    ({"prep_minutes": "121"}, "out_of_range"),
    ({"order_value": "0"}, "out_of_range"),
    ({"order_value": "1000.01"}, "out_of_range"),
])
def test_rules_beyond_simulator_defects(changes, reason):
    assert _clean(**changes) == (None, reason)


@pytest.mark.parametrize("changes", [
    {"distance_km": "50"}, {"prep_minutes": "120"}, {"order_value": "1000"},
    {"distance_km": "0.1"}, {"prep_minutes": "1"}, {"order_value": "0.01"},
])
def test_upper_bounds_are_inclusive(changes):
    features, reason = _clean(**changes)
    assert reason is None and features is not None


def test_first_failing_rule_wins():
    # Blank and out of range: blank is checked first.
    assert _clean(order_value="", distance_km="-1.0")[1] == "blank"
    # Not a number and bad label: not_a_number is checked first.
    assert _clean(prep_minutes="n/a", was_late="yes")[1] == "not_a_number"


def test_duplicates_keep_first_reject_later():
    seen = set()
    assert clean_row(dict(CLEAN), seen)[1] is None
    assert clean_row(dict(CLEAN), seen) == (None, "duplicate_id")
    assert clean_row({**CLEAN, "distance_km": "2.0"}, seen) == (None, "duplicate_id")


def test_invalid_first_occurrence_still_claims_the_id():
    seen = set()
    assert clean_row({**CLEAN, "distance_km": "999.0"}, seen) == (None, "out_of_range")
    assert clean_row(dict(CLEAN), seen) == (None, "duplicate_id")


@pytest.mark.parametrize("hh", range(24))
def test_hour_is_the_utc_hh(hh):
    features, _ = _clean(timestamp=f"2026-09-28T{hh:02d}:30:00Z")
    assert features["hour"] == str(hh)


@pytest.mark.parametrize("hhmm,expected", [
    ("10:59", "0"), ("11:00", "1"), ("13:59", "1"), ("14:00", "0"),
    ("16:59", "0"), ("17:00", "1"), ("20:59", "1"), ("21:00", "0"), ("00:00", "0"),
])
def test_is_peak_boundaries(hhmm, expected):
    features, _ = _clean(timestamp=f"2026-09-28T{hhmm}:00Z")
    assert features["is_peak"] == expected


# ---------- files ----------

NAME = "orders_20260928T140203Z_0003.csv"


def test_process_file_writes_name_matched_outputs(tmp_data_dir):
    cfg = load_config()
    rows = [CLEAN, {**CLEAN, "order_id": "ORD-0000000002", "distance_km": "-1.0"},
            {**CLEAN, "order_id": "ORD-0000000003", "was_late": "0"}]
    raw = _write_raw(tmp_data_dir / "raw", NAME, rows)

    result = process_file(raw, cfg)

    assert result.features_path == tmp_data_dir / "features" / "features_20260928T140203Z_0003.csv"
    assert result.rejects_path == tmp_data_dir / "quality" / "rejects_20260928T140203Z_0003.csv"
    header, kept = _read(result.features_path)
    assert header == FEATURE_COLUMNS and [r["order_id"] for r in kept] == [
        "ORD-3f9a1c02be", "ORD-0000000003",
    ]
    header, rejected = _read(result.rejects_path)
    # raw_line added deliberately: the rejected row exactly as it arrived.
    assert header == RAW_COLUMNS + ("reason", "raw_line")
    assert rejected == [{
        **rows[1],
        "reason": "out_of_range",
        "raw_line": ",".join(rows[1][c] for c in RAW_COLUMNS),
    }]
    assert (result.kept, result.rejected) == (2, 1)
    assert result.kept + result.rejected == len(rows)
    assert not list(tmp_data_dir.rglob("*.tmp"))


def test_rejects_file_is_header_only_when_nothing_rejected(tmp_data_dir):
    raw = _write_raw(tmp_data_dir / "raw", NAME, [CLEAN])
    result = process_file(raw, load_config())
    assert _read(result.rejects_path) == (RAW_COLUMNS + ("reason", "raw_line"), [])


def test_features_written_last_and_no_tmp_left_on_failure(tmp_data_dir, monkeypatch):
    raw = _write_raw(tmp_data_dir / "raw", NAME, [CLEAN])
    real_replace = os.replace

    def fail_on_features(src, dst):
        if "features_" in str(dst):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(preprocess.os, "replace", fail_on_features)
    with pytest.raises(OSError):
        process_file(raw, load_config())
    assert not list(tmp_data_dir.rglob("*.tmp"))
    assert not list((tmp_data_dir / "features").glob("*"))
    assert pending_files(load_config()) == [raw]  # so it's redone next time


def test_process_file_skips_bad_header(tmp_data_dir):
    header = ("order_id", "when", "distance_km", "prep_minutes", "order_value", "was_late")
    raw = _write_raw(tmp_data_dir / "raw", NAME, [], header=header)
    with pytest.raises(preprocess.SkippedFile, match="unexpected header"):
        process_file(raw, load_config())
    assert not list((tmp_data_dir / "features").glob("*"))
    assert not list((tmp_data_dir / "quality").glob("*"))


def test_process_file_skips_vanished_file(tmp_data_dir):
    load_config()
    with pytest.raises(preprocess.SkippedFile, match="disappeared"):
        process_file(tmp_data_dir / "raw" / NAME, load_config())


_HEADER = ",".join(RAW_COLUMNS).encode()
_GOOD_LINE = b"ORD-1,2026-09-28T14:02:01Z,6.4,18,42.50,1"
UNREADABLE = {
    # Regression: each of these used to escape process_file and crash the loop
    # (and crash again on every restart, since the file stays pending).
    "latin1": (_HEADER + b"\nORD-\xe9,2026-09-28T14:02:01Z,6.4,18,42.50,1\n", "not UTF-8"),
    "nul_in_row": (_HEADER + b"\n" + _GOOD_LINE + b"\x00\n", "malformed CSV"),
    "nul_in_header": (_HEADER + b"\x00\n" + _GOOD_LINE + b"\n", "malformed CSV"),
}


@pytest.mark.parametrize("content,message", UNREADABLE.values(), ids=list(UNREADABLE))
def test_process_file_skips_unreadable_file(tmp_data_dir, content, message):
    raw = tmp_data_dir / "raw" / NAME
    raw.parent.mkdir(parents=True)
    raw.write_bytes(content)
    with pytest.raises(preprocess.SkippedFile, match=message):
        process_file(raw, load_config())
    assert not list((tmp_data_dir / "features").glob("*"))
    assert not list((tmp_data_dir / "quality").glob("*"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_process_file_skips_file_without_read_permission(tmp_data_dir):
    raw = _write_raw(tmp_data_dir / "raw", NAME, [CLEAN])
    raw.chmod(0)
    try:
        with pytest.raises(preprocess.SkippedFile, match="unreadable"):
            process_file(raw, load_config())
    finally:
        raw.chmod(0o644)


def test_run_survives_unreadable_files_and_processes_the_rest(tmp_data_dir, capsys):
    cfg = load_config()
    raw = tmp_data_dir / "raw"
    raw.mkdir(parents=True)
    for i, (content, _) in enumerate(UNREADABLE.values(), start=1):
        (raw / f"orders_20260928T14000{i}Z_0001.csv").write_bytes(content)
    good = _write_raw(raw, "orders_20260928T150000Z_0001.csv", [CLEAN])

    assert preprocess.run(cfg, poll_seconds=0.01, once=True) == (1, 1, 0)
    out = capsys.readouterr().out
    assert out.count("skipped orders_") == len(UNREADABLE)
    assert (tmp_data_dir / "features" / f"features_{good.name[7:]}").exists()


def test_pending_files_lists_unprocessed_sorted_and_ignores_others(tmp_data_dir):
    cfg = load_config()
    raw = tmp_data_dir / "raw"
    later = _write_raw(raw, "orders_20260928T150000Z_0002.csv", [CLEAN])
    earlier = _write_raw(raw, "orders_20260928T140000Z_0001.csv", [CLEAN])
    (raw / ".orders_20260928T160000Z_0003.csv.tmp").write_text("half a file")
    (raw / "notes.txt").write_text("not a batch")
    (raw / "orders_20260928T170000Z_0004.txt").write_text("wrong extension")

    assert pending_files(cfg) == [earlier, later]
    process_file(earlier, cfg)
    assert pending_files(cfg) == [later]


def test_processing_twice_is_a_no_op(tmp_data_dir, capsys):
    cfg = load_config()
    _write_raw(tmp_data_dir / "raw", NAME, [CLEAN])
    assert preprocess.run(cfg, poll_seconds=0.01, once=True) == (1, 1, 0)
    features = tmp_data_dir / "features" / "features_20260928T140203Z_0003.csv"
    mtime = features.stat().st_mtime_ns

    assert preprocess.run(cfg, poll_seconds=0.01, once=True) == (0, 0, 0)
    assert features.stat().st_mtime_ns == mtime


# ---------- run loop ----------

def test_run_once_processes_backlog_and_logs(tmp_data_dir, capsys):
    cfg = load_config()
    raw = tmp_data_dir / "raw"
    _write_raw(raw, "orders_20260928T140000Z_0001.csv", [CLEAN])
    _write_raw(raw, "orders_20260928T141500Z_0002.csv",
               [CLEAN, {**CLEAN, "order_id": "ORD-2", "distance_km": "999.0"}])

    assert preprocess.run(cfg, poll_seconds=0.01, once=True) == (2, 2, 1)
    out = capsys.readouterr().out
    assert "found 2 pending batches" in out
    assert ("orders_20260928T141500Z_0002.csv → features_20260928T141500Z_0002.csv: "
            "kept 1, rejected 1 (out_of_range 1) | total kept 2, rejected 1") in out
    assert "stopped after 2 batches: kept 2, rejected 1" in out


def test_run_skips_bad_file_once_and_keeps_going(tmp_data_dir, capsys, monkeypatch):
    cfg = load_config()
    raw = tmp_data_dir / "raw"
    _write_raw(raw, "orders_20260928T140000Z_0001.csv", [],
               header=("a", "b", "c", "d", "e", "f"))
    good = _write_raw(raw, "orders_20260928T141500Z_0002.csv", [CLEAN])

    polls = []

    def fake_sleep(_seconds):
        polls.append(1)
        if len(polls) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(preprocess.time, "sleep", fake_sleep)
    assert preprocess.run(cfg, poll_seconds=0.01) == (1, 1, 0)
    out = capsys.readouterr().out
    assert out.count("skipped orders_20260928T140000Z_0001.csv: unexpected header") == 1
    assert (tmp_data_dir / "features" / f"features_{good.name[7:]}").exists()


def test_run_idle_is_silent_and_ctrl_c_stops_cleanly(tmp_data_dir, capsys, monkeypatch):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(preprocess.time, "sleep", interrupt)
    assert preprocess.run(load_config(), poll_seconds=5) == (0, 0, 0)
    out = capsys.readouterr().out
    assert out.strip() == "[preprocess] stopped after 0 batches: kept 0, rejected 0"
    assert not list(tmp_data_dir.rglob("*.tmp"))


# ---------- wrong field count (review fix: extra fields were silently dropped) ----------

HEADER_LINE = ",".join(RAW_COLUMNS)
GOOD_LINE = "ORD-0000000001,2026-09-28T14:02:01Z,6.4,18,20.00,0"


def _write_lines(raw_dir, lines):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / NAME
    path.write_text("\r\n".join([HEADER_LINE, *lines]) + "\r\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("line", [
    "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,EXTRA",   # one extra value
    "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,",        # trailing comma
    "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00",           # one value short
    "ORD-0000000002",                                             # almost everything missing
], ids=["extra", "trailing-comma", "short", "id-only"])
def test_wrong_field_count_is_rejected_not_kept(tmp_data_dir, line):
    raw = _write_lines(tmp_data_dir / "raw", [GOOD_LINE, line])
    result = process_file(raw, load_config())

    assert (result.kept, result.rejected) == (1, 1)
    assert result.reasons == {"wrong_field_count": 1}
    _, kept = _read(result.features_path)
    assert [r["order_id"] for r in kept] == ["ORD-0000000001"]
    _, rejected = _read(result.rejects_path)
    assert [(r["order_id"], r["reason"]) for r in rejected] == [
        ("ORD-0000000002", "wrong_field_count"),
    ]


def test_wrong_field_count_row_still_claims_its_id():
    seen = set()
    extra = {**CLEAN, None: ["EXTRA"]}  # how csv.DictReader presents an extra value
    assert clean_row(extra, seen) == (None, "wrong_field_count")
    assert clean_row(dict(CLEAN), seen) == (None, "duplicate_id")


def test_wrong_field_count_checked_before_value_rules():
    # A shifted row can't be judged on its values, so the field count wins.
    short = {**CLEAN, "was_late": None, "distance_km": "-1.0"}
    assert clean_row(short, set()) == (None, "wrong_field_count")


def test_log_line_names_wrong_field_count(tmp_data_dir, capsys):
    _write_lines(tmp_data_dir / "raw", [GOOD_LINE, GOOD_LINE.replace("01,", "02,", 1) + ",EXTRA"])
    preprocess.run(load_config(), poll_seconds=0.01, once=True)
    assert "kept 1, rejected 1 (wrong_field_count 1)" in capsys.readouterr().out


@pytest.mark.parametrize("line,reason", [
    # A properly quoted comma is one field: the count is right, the value is judged on its own.
    ('ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,"1,234.00",0', "not_a_number"),
    # The same value unquoted splits into two fields and shifts the label column.
    ("ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,1,234.00,0", "wrong_field_count"),
    ("ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,a,b", "wrong_field_count"),
    ("   ", "wrong_field_count"),
    # Six fields with an empty last one is a blank, not a count problem.
    ("ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,", "blank"),
], ids=["quoted-comma", "unquoted-comma", "two-extra", "whitespace-only", "empty-last-field"])
def test_field_count_edge_cases(tmp_data_dir, line, reason):
    result = process_file(_write_lines(tmp_data_dir / "raw", [GOOD_LINE, line]), load_config())
    assert (result.kept, dict(result.reasons)) == (1, {reason: 1})


def test_quoted_values_with_right_field_count_are_kept(tmp_data_dir):
    line = '"ORD-0000000002","2026-09-28T14:02:02Z","6.4","18","20.00","1"'
    result = process_file(_write_lines(tmp_data_dir / "raw", [line]), load_config())
    assert (result.kept, result.rejected) == (1, 0)


def test_wrong_field_count_rows_never_reach_features(tmp_data_dir):
    # Many malformed rows mixed with good ones: every kept row has exactly the feature columns.
    lines = []
    for i in range(1, 21):
        good = f"ORD-{i:010d},2026-09-28T14:02:02Z,6.4,18,20.00,0"
        lines.append(good + (",EXTRA" if i % 3 == 0 else ""))
    result = process_file(_write_lines(tmp_data_dir / "raw", lines), load_config())
    assert (result.kept, dict(result.reasons)) == (14, {"wrong_field_count": 6})
    header, kept = _read(result.features_path)
    assert tuple(header) == FEATURE_COLUMNS
    assert all(None not in r and len(r) == len(FEATURE_COLUMNS) for r in kept)
    assert not any(int(r["order_id"][4:]) % 3 == 0 for r in kept)


# ---------- raw_line: rejects keep the row exactly as it arrived ----------

def _raw_lines(result):
    _, rejected = _read(result.rejects_path)
    return [r["raw_line"] for r in rejected]


def test_raw_line_keeps_extra_and_missing_fields(tmp_data_dir):
    extra = "ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,20.00,0,EXTRA,MORE"
    short = "ORD-0000000003,2026-09-28T14:02:03Z,6.4"
    result = process_file(_write_lines(tmp_data_dir / "raw", [GOOD_LINE, extra, short]),
                          load_config())
    assert _raw_lines(result) == [extra, short]
    # The per-column copy can't tell a short row from blanks; raw_line can.
    _, rejected = _read(result.rejects_path)
    assert rejected[1]["order_value"] == "" and rejected[1]["raw_line"].count(",") == 2


def test_raw_line_is_verbatim_including_quotes_and_spaces(tmp_data_dir):
    quoted = '"ORD-0000000002", 2026-09-28T14:02:02Z ,6.4,18,"1,234.00",0'
    result = process_file(_write_lines(tmp_data_dir / "raw", [quoted]), load_config())
    assert _raw_lines(result) == [quoted]


def test_raw_line_strips_only_the_line_ending(tmp_data_dir):
    # _write_lines uses CRLF; neither \r nor \n may leak into raw_line.
    bad = "ORD-0000000002,2026-09-28T14:02:02Z,-1.0,18,20.00,0"
    result = process_file(_write_lines(tmp_data_dir / "raw", [bad]), load_config())
    assert _raw_lines(result) == [bad]


def test_raw_line_keeps_multiline_record_whole_and_stays_aligned(tmp_data_dir):
    # A quoted field may contain a newline: that record spans two physical lines,
    # and the rows after it must still get their own raw_line, not a neighbour's.
    multiline = 'ORD-0000000002,2026-09-28T14:02:02Z,6.4,18,"20\r\n.00",0'  # not_a_number
    after = "ORD-0000000003,2026-09-28T14:02:03Z,999.0,18,20.00,0"
    raw = _write_lines(tmp_data_dir / "raw", [multiline, GOOD_LINE, after])
    result = process_file(raw, load_config())
    assert (result.kept, result.rejected) == (1, 2)
    assert _raw_lines(result) == [multiline, after]


def test_blank_lines_are_skipped_and_do_not_shift_raw_lines(tmp_data_dir):
    first = "ORD-0000000002,2026-09-28T14:02:02Z,-1.0,18,20.00,0"
    second = "ORD-0000000003,2026-09-28T14:02:03Z,999.0,18,20.00,0"
    result = process_file(_write_lines(tmp_data_dir / "raw", ["", first, "", "", second]),
                          load_config())
    assert _raw_lines(result) == [first, second]


def test_rows_seen_by_clean_row_are_unchanged(tmp_data_dir, monkeypatch):
    # Switching away from csv.DictReader must not change what the rules see.
    import io

    text = "\r\n".join([
        HEADER_LINE, GOOD_LINE, "ORD-2,2026-09-28T14:02:02Z,6.4,18,20.00,0,EXTRA",
        "", "ORD-3,2026-09-28T14:02:03Z", '"ORD-4","x",,"n/a",1,yes',
    ]) + "\r\n"
    expected = list(csv.DictReader(io.StringIO(text, newline="")))
    got = [row for row, _ in preprocess._read_raw(io.StringIO(text, newline=""))]
    assert got == expected
