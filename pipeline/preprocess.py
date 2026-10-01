"""Preprocess: turns each raw order batch into a clean, model-ready features file.

Run it with ``make preprocess`` (``python -u -m pipeline.preprocess`` underneath).
It watches data/raw/ and, for every ``orders_<stamp>_<seq>.csv`` there, writes:

- ``data/quality/rejects_<stamp>_<seq>.csv``: every dropped row plus its reason
- ``data/features/features_<stamp>_<seq>.csv``: the kept rows with ``hour`` and
  ``is_peak`` added and ``was_late`` kept as the label (written last, so its
  existence means the batch is done)

It talks to the simulator only through data/: it never imports another stage,
and it never edits or deletes anything in data/raw/.
"""
from __future__ import annotations

import csv
import math
import os
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from pipeline.config import Config, load_config
from pipeline.paths import display_path, ensure_data_dirs, features_dir, quality_dir, raw_dir

# Input contract: what the simulator writes. Defined here, not imported; a
# regression test keeps these equal to the simulator's.
RAW_GLOB = "orders_*.csv"
RAW_COLUMNS = ("order_id", "timestamp", "distance_km", "prep_minutes", "order_value", "was_late")

# Output contract: what training reads. was_late stays last, as the label.
FEATURE_COLUMNS = (
    "order_id", "timestamp", "hour", "is_peak",
    "distance_km", "prep_minutes", "order_value", "was_late",
)
# raw_line is the row exactly as it appeared in the raw file (minus the line
# ending), so a malformed row's extra or missing fields are still on record.
REJECT_COLUMNS = RAW_COLUMNS + ("reason", "raw_line")

# Reject reasons, in the order the rules are checked. The first failing rule wins.
REJECT_REASONS = (
    "duplicate_id", "wrong_field_count", "blank", "bad_timestamp", "not_a_number",
    "out_of_range", "bad_label",
)

# DashBite's plausibility bounds, exclusive low and inclusive high: low < value <= high.
# Deliberately looser than what the simulator generates; they're about what a
# real order could be, not about one generator's ranges.
VALID_RANGES = {
    "distance_km": (0.0, 50.0),
    "prep_minutes": (0, 120),
    "order_value": (0.0, 1000.0),
}

# Lunch 11:00-13:59 and dinner 17:00-20:59, UTC.
PEAK_HOURS = frozenset({11, 12, 13, 17, 18, 19, 20})

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_TIMESTAMP_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class BatchResult(NamedTuple):
    features_path: Path
    rejects_path: Path
    kept: int
    reasons: Counter  # reason -> rejected count

    @property
    def rejected(self) -> int:
        return sum(self.reasons.values())


def _read_raw(fh) -> list[tuple[dict[str, str | None], str]]:
    """Parse a raw batch into ``(row, raw_line)`` pairs.

    Each row is shaped like csv.DictReader's (missing fields are None, extras
    sit under the key None), so ``clean_row`` sees exactly what it always has.
    ``raw_line`` is the original text of that record, taken from the lines the
    parser consumed for it, so a quoted field spanning lines is kept whole.
    """
    consumed: list[str] = []

    def lines():
        for line in fh:
            consumed.append(line)
            yield line

    reader = csv.reader(lines())
    if tuple(next(reader, None) or ()) != RAW_COLUMNS:
        raise SkippedFile("unexpected header")

    records = []
    while True:
        consumed.clear()
        fields = next(reader, None)
        if fields is None:
            return records
        if not fields:  # a blank line; DictReader skips these too
            continue
        row: dict[str | None, str | None] = dict.fromkeys(RAW_COLUMNS)
        row.update(zip(RAW_COLUMNS, fields))
        if len(fields) > len(RAW_COLUMNS):
            row[None] = fields[len(RAW_COLUMNS):]
        records.append((row, "".join(consumed).rstrip("\r\n")))


class SkippedFile(Exception):
    """A raw file preprocess can't use at all (bad header, vanished). Logged, never fatal."""


def _parse_timestamp(value: str) -> datetime | None:
    if not _TIMESTAMP_SHAPE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:  # right shape, impossible date such as month 13
        return None


def _finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _whole_number(value: str) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _in_range(column: str, value: float) -> bool:
    low, high = VALID_RANGES[column]
    return low < value <= high


def _wrong_field_count(row: dict) -> bool:
    """csv.DictReader puts extra values under the key None and fills missing ones with None."""
    return None in row or any(row.get(col) is None for col in RAW_COLUMNS)


def clean_row(
    row: dict[str, str | None], seen_ids: set[str]
) -> tuple[dict[str, str] | None, str | None]:
    """Validate one raw row, in file order.

    Returns ``(features, None)`` for a kept row or ``(None, reason)`` for a
    rejected one. ``seen_ids`` is updated: the first occurrence of an id claims
    it even if that row is rejected for another reason.
    """
    values = {col: (row.get(col) or "").strip() for col in RAW_COLUMNS}

    order_id = values["order_id"]
    if order_id:
        if order_id in seen_ids:
            return None, "duplicate_id"
        seen_ids.add(order_id)

    # Too many or too few fields means the columns are shifted: no value can be trusted.
    if _wrong_field_count(row):
        return None, "wrong_field_count"

    if any(v == "" for v in values.values()):
        return None, "blank"

    moment = _parse_timestamp(values["timestamp"])
    if moment is None:
        return None, "bad_timestamp"

    distance = _finite_float(values["distance_km"])
    prep = _whole_number(values["prep_minutes"])
    value = _finite_float(values["order_value"])
    if distance is None or prep is None or value is None:
        return None, "not_a_number"

    if not (
        _in_range("distance_km", distance)
        and _in_range("prep_minutes", prep)
        and _in_range("order_value", value)
    ):
        return None, "out_of_range"

    if values["was_late"] not in ("0", "1"):
        return None, "bad_label"

    return {
        "order_id": order_id,
        "timestamp": values["timestamp"],
        "hour": str(moment.hour),
        "is_peak": "1" if moment.hour in PEAK_HOURS else "0",
        "distance_km": f"{distance:.1f}",
        "prep_minutes": str(prep),
        "order_value": f"{value:.2f}",
        "was_late": values["was_late"],
    }, None


def batch_key(raw_path: Path) -> str:
    """``orders_<stamp>_<seq>.csv`` -> ``<stamp>_<seq>``, shared by all three files."""
    return raw_path.name[len("orders_"):-len(".csv")]


def features_path_for(raw_path: Path, cfg: Config) -> Path:
    return features_dir(cfg) / f"features_{batch_key(raw_path)}.csv"


def rejects_path_for(raw_path: Path, cfg: Config) -> Path:
    return quality_dir(cfg) / f"rejects_{batch_key(raw_path)}.csv"


def pending_files(cfg: Config) -> list[Path]:
    """Raw batches with no features file yet, oldest first. The folders are the state."""
    return sorted(
        (p for p in raw_dir(cfg).glob(RAW_GLOB) if p.is_file()
         and not features_path_for(p, cfg).exists()),
        key=lambda p: p.name,
    )


def _write_csv_atomic(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def process_file(raw_path: Path, cfg: Config) -> BatchResult:
    """Clean one raw batch. Writes rejects, then features (last = done marker).

    Raises ``SkippedFile`` if the file can't be used at all; nothing is written then.
    """
    try:
        with open(raw_path, newline="", encoding="utf-8") as fh:
            raw_rows = _read_raw(fh)
    except FileNotFoundError:
        raise SkippedFile("file disappeared") from None
    # Anything else that makes the file unreadable is the file's problem, not the
    # loop's: skip it rather than crash (and crash again on every restart).
    except UnicodeDecodeError:
        raise SkippedFile("not UTF-8 text") from None
    except csv.Error as exc:
        raise SkippedFile(f"malformed CSV ({exc})") from None
    except OSError as exc:
        raise SkippedFile(f"unreadable ({exc.strerror or exc})") from None

    kept: list[dict[str, str]] = []
    rejects: list[dict[str, str]] = []
    reasons: Counter = Counter()
    seen_ids: set[str] = set()
    for row, raw_line in raw_rows:
        features, reason = clean_row(row, seen_ids)
        if features is not None:
            kept.append(features)
        else:
            rejects.append({
                **{c: row.get(c) or "" for c in RAW_COLUMNS},
                "reason": reason,
                "raw_line": raw_line,
            })
            reasons[reason] += 1

    rejects_path = rejects_path_for(raw_path, cfg)
    features_path = features_path_for(raw_path, cfg)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(rejects_path, REJECT_COLUMNS, rejects)
    _write_csv_atomic(features_path, FEATURE_COLUMNS, kept)
    return BatchResult(features_path, rejects_path, len(kept), reasons)


def _log(message: str) -> None:
    print(f"[preprocess {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _describe_reasons(reasons: Counter) -> str:
    if not reasons:
        return ""
    parts = [f"{r} {reasons[r]}" for r in REJECT_REASONS if reasons[r]]
    return f" ({', '.join(parts)})"


def run(cfg: Config, poll_seconds: float, once: bool = False) -> tuple[int, int, int]:
    """Process every pending raw batch, then poll for more until Ctrl+C.

    With ``once=True``, do a single pass and return. Returns ``(files, kept, rejected)``.
    """
    ensure_data_dirs(cfg)
    files = kept = rejected = 0
    skipped: set[str] = set()  # bad files are logged once per run, not every poll
    first_pass = True
    try:
        while True:
            pending = [p for p in pending_files(cfg) if p.name not in skipped]
            if first_pass and pending:
                noun = "batch" if len(pending) == 1 else "batches"
                _log(f"found {len(pending)} pending {noun}")
            first_pass = False

            for raw_path in pending:
                try:
                    result = process_file(raw_path, cfg)
                except SkippedFile as exc:
                    skipped.add(raw_path.name)
                    _log(f"skipped {raw_path.name}: {exc}")
                    continue
                files += 1
                kept += result.kept
                rejected += result.rejected
                _log(
                    f"{raw_path.name} → {result.features_path.name}: kept {result.kept}, "
                    f"rejected {result.rejected}{_describe_reasons(result.reasons)} "
                    f"| total kept {kept}, rejected {rejected}"
                )

            if once:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        pass
    print(
        f"[preprocess] stopped after {files} batches: kept {kept}, rejected {rejected}",
        flush=True,
    )
    return files, kept, rejected


def _sigterm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    # `make stop` sends SIGTERM; give it the same clean exit as Ctrl+C.
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    how = "(one pass)" if cfg.preprocess_once else "(Ctrl+C to stop)"
    print(
        f"[preprocess] watching {display_path(raw_dir(cfg))} every "
        f"{cfg.preprocess_poll_seconds:g}s → writing {display_path(features_dir(cfg))} "
        f"(rejects → {display_path(quality_dir(cfg))}) {how}",
        flush=True,
    )
    run(cfg, poll_seconds=cfg.preprocess_poll_seconds, once=cfg.preprocess_once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
