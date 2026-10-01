"""Golden batch: a fixed seed must keep producing exactly these raw rows.

Preprocess (and the demo's "same seed, same data" promise) depend on the
simulator's output staying byte-for-byte stable: the RNG draw order, the value
formatting and the exact defect values. Seed 2027 at a 50% messy rate gives 20
rows that hit all six defects plus clean rows, both labels included.

If a change to the simulator is deliberate, regenerate and review the diff:

    .venv/bin/python -m tests.regression.test_golden_batch
"""
from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.simulator import COLUMNS, MESSY_DEFECTS, find_defect, generate_batch, write_batch

pytestmark = pytest.mark.regression

GOLDEN = Path(__file__).resolve().parent.parent / "fixtures" / "golden_batch_seed2027.csv"
SEED, N, MESSY_RATE = 2027, 20, 0.5
NOW = datetime(2026, 9, 28, 14, 2, 3, tzinfo=timezone.utc)


def golden_rows() -> list[dict[str, str]]:
    return generate_batch(random.Random(SEED), N, NOW, MESSY_RATE)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def test_seeded_batch_matches_golden_fixture():
    header, expected = read_csv(GOLDEN)
    assert tuple(header) == COLUMNS
    assert golden_rows() == expected


def test_written_file_matches_golden_fixture(tmp_path):
    path = write_batch(golden_rows(), tmp_path, NOW, seq=1)
    assert read_csv(path) == read_csv(GOLDEN)


def test_golden_fixture_covers_every_defect_and_clean_rows():
    # Guards the fixture itself: if it stops exercising a defect, it stops pinning it.
    _, rows = read_csv(GOLDEN)
    seen, found = set(), []
    for row in rows:
        found.append(find_defect(row, seen))
        seen.add(row["order_id"])
    assert set(MESSY_DEFECTS) <= set(found)
    assert None in found
    assert {r["was_late"] for r in rows if r["was_late"] != "yes"} == {"0", "1"}


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    written = write_batch(golden_rows(), GOLDEN.parent, NOW, seq=1)
    written.replace(GOLDEN)
    print(f"wrote {GOLDEN}", file=sys.stderr)
