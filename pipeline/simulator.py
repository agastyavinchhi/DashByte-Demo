"""Order simulator: writes a CSV batch of synthetic DashBite orders to data/raw/ every few seconds.

Run it with ``make simulator`` (``python -u -m pipeline.simulator`` underneath).
Its only output is new files in data/raw/. It never reads, edits or deletes
anything in data/. Some rows are deliberately messy; cleaning is a later stage's job.
"""
from __future__ import annotations

import csv
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline.config import Config, load_config
from pipeline.paths import display_path, ensure_data_dirs, raw_dir

# The raw hand-off contract: header and column order are fixed, values may be messy.
COLUMNS = ("order_id", "timestamp", "distance_km", "prep_minutes", "order_value", "was_late")
FILE_GLOB = "orders_*.csv"
FILE_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
ROW_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The lateness rule later stages learn:
#   estimated delivery = prep + 3 min/km + noise (+ a rush-hour delay at peak hours).
LATE_THRESHOLD_MINUTES = Config.sim_late_threshold_minutes  # default; SIM_LATE_THRESHOLD_MINUTES overrides
MINUTES_PER_KM = 3
NOISE_SD_MINUTES = 4.0
PEAK_DELAY_MINUTES = Config.sim_peak_delay_minutes  # default; SIM_PEAK_DELAY_MINUTES overrides

# Lunch 11:00-13:59 and dinner 17:00-20:59, UTC. Preprocess keeps its own copy
# (stages never import each other); a regression test keeps the two equal.
SIM_PEAK_HOURS = frozenset({11, 12, 13, 17, 18, 19, 20})

# Value ranges. Distances skew short (median ~4 km), as most delivery orders are
# local. At the default 30-minute threshold about 44% of orders are late.
DISTANCE_KM_RANGE = (0.5, 15.0)
DISTANCE_KM_MEDIAN = 4.0
DISTANCE_KM_SPREAD = 0.6
PREP_MINUTES_RANGE = (5, 30)
PREP_MINUTES_MODE = 12
ORDER_VALUE_RANGE = (8.0, 120.0)
ORDER_VALUE_MODE = 28.0

# duplicate_id has no fixed value: the row reuses an earlier order_id in the same batch.
COPY_EARLIER_ID = "<order_id of an earlier row>"

# Every defect a messy row can get: name -> (column, exact value written).
# Tests and the preprocess stage import this instead of retyping the values.
MESSY_DEFECTS = {
    "blank": ("order_value", ""),
    "negative_distance": ("distance_km", "-1.0"),
    "missing_prep": ("prep_minutes", "n/a"),
    "text_label": ("was_late", "yes"),
    "outlier_distance": ("distance_km", "999.0"),
    "duplicate_id": ("order_id", COPY_EARLIER_ID),
}


def to_utc(moment: datetime) -> datetime:
    """The one place times are normalised: naive means UTC, aware is converted to UTC.

    File names and row timestamps both go through here, so they always agree.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def is_sim_peak(moment: datetime) -> bool:
    return to_utc(moment).hour in SIM_PEAK_HOURS


def is_late(
    prep_minutes: float,
    distance_km: float,
    noise: float = 0.0,
    threshold: float = LATE_THRESHOLD_MINUTES,
    peak_delay: float = 0.0,
    at: datetime | None = None,
) -> bool:
    """Apply the delivery rule. ``peak_delay`` counts only when ``at`` is a rush-hour time."""
    estimate = prep_minutes + MINUTES_PER_KM * distance_km + noise
    if at is not None and is_sim_peak(at):
        estimate += peak_delay
    return estimate > threshold


def _new_order_id(rng: random.Random, taken: set[str]) -> str:
    while True:
        order_id = f"ORD-{rng.getrandbits(40):010x}"
        if order_id not in taken:
            return order_id


def _clean_row(
    rng: random.Random,
    order_id: str,
    ts: datetime,
    noise_sd: float,
    late_threshold: float,
    peak_delay: float,
) -> dict[str, str]:
    lo, hi = DISTANCE_KM_RANGE
    distance = round(
        min(hi, max(lo, rng.lognormvariate(math.log(DISTANCE_KM_MEDIAN), DISTANCE_KM_SPREAD))), 1
    )
    prep = int(round(rng.triangular(*PREP_MINUTES_RANGE, PREP_MINUTES_MODE)))
    value = round(rng.triangular(*ORDER_VALUE_RANGE, ORDER_VALUE_MODE), 2)
    noise = rng.gauss(0, noise_sd) if noise_sd > 0 else 0.0
    return {
        "order_id": order_id,
        "timestamp": ts.strftime(ROW_TIME_FORMAT),
        "distance_km": f"{distance:.1f}",
        "prep_minutes": str(prep),
        "order_value": f"{value:.2f}",
        # The rush-hour delay depends on the timestamp alone (no extra random
        # draw), so off-peak rows come out exactly as before it existed.
        "was_late": "1" if is_late(prep, distance, noise, late_threshold, peak_delay, ts) else "0",
    }


def generate_batch(
    rng: random.Random,
    n: int,
    now: datetime,
    messy_rate: float = 0.05,
    *,
    span_seconds: float = 3.0,
    noise_sd: float = NOISE_SD_MINUTES,
    late_threshold: float = LATE_THRESHOLD_MINUTES,
    peak_delay: float = PEAK_DELAY_MINUTES,
) -> list[dict[str, str]]:
    """Return ``n`` order rows as strings, exactly as they will appear in the CSV.

    Timestamps are spread evenly across the ``span_seconds`` before ``now`` and
    end at ``now``. Each row has a ``messy_rate`` chance of one defect from
    ``MESSY_DEFECTS``.
    """
    now = to_utc(now)

    rows: list[dict[str, str]] = []
    taken: set[str] = set()
    for i in range(n):
        ts = now - timedelta(seconds=span_seconds * (n - 1 - i) / n)
        order_id = _new_order_id(rng, taken)
        taken.add(order_id)
        row = _clean_row(rng, order_id, ts, noise_sd, late_threshold, peak_delay)

        if messy_rate > 0 and rng.random() < messy_rate:
            # The first row has nothing earlier to duplicate.
            choices = [d for d in MESSY_DEFECTS if rows or d != "duplicate_id"]
            defect = rng.choice(choices)
            column, value = MESSY_DEFECTS[defect]
            if defect == "duplicate_id":
                value = rng.choice(rows)["order_id"]
            row[column] = value
        rows.append(row)
    return rows


def find_defect(row: dict[str, str], earlier_ids: set[str]) -> str | None:
    """Name the defect in ``row``, or None if it's clean. Every defect value is unambiguous."""
    if row["order_id"] in earlier_ids:
        return "duplicate_id"
    for name, (column, value) in MESSY_DEFECTS.items():
        if value is not COPY_EARLIER_ID and row[column] == value:
            return name
    return None


def count_messy(rows: list[dict[str, str]]) -> int:
    seen: set[str] = set()
    messy = 0
    for row in rows:
        if find_defect(row, seen) is not None:
            messy += 1
        seen.add(row["order_id"])
    return messy


def batch_filename(now: datetime, seq: int) -> str:
    return f"orders_{to_utc(now).strftime(FILE_TIME_FORMAT)}_{seq:04d}.csv"


def write_batch(rows: list[dict[str, str]], out_dir: Path, now: datetime, seq: int) -> Path:
    """Write one batch as ``orders_<UTC time>_<seq>.csv``, all at once.

    The rows go to a hidden ``.tmp`` file that is then renamed into place, so a
    reader never sees half a file. Refuses to overwrite an existing batch.
    """
    final = Path(out_dir) / batch_filename(now, seq)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing batch {final}")

    tmp = final.with_name(f".{final.name}.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


def _log(message: str) -> None:
    print(f"[simulator {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run(
    cfg: Config,
    interval: float,
    max_batches: int | None = None,
    seed: int | None = None,
    messy_rate: float = 0.05,
    *,
    start: datetime | None = None,
) -> tuple[int, int]:
    """Write a batch every ``interval`` seconds until ``max_batches`` or Ctrl+C.

    Batches are stamped with a simulated clock: it starts at ``start`` (default:
    the real time now) and runs ``cfg.sim_clock_speed`` times faster than real
    time, so a demo can sweep through lunch and dinner rush. Speed 1 is real time.

    Returns ``(batches, orders)`` written.
    """
    rng = random.Random(seed)
    ensure_data_dirs(cfg)
    out_dir = raw_dir(cfg)
    speed = cfg.sim_clock_speed
    sim_start = to_utc(start) if start is not None else datetime.now(timezone.utc)
    real_start = time.monotonic()
    batches = orders = 0
    seq = 0
    try:
        while max_batches is None or batches < max_batches:
            elapsed = time.monotonic() - real_start
            now = (sim_start + timedelta(seconds=elapsed * speed)).replace(microsecond=0)
            rows = generate_batch(
                rng, cfg.batch_size, now, messy_rate,
                span_seconds=interval * speed,
                late_threshold=cfg.sim_late_threshold_minutes,
                peak_delay=cfg.sim_peak_delay_minutes,
            )
            # A restart in the same second as the previous run's batch would reuse
            # its name. Step past it: names stay unique and still sort in time order.
            seq += 1
            while (out_dir / batch_filename(now, seq)).exists():
                seq += 1
            path = write_batch(rows, out_dir, now, seq)
            batches += 1
            orders += len(rows)
            _log(
                f"new orders arrived: {len(rows)} ({count_messy(rows)} messy) "
                f"→ {display_path(path)} | total {orders} | sim time {now:%Y-%m-%d %H:%M}Z"
            )
            if max_batches is None or batches < max_batches:
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print(f"[simulator] stopped after {batches} batches, {orders} orders", flush=True)
    return batches, orders


def describe_clock(speed: float) -> str:
    if speed == 1:
        return "clock 1× (real time)"
    minutes_per_day = 24 * 60 / speed
    if minutes_per_day < 60:
        day = f"{minutes_per_day:.1f} min"
    elif minutes_per_day < 48 * 60:
        day = f"{minutes_per_day / 60:.1f} h"
    else:
        day = f"{minutes_per_day / (24 * 60):.1f} days"
    return f"clock {speed:g}× (1 simulated day ≈ {day})"


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

    extras = []
    if cfg.sim_max_batches is not None:
        extras.append(f"up to {cfg.sim_max_batches} batches")
    if cfg.sim_seed is not None:
        extras.append(f"seed {cfg.sim_seed}")
    extra = f", {', '.join(extras)}" if extras else ""
    print(
        f"[simulator] batch size {cfg.batch_size}, every {cfg.sim_interval_seconds:g}s, "
        f"messy rate {cfg.sim_messy_rate:.0%}, late after {cfg.sim_late_threshold_minutes:g} min, "
        f"{describe_clock(cfg.sim_clock_speed)}{extra} → writing to {display_path(raw_dir(cfg))} "
        f"(Ctrl+C to stop)",
        flush=True,
    )
    run(
        cfg,
        interval=cfg.sim_interval_seconds,
        max_batches=cfg.sim_max_batches,
        seed=cfg.sim_seed,
        messy_rate=cfg.sim_messy_rate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
