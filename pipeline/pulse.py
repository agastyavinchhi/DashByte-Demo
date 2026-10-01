"""Model Pulse helpers: everything the dashboard shows, computed from data/.

Pure and read-only. Nothing here imports Streamlit, writes a file, or imports
another stage. Each contract it reads (columns, file names, reason codes, the
"newest model" rule) is defined here, and regression tests keep it equal to the
producer's. ``pipeline/dashboard.py`` only lays out what ``build_view`` returns.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.paths import display_path, features_dir, models_dir, predictions_dir, quality_dir

# ---------- contracts (kept equal to the producers by regression tests) ----------

FEATURES_GLOB = "features_*.csv"
PREDICTIONS_GLOB = "predictions_*.csv"
REJECTS_GLOB = "rejects_*.csv"
BATCH_KEY = r"\d{8}T\d{6}Z_\d{4,}"  # <stamp>_<seq>, shared by all three batch files
BATCH_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FEATURE_COLUMNS = ("order_id", "timestamp")  # the only feature columns the page needs
PREDICTION_COLUMNS = ("order_id", "late_probability", "predicted_late", "checkpoint_id")
REJECT_COLUMNS = ("reason",)
MODEL_SIDECAR_PATTERN = re.compile(r"model_v(\d{4,})\.json")

REASON_LABELS = {
    "out_of_range": "Value out of range",
    "blank": "Missing value",
    "not_a_number": "Not a number",
    "duplicate_id": "Duplicate order ID",
    "bad_label": "Late flag not 0/1",
    "wrong_field_count": "Wrong number of fields",
    "bad_timestamp": "Unreadable timestamp",
}
# The same reasons as the subject of a sentence, for the failures headline.
REASON_PHRASES = {
    "Value out of range": "Out-of-range values",
    "Missing value": "Missing values",
    "Not a number": "Non-numeric values",
    "Duplicate order ID": "Duplicate order IDs",
    "Late flag not 0/1": "Invalid late flags",
    "Wrong number of fields": "Rows with the wrong number of fields",
    "Unreadable timestamp": "Unreadable timestamps",
}

# ---------- tuning ----------

RECENT_BUCKETS = 10  # "the last 10 minutes" at 1-minute buckets
STEADY_BAND = 0.15  # recent volume within ±15% of the window mean reads as steady
FLAG_CHANGE_PP = 5  # a flag-rate move of 5 percentage points or more is called out
STALL_MINUTES = 2  # no orders for this long reads as a stalled feed
FAST_CLOCK_LEAD = pd.Timedelta(minutes=5)

FAST_CLOCK_WARNING = (
    "The order feed is on a fast simulated clock, so per-minute charts will be sparse. "
    "For the demo, restart it at real time (the default): `make stop`, `make clean-data`, "
    "then `make run`."
)
_EPS = 1e-9


def _log(message: str) -> None:
    print(f"[dashboard {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ---------- loading ----------

def _empty(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


FEATURES_SCHEMA = {"order_id": "object", "timestamp": "datetime64[ns, UTC]",
                   "batch": "object", "batch_time": "datetime64[ns, UTC]"}
PREDICTIONS_SCHEMA = {"order_id": "object", "late_probability": "float64",
                      "predicted_late": "int64", "checkpoint_id": "object", "batch": "object"}
REJECTS_SCHEMA = {"reason": "object", "batch": "object", "batch_time": "datetime64[ns, UTC]"}


class Unreadable(Exception):
    """A batch file the page can't use. Skipped and logged once; never fatal."""


def batch_key(name: str, prefix: str) -> str | None:
    match = re.fullmatch(rf"{prefix}({BATCH_KEY})\.csv", name)
    return match.group(1) if match else None


def batch_time(key: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(key.split("_")[0], BATCH_TIME_FORMAT), tz="UTC")


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise Unreadable("empty file") from None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise Unreadable(f"unreadable ({type(exc).__name__})") from None
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise Unreadable(f"missing column {', '.join(missing)}")
    return frame[list(columns)]


def _parse_features(path: Path, key: str) -> pd.DataFrame:
    frame = _read(path, FEATURE_COLUMNS)
    timestamps = pd.to_datetime(frame["timestamp"], format=TIMESTAMP_FORMAT, utc=True,
                                errors="coerce")
    if timestamps.isna().any():
        raise Unreadable("unreadable timestamp")
    return pd.DataFrame({"order_id": frame["order_id"], "timestamp": timestamps,
                         "batch": key, "batch_time": batch_time(key)})


def _parse_predictions(path: Path, key: str) -> pd.DataFrame:
    frame = _read(path, PREDICTION_COLUMNS)
    probability = pd.to_numeric(frame["late_probability"], errors="coerce")
    flag = pd.to_numeric(frame["predicted_late"], errors="coerce")
    if probability.isna().any() or not flag.isin([0, 1]).all():
        raise Unreadable("non-numeric prediction")
    return pd.DataFrame({"order_id": frame["order_id"], "late_probability": probability,
                         "predicted_late": flag.astype("int64"),
                         "checkpoint_id": frame["checkpoint_id"], "batch": key})


def _parse_rejects(path: Path, key: str) -> pd.DataFrame:
    frame = _read(path, REJECT_COLUMNS)
    return pd.DataFrame({"reason": frame["reason"], "batch": key,
                         "batch_time": batch_time(key)})


@dataclass
class Cache:
    """Parsed batch files, kept for the life of the dashboard process.

    Batch files are written once and never change, so each is parsed once
    (re-parsed only if its modification time or size changes).
    """
    files: dict = field(default_factory=dict)  # (kind, name) -> (identity, frame | None)
    combined: dict = field(default_factory=dict)  # kind -> (frozenset of (name, identity), frame)
    logged: set = field(default_factory=set)

    def log_once(self, message: str) -> None:
        if message not in self.logged:
            self.logged.add(message)
            _log(message)


class Folder(NamedTuple):
    kind: str
    path: Path
    prefix: str
    parse: Callable[[Path, str], pd.DataFrame]
    schema: dict


def _load_folder(folder: Folder, cache: Cache) -> tuple[pd.DataFrame, int, int]:
    """(combined frame, readable files, newly parsed files) for one batch folder."""
    present = {}
    for path in folder.path.glob(f"{folder.prefix}*.csv"):
        key = batch_key(path.name, folder.prefix)
        if key is None or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue  # vanished between glob and stat
        present[path.name] = (key, path, (stat.st_mtime_ns, stat.st_size))

    new = 0
    for name, (key, path, identity) in present.items():
        cached = cache.files.get((folder.kind, name))
        if cached is not None and cached[0] == identity:
            continue
        try:
            frame = folder.parse(path, key)
            new += 1
        except Unreadable as reason:
            frame = None
            cache.log_once(f"skipped {name}: {reason}")
        cache.files[(folder.kind, name)] = (identity, frame)

    readable = {name: (identity, cache.files[(folder.kind, name)][1])
                for name, (_, _, identity) in present.items()
                if cache.files[(folder.kind, name)][1] is not None}
    signature = frozenset((name, identity) for name, (identity, _) in readable.items())

    previous = cache.combined.get(folder.kind)
    if previous is not None and previous[0] == signature:
        combined = previous[1]
    elif previous is not None and previous[0] <= signature:
        added = [readable[name][1] for name, _ in signature - previous[0]]
        combined = pd.concat([previous[1], *added], ignore_index=True)
    else:
        frames = [frame for _, frame in readable.values()]
        combined = pd.concat(frames, ignore_index=True) if frames else _empty(folder.schema)
    if combined.empty:
        combined = _empty(folder.schema)
    cache.combined[folder.kind] = (signature, combined)
    return combined, len(readable), new


class ModelInfo(NamedTuple):
    checkpoint_id: str
    version: int
    trained_at: str | None


def _readable_sidecar(path: Path) -> dict | None:
    """Same rule as train's latest_published and infer: valid JSON with an integer rows_total."""
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        int(sidecar["rows_total"])
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        return None
    return sidecar


def newest_model(cfg: Config) -> ModelInfo | None:
    """The highest version with a readable sidecar. Sidecars only; never a .joblib."""
    versions = []
    for path in models_dir(cfg).glob("model_v*.json"):
        match = MODEL_SIDECAR_PATTERN.fullmatch(path.name)
        if match:
            versions.append((int(match.group(1)), path))
    for version, path in sorted(versions, reverse=True):
        sidecar = _readable_sidecar(path)
        if sidecar is not None:
            return ModelInfo(f"model_v{version:04d}", version, sidecar.get("trained_at"))
    return None


class Artifacts(NamedTuple):
    features: pd.DataFrame
    predictions: pd.DataFrame
    rejects: pd.DataFrame
    newest_model: ModelInfo | None
    counts: dict  # folder kind -> readable files
    new_files: int  # files parsed for the first time on this call


def load_artifacts(cfg: Config, cache: Cache) -> Artifacts:
    """Read everything the page needs from data/, read-only. Missing folders give empty frames."""
    folders = (
        Folder("features", features_dir(cfg), "features_", _parse_features, FEATURES_SCHEMA),
        Folder("predictions", predictions_dir(cfg), "predictions_", _parse_predictions,
               PREDICTIONS_SCHEMA),
        Folder("rejects", quality_dir(cfg), "rejects_", _parse_rejects, REJECTS_SCHEMA),
    )
    frames, counts, new_files = {}, {}, 0
    for folder in folders:
        frames[folder.kind], counts[folder.kind], new = _load_folder(folder, cache)
        new_files += new
    return Artifacts(frames["features"], frames["predictions"], frames["rejects"],
                     newest_model(cfg), counts, new_files)


# ---------- time windows ----------

def window_bounds(newest: pd.Timestamp, window: pd.Timedelta,
                  bucket: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The window ending with the bucket that holds the newest order (anchored to data, not the wall clock)."""
    end = newest.floor(bucket) + bucket
    return end - window, end


def _bucket_index(start: pd.Timestamp, end: pd.Timestamp, bucket: pd.Timedelta) -> pd.DatetimeIndex:
    index = pd.date_range(start, end, freq=bucket, inclusive="left")
    index.name = "time"
    return index


def _in(times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (times >= start) & (times < end)


def sample_volume(features: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Clean orders with a timestamp in [start, end)."""
    return int(_in(features["timestamp"], start, end).sum())


def volume_over_time(features: pd.DataFrame, end: pd.Timestamp, window: pd.Timedelta,
                     bucket: pd.Timedelta) -> pd.Series:
    """Orders per bucket across the window. An empty bucket is 0: the feed stalled."""
    start = end - window
    index = _bucket_index(start, end, bucket)
    times = features.loc[_in(features["timestamp"], start, end), "timestamp"]
    slots = ((times - start) // bucket).astype("int64")
    counts = slots.value_counts().reindex(range(len(index)), fill_value=0)
    return pd.Series(counts.to_numpy(dtype="int64"), index=index, name="orders")


def join_predictions(predictions: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Predictions with their order timestamp, joined one-to-one on (batch, order_id).

    Returns ``(joined, orphans)``: orphans are predictions with no feature row,
    which means a broken join and is worth logging.
    """
    keys = ["batch", "order_id"]
    lookup = features[[*keys, "timestamp"]].drop_duplicates(keys)
    joined = predictions.merge(lookup, on=keys, how="inner", validate="many_to_one")
    return joined, len(predictions) - len(joined)


def late_flag_rate_over_time(predictions: pd.DataFrame, features: pd.DataFrame,
                             end: pd.Timestamp, window: pd.Timedelta,
                             bucket: pd.Timedelta) -> pd.Series:
    """% of predictions flagged late per bucket (0-100). A bucket with no predictions is NaN.

    The orphan count from the join is in ``series.attrs["orphans"]``.
    """
    joined, orphans = join_predictions(predictions, features)
    start = end - window
    index = _bucket_index(start, end, bucket)
    inside = joined[_in(joined["timestamp"], start, end)]
    slots = ((inside["timestamp"] - start) // bucket).astype("int64")
    rates = inside["predicted_late"].groupby(slots).mean() * 100
    series = pd.Series(rates.reindex(range(len(index))).to_numpy(dtype="float64"),
                       index=index, name="flagged_late_pct")
    series.attrs["orphans"] = orphans
    return series


def _checkpoint_version(checkpoint_id: str) -> int:
    match = re.fullmatch(r"model_v(\d+)", str(checkpoint_id))
    return int(match.group(1)) if match else -1


def score_summary(predictions: pd.DataFrame) -> dict:
    """Counts and shares of what the model said. Empty input gives n=0 and Nones."""
    if predictions.empty:
        return {"n": 0, "flag_rate": None, "mean_probability": None, "p10": None,
                "p50": None, "p90": None, "checkpoints": {}, "current_checkpoint": None}
    probability = predictions["late_probability"]
    checkpoints = predictions["checkpoint_id"].value_counts()
    return {
        "n": int(len(predictions)),
        "flag_rate": float(predictions["predicted_late"].mean() * 100),
        "mean_probability": float(probability.mean()),
        "p10": float(np.percentile(probability, 10)),
        "p50": float(np.percentile(probability, 50)),
        "p90": float(np.percentile(probability, 90)),
        "checkpoints": {str(k): int(v) for k, v in checkpoints.items()},
        "current_checkpoint": max(checkpoints.index, key=_checkpoint_version),
    }


def drop_rate(features: pd.DataFrame, rejects: pd.DataFrame, start: pd.Timestamp,
              end: pd.Timestamp) -> float | None:
    """Rejected / (kept + rejected) as a %, over batches stamped in [start, end). None if no rows."""
    kept = int(_in(features["batch_time"], start, end).sum())
    rejected = int(_in(rejects["batch_time"], start, end).sum())
    total = kept + rejected
    return None if total == 0 else rejected / total * 100


def failure_counts(rejects: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Dropped rows per plain-English reason in the window, most common first."""
    reasons = rejects.loc[_in(rejects["batch_time"], start, end), "reason"]
    labels = reasons.map(lambda r: REASON_LABELS.get(r, r))  # unknown codes stay visible
    counts = labels.value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return pd.Series([int(c) for _, c in ordered], index=[label for label, _ in ordered],
                     name="rows_dropped", dtype="int64")


# ---------- headlines: each chart's title states its finding ----------

def bucket_minutes_of(series: pd.Series) -> int:
    if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 1:
        return max(1, int((series.index[1] - series.index[0]) / pd.Timedelta(minutes=1)))
    return 1


def per_bucket(bucket_minutes: int) -> str:
    return "per minute" if bucket_minutes == 1 else f"per {bucket_minutes} min"


def _about(value: float) -> str:
    return f"{int(round(value, -1)):,}" if value >= 100 else f"{int(round(value))}"


def _duration(minutes: float) -> str:
    """Readable length of a stall: minutes, then hours, then days."""
    minutes = int(minutes)
    if minutes < 120:
        return f"{minutes} minutes"
    if minutes < 48 * 60:
        return f"{minutes // 60} hours"
    return f"{minutes // (24 * 60)} days"


def volume_headline(series: pd.Series | None, idle_minutes: float = 0.0) -> str:
    """Steady / rose / fell / stalled / nothing yet, as a sentence.

    ``idle_minutes`` is how long ago the newest order arrived, by the wall clock.
    The window is anchored to the newest order, so a stopped feed only shows up
    there. The newest bucket is still filling, so it's left out of the comparison.
    """
    if series is None or series.empty or series.sum() == 0:
        return "No orders yet — start the pipeline with make run"
    minutes = bucket_minutes_of(series)
    trailing_empty = 0
    for value in reversed(series.to_list()):
        if value:
            break
        trailing_empty += minutes
    stalled = max(idle_minutes, trailing_empty)
    if stalled >= STALL_MINUTES:
        return f"No orders in the last {_duration(stalled)} — is the feed running?"

    # Both ends are partial: the newest bucket is still filling, and the feed
    # usually started partway through its first bucket. Compare whole buckets only.
    complete = series.iloc[:-1]
    nonzero = np.flatnonzero(complete.to_numpy())
    history = complete.iloc[nonzero[0] + 1:] if len(nonzero) else complete.iloc[:0]
    if history.empty or history.sum() == 0:
        return "Orders started arriving in the last few minutes"
    recent = history.iloc[-RECENT_BUCKETS:]
    baseline, now = history.mean(), recent.mean()
    change = now / baseline - 1
    if abs(change) <= STEADY_BAND + _EPS:
        return f"Orders are steady at ~{_about(now)} {per_bucket(minutes)}"
    direction = "rose" if change > 0 else "fell"
    span = len(recent) * minutes
    return f"Orders {direction} {abs(change):.0%} in the last {span} minutes"


def flag_rate_headline(series: pd.Series | None, window_rate: float | None = None) -> str:
    """Compares the last RECENT_BUCKETS buckets with the rest of the window.

    ``window_rate`` is the share of every prediction in the window (the KPI). When
    given, a steady headline quotes it, so the title and the KPI beside it agree.
    """
    if series is None or series.dropna().empty:
        return "No predictions yet — waiting for the first model"
    recent = series.iloc[-RECENT_BUCKETS:].dropna()
    earlier = series.iloc[:-RECENT_BUCKETS].dropna()
    steady = series.dropna().mean() if window_rate is None else window_rate
    if recent.empty or earlier.empty:
        return f"The model is flagging a steady {steady:.0f}% of orders late"
    now, before = recent.mean(), earlier.mean()
    if abs(now - before) >= FLAG_CHANGE_PP - _EPS:
        word = "more" if now > before else "fewer"
        return f"The model is flagging {word} orders late ({now:.0f}% vs {before:.0f}%)"
    # Steady: quote the whole window, so the title agrees with the KPI beside it.
    return f"The model is flagging a steady {steady:.0f}% of orders late"


def failures_headline(counts: pd.Series, window_minutes: int = 60) -> str:
    if counts is None or counts.sum() == 0:
        return f"No rows dropped in the last {window_minutes} minutes"
    top = counts.index[0]
    share = counts.iloc[0] / counts.sum()
    return f"{REASON_PHRASES.get(top, top)} cause most dropped rows ({share:.0%})"


def clock_check(newest_timestamp: pd.Timestamp | None, now: pd.Timestamp) -> str | None:
    """Warn when the newest order is more than 5 minutes *ahead* of the wall clock.

    Only a fast simulated clock can do that; a stopped feed is merely behind.
    """
    if newest_timestamp is None or pd.isna(newest_timestamp):
        return None
    return FAST_CLOCK_WARNING if newest_timestamp - now > FAST_CLOCK_LEAD else None


# ---------- the whole page, as data ----------

class Kpi(NamedTuple):
    label: str
    value: str
    delta: str | None
    delta_color: str  # "normal", "inverse" or "off"


@dataclass
class View:
    caption: str
    warning: str | None
    kpis: list
    volume: pd.Series | None
    volume_title: str
    flag_rate: pd.Series | None
    flag_title: str
    failures: pd.Series | None
    failures_title: str
    log_line: str
    numbers: dict  # the raw values behind the KPIs, for tests and the log


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _pp(now: float | None, before: float | None, digits: int = 1) -> str | None:
    if now is None or before is None:
        return None
    change = round(now - before, digits) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{change:+.{digits}f} pp"


def build_view(art: Artifacts, cfg: Config, now: pd.Timestamp,
               cache: Cache | None = None) -> View:
    """Everything the page shows, computed from artifacts. ``now`` is the wall clock (UTC)."""
    window = pd.Timedelta(minutes=cfg.dashboard_window_minutes)
    bucket = pd.Timedelta(minutes=cfg.dashboard_bucket_minutes)
    window_label = f"{cfg.dashboard_window_minutes} min"
    features, predictions, rejects = art.features, art.predictions, art.rejects
    model = art.newest_model.checkpoint_id if art.newest_model else None

    newest = features["timestamp"].max() if not features.empty else None
    numbers: dict = {"orders": None, "drop_rate": None, "flag_rate": None}
    volume = flag = failures = None
    kpis = [Kpi(f"Orders (last {window_label})", "—", None, "normal"),
            Kpi("Rows dropped", "—", None, "inverse"),
            Kpi("Flagged late", "—", None, "off")]
    scored_by = None
    idle = 0.0

    if newest is not None:
        start, end = window_bounds(newest, window, bucket)
        prev_start = start - window
        # Deltas compare like with like: only once the previous window is fully
        # covered by data (its first bucket may be partial, as the feed usually
        # starts mid-minute). With just a few minutes of older data, a steady
        # feed would otherwise show a huge "+23,600 orders" against a near-empty hour.
        has_previous = bool(features["timestamp"].min() < prev_start + bucket)
        idle = max(0.0, (now - newest) / pd.Timedelta(minutes=1))

        orders = sample_volume(features, start, end)
        drops = drop_rate(features, rejects, start, end)
        joined, orphans = join_predictions(predictions, features)
        in_window = joined[_in(joined["timestamp"], start, end)]
        summary = score_summary(in_window)
        scored_by = summary["current_checkpoint"]
        numbers = {"orders": orders, "drop_rate": drops, "flag_rate": summary["flag_rate"],
                   "orphans": orphans, "window": (start, end)}
        if orphans and cache is not None:
            cache.log_once(f"{orphans} predictions have no matching feature row (broken join)")

        previous = {}
        if has_previous:
            prev_joined = joined[_in(joined["timestamp"], prev_start, start)]
            previous = {"orders": sample_volume(features, prev_start, start),
                        "drop_rate": drop_rate(features, rejects, prev_start, start),
                        "flag_rate": score_summary(prev_joined)["flag_rate"]}
        kpis = [
            Kpi(f"Orders (last {window_label})", f"{orders:,}",
                f"{orders - previous['orders']:+,}" if has_previous else None, "normal"),
            Kpi("Rows dropped", _pct(drops), _pp(drops, previous.get("drop_rate")), "inverse"),
            Kpi("Flagged late",
                "—" if summary["flag_rate"] is None else f"{summary['flag_rate']:.0f}%",
                _pp(summary["flag_rate"], previous.get("flag_rate"), 0), "off"),
        ]
        volume = volume_over_time(features, end, window, bucket)
        flag = late_flag_rate_over_time(predictions, features, end, window, bucket)
        if flag.dropna().empty:
            flag = None
        failures = failure_counts(rejects, start, end)
        if failures.empty:
            failures = None

    volume_title = volume_headline(volume, idle)
    flag_title = flag_rate_headline(flag, numbers["flag_rate"])
    failures_title = failures_headline(failures, cfg.dashboard_window_minutes)

    through = "—" if newest is None else f"{newest:%H:%M}Z"
    caption = (f"Reading {display_path(cfg.data_dir)}/ · newest model {model or 'none yet'} · "
               f"scored by {scored_by or '—'} · data through {through} · "
               f"window: last {window_label}")
    counts = art.counts
    log_line = (
        f"read {display_path(cfg.data_dir)}/: {counts['features']} feature, "
        f"{counts['predictions']} prediction, {counts['rejects']} reject files "
        f"(+{art.new_files} new), newest model {model or 'none yet'} → last {window_label}: "
        f"{numbers['orders'] or 0:,} orders, {_pct(numbers['drop_rate'])} dropped, "
        f"{'—' if numbers['flag_rate'] is None else format(numbers['flag_rate'], '.0f') + '%'}"
        f" flagged late"
    )
    return View(caption, clock_check(newest, now), kpis, volume, volume_title, flag, flag_title,
                failures, failures_title, log_line, numbers)
