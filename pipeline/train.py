"""Train: the model's write path. Publishes versioned checkpoints to data/models/.

Run it with ``make train`` (``python -u -m pipeline.train`` underneath).
It watches data/features/ and, whenever at least ``TRAIN_EVERY_N_EVENTS`` new
labeled rows have arrived since the latest published model, trains a
LogisticRegression on distance_km and prep_minutes and publishes:

- ``data/models/model_v<NNNN>.joblib``: the fitted model
- ``data/models/model_v<NNNN>.json``: a small, readable metrics sidecar,
  written last, so a version counts as published only once it exists

Training only publishes. It never imports, calls or waits on inference (or
any other stage); the folders are the whole interface.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from pipeline.config import Config, load_config
from pipeline.paths import display_path, ensure_data_dirs, features_dir, models_dir

# Input contract: what preprocess writes. Defined here, not imported; a
# regression test keeps these in line with preprocess's output.
FEATURES_GLOB = "features_*.csv"
FEATURE_COLUMNS = ("distance_km", "prep_minutes")
LABEL_COLUMN = "was_late"
TIMESTAMP_COLUMN = "timestamp"  # only for the sidecar's data_through
NEEDED_COLUMNS = FEATURE_COLUMNS + (LABEL_COLUMN, TIMESTAMP_COLUMN)

HOLDOUT_FRACTION = 0.2  # newest 20% of rows test the model; no shuffling
MIN_TRAIN_ROWS = 50

MODEL_PATTERN = re.compile(r"model_v(\d{4,})\.(joblib|json)")
MODEL_TYPE = "LogisticRegression"

# Every key in a metrics sidecar, in the order it's written.
SIDECAR_KEYS = (
    "version", "model_file", "model_type", "feature_columns", "label_column",
    "trained_at", "rows_total", "rows_new", "n_train", "n_test", "data_through",
    "late_rate_test", "baseline_accuracy", "accuracy", "precision", "recall", "f1",
    "roc_auc", "coefficients", "intercept", "train_every_n_events",
)


class Published(NamedTuple):
    version: int  # 0 = nothing published yet
    rows_total: int


class NotReady(Exception):
    """Not enough (or not varied enough) data to train yet. Not an error."""


# ---------- reading ----------

def _log(message: str) -> None:
    print(f"[train {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def feature_files(cfg: Config) -> list[Path]:
    """Feature files in name order, which is time order."""
    return sorted((p for p in features_dir(cfg).glob(FEATURES_GLOB) if p.is_file()),
                  key=lambda p: p.name)


class BadFile(Exception):
    """A feature file train can't use. The whole file is skipped, and logged once."""


def _usable(row: tuple[str | None, ...]) -> bool:
    distance, prep, label, _ = row
    try:
        return (math.isfinite(float(distance)) and math.isfinite(float(prep))
                and label in ("0", "1"))
    except (TypeError, ValueError):  # None (short row) or not a number
        return False


def _read_feature_file(path: Path) -> list[tuple[str, ...]]:
    """The needed columns of every row. Raises ``BadFile`` if train can't use the file.

    Preprocess has already cleaned these rows, so this is only a guard: one bad
    file (hand-edited, not UTF-8, ...) must cost that file, not every future model.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not set(NEEDED_COLUMNS) <= set(reader.fieldnames or ()):
                raise BadFile("missing columns")
            rows = [tuple(row[c] for c in NEEDED_COLUMNS) for row in reader]
    except UnicodeDecodeError:
        raise BadFile("not UTF-8 text") from None
    except csv.Error as exc:
        raise BadFile(f"malformed CSV ({exc})") from None
    except OSError as exc:
        raise BadFile(f"unreadable ({exc.strerror or exc})") from None
    for line, row in enumerate(rows, start=2):
        if not _usable(row):
            raise BadFile(f"unusable value on line {line}")
    return rows


def load_rows(cfg: Config, cache: dict | None = None) -> list[tuple[str, ...]]:
    """Every labeled row across all feature files, oldest first.

    Feature files never change once written, so ``cache`` (name -> rows, or
    None for a skipped file) lets a long-running loop read each file only
    once, and log a bad file only once.
    """
    cache = {} if cache is None else cache
    rows: list[tuple[str, ...]] = []
    for path in feature_files(cfg):
        if path.name not in cache:
            try:
                cache[path.name] = _read_feature_file(path)
            except BadFile as reason:
                cache[path.name] = None
                _log(f"skipped {path.name}: {reason}")
        rows.extend(cache[path.name] or ())
    return rows


def rows_available(cfg: Config, cache: dict | None = None) -> int:
    return len(load_rows(cfg, cache))


def _versions(cfg: Config, suffix: str) -> list[int]:
    found = []
    for path in models_dir(cfg).glob(f"model_v*.{suffix}"):
        match = MODEL_PATTERN.fullmatch(path.name)
        if match and match.group(2) == suffix:
            found.append(int(match.group(1)))
    return sorted(found)


def model_paths(cfg: Config, version: int) -> tuple[Path, Path]:
    stem = f"model_v{version:04d}"
    return models_dir(cfg) / f"{stem}.joblib", models_dir(cfg) / f"{stem}.json"


def read_sidecar(path: Path) -> dict | None:
    """A sidecar's contents, or None if it can't be read or has no usable rows_total."""
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        int(sidecar["rows_total"])
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        return None
    return sidecar


def unreadable_sidecars(cfg: Config) -> list[str]:
    return [model_paths(cfg, v)[1].name for v in _versions(cfg, "json")
            if read_sidecar(model_paths(cfg, v)[1]) is None]


def latest_published(cfg: Config) -> Published:
    """The highest version with a readable sidecar.

    A .joblib without its .json isn't published, and neither is one whose .json
    can't be read: a damaged sidecar must not stop every future version.
    """
    for version in reversed(_versions(cfg, "json")):
        sidecar = read_sidecar(model_paths(cfg, version)[1])
        if sidecar is not None:
            return Published(version, int(sidecar["rows_total"]))
    return Published(0, 0)


def next_version(cfg: Config) -> int:
    # Count half-published versions too, so a crash mid-publish never blocks the next one.
    existing = _versions(cfg, "json") + _versions(cfg, "joblib")
    return max(existing, default=0) + 1


# ---------- training ----------

def train_model(rows: list[tuple[str, ...]]) -> tuple[LogisticRegression, dict]:
    """Fit on the oldest 80% of ``rows``, score on the newest 20%.

    Rows are ``(distance_km, prep_minutes, was_late, timestamp)`` strings in time
    order. Raises ``NotReady`` if there's too little data or only one label.
    """
    if len(rows) < MIN_TRAIN_ROWS:
        raise NotReady(f"need at least {MIN_TRAIN_ROWS} rows")

    X = np.array([[float(r[0]), float(r[1])] for r in rows])
    y = np.array([int(r[2]) for r in rows])
    n_test = max(1, round(len(rows) * HOLDOUT_FRACTION))
    n_train = len(rows) - n_test
    X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    if len(set(y_train.tolist())) < 2:
        raise NotReady("need both on-time and late orders")

    # NumPy 2.0 on macOS's Accelerate BLAS raises spurious divide/overflow/invalid
    # warnings on ordinary, finite matmuls (fixed in NumPy 2.1, which needs Python
    # 3.10+). Silence them here, then check the model, so a real numerical
    # problem still fails loudly instead of flooding the log.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
    if not (np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
            and np.isfinite(probabilities).all()):
        raise ValueError("training produced non-finite values")

    late_rate = float(y_test.mean())
    roc_auc = (
        roc_auc_score(y_test, probabilities)
        if len(set(y_test.tolist())) == 2 else None  # undefined with one class
    )
    r3 = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
    metrics = {
        "rows_total": len(rows),
        "n_train": n_train,
        "n_test": n_test,
        "data_through": max(r[3] for r in rows),
        "late_rate_test": r3(late_rate),
        "baseline_accuracy": r3(max(late_rate, 1 - late_rate)),
        "accuracy": r3(accuracy_score(y_test, predicted)),
        "precision": r3(precision_score(y_test, predicted, zero_division=0)),
        "recall": r3(recall_score(y_test, predicted, zero_division=0)),
        "f1": r3(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": r3(roc_auc),
        "coefficients": {c: r3(w) for c, w in zip(FEATURE_COLUMNS, model.coef_[0])},
        "intercept": r3(model.intercept_[0]),
    }
    return model, metrics


# ---------- publishing ----------

def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def format_sidecar(sidecar: dict) -> str:
    """One key per line, with short lists and dicts kept on that line, so `cat` stays readable.

    Still plain JSON: ``json.loads`` reads it back unchanged.
    """
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in sidecar.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def publish(cfg: Config, model: LogisticRegression, metrics: dict) -> int:
    """Write model_v<N>.joblib, then model_v<N>.json (the publish marker). Returns N."""
    models_dir(cfg).mkdir(parents=True, exist_ok=True)
    version = next_version(cfg)
    model_path, sidecar_path = model_paths(cfg, version)
    for path in (model_path, sidecar_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite published model {path}")

    sidecar = {
        "version": version,
        "model_file": model_path.name,
        "model_type": MODEL_TYPE,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **metrics,
        "rows_new": metrics["rows_total"] - latest_published(cfg).rows_total,
        "train_every_n_events": cfg.train_every_n_events,
    }
    sidecar = {key: sidecar[key] for key in SIDECAR_KEYS}

    payload = {"model": model, "feature_columns": list(FEATURE_COLUMNS), "version": version}
    _write_atomic(model_path, lambda tmp: joblib.dump(payload, tmp))
    _write_atomic(sidecar_path, lambda tmp: tmp.write_text(format_sidecar(sidecar)))
    return version


# ---------- loop ----------

def _vtag(version: int) -> str:
    return f"v{version:04d}"


def run(cfg: Config, poll_seconds: float, once: bool = False) -> int:
    """Train and publish whenever enough new rows have arrived. Returns models published.

    With ``once=True``, check a single time and return.
    """
    ensure_data_dirs(cfg)
    threshold = cfg.train_every_n_events
    cache: dict = {}
    published = 0
    last_seen = None  # (rows, latest version) at the last attempt: skip if unchanged
    # An identical failure on polls with nothing new (e.g. an unexpected read
    # error) is logged once, not every few seconds.
    last_error = None
    warned_sidecars: set[str] = set()
    try:
        while True:
            try:
                for name in unreadable_sidecars(cfg):
                    if name not in warned_sidecars:
                        warned_sidecars.add(name)
                        _log(f"ignoring {name}: unreadable sidecar, so that version "
                             f"doesn't count as published")
                latest = latest_published(cfg)
                rows = load_rows(cfg, cache)
                state = (len(rows), latest.version)
                if state != last_seen:
                    last_seen = state
                    last_error = None  # a new attempt reports its own outcome
                    new = len(rows) - latest.rows_total
                    since = f" since {_vtag(latest.version)}" if latest.version else ""
                    if new < threshold:
                        _log(f"waiting: {max(new, 0)}/{threshold} new rows{since}")
                    else:
                        _log(f"{new} new rows{since} (threshold {threshold}) "
                             f"→ training on {len(rows):,} rows")
                        try:
                            model, metrics = train_model(rows)
                        except NotReady as reason:
                            _log(f"waiting: {reason}")
                        else:
                            version = publish(cfg, model, metrics)
                            published += 1
                            model_path, _ = model_paths(cfg, version)
                            _log(
                                f"published model_{_vtag(version)} → {display_path(model_path)} "
                                f"| accuracy {metrics['accuracy']:.2f} vs baseline "
                                f"{metrics['baseline_accuracy']:.2f} | f1 {metrics['f1']:.2f}"
                            )
                last_error = None
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # one bad attempt never kills the loop
                message = f"training failed: {type(exc).__name__}: {exc}"
                if message != last_error:
                    _log(message)
                last_error = message

            if once:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        pass

    latest_version = latest_published(cfg).version
    latest_text = f"latest {_vtag(latest_version)}" if latest_version else "no models yet"
    noun = "model" if published == 1 else "models"
    print(f"[train] stopped after publishing {published} {noun} ({latest_text})", flush=True)
    return published


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

    how = "(one check)" if cfg.train_once else "(Ctrl+C to stop)"
    print(
        f"[train] watching {display_path(features_dir(cfg))} every {cfg.train_poll_seconds:g}s "
        f"→ publishing to {display_path(models_dir(cfg))} every {cfg.train_every_n_events} "
        f"new rows ({MODEL_TYPE} on {', '.join(FEATURE_COLUMNS)}) {how}",
        flush=True,
    )
    run(cfg, poll_seconds=cfg.train_poll_seconds, once=cfg.train_once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
