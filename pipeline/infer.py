"""Infer: the model's read path. Scores new orders with the newest checkpoint on disk.

Run it with ``make infer`` (``python -u -m pipeline.infer`` underneath).
Every poll it:

1. picks the newest *published* checkpoint in data/models/ (the highest
   ``model_v<NNNN>`` whose ``.json`` sidecar is readable) and loads it if it's
   newer than the one in use;
2. scores every ``data/features/features_<stamp>_<seq>.csv`` that has no
   ``data/predictions/predictions_<stamp>_<seq>.csv`` yet.

It is a consumer of artifacts only. It never imports, calls, starts or signals
training, and it never writes to data/models/. With no checkpoint yet it waits;
with training stopped it keeps using the last checkpoint on disk.
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
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np

from pipeline.config import Config, load_config
from pipeline.paths import (
    display_path,
    ensure_data_dirs,
    features_dir,
    models_dir,
    predictions_dir,
)

# Input contracts. Defined here, not imported; regression tests keep them equal
# to what preprocess and train produce.
FEATURES_GLOB = "features_*.csv"
MODEL_SIDECAR_PATTERN = re.compile(r"model_v(\d{4,})\.json")

# Output contract: what the dashboard reads.
PREDICTION_COLUMNS = ("order_id", "late_probability", "predicted_late", "checkpoint_id")
PREDICT_THRESHOLD = 0.5


class Checkpoint(NamedTuple):
    """A published version: its sidecar exists and is readable."""
    version: int
    sidecar_path: Path
    sidecar: dict

    @property
    def checkpoint_id(self) -> str:
        return f"model_v{self.version:04d}"


class Loaded(NamedTuple):
    """A checkpoint that loaded and is ready to score."""
    checkpoint: Checkpoint
    model: Any
    feature_columns: tuple[str, ...]

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint.checkpoint_id


class BatchResult(NamedTuple):
    predictions_path: Path
    orders: int
    predicted_late: int


class CannotLoad(Exception):
    """A published checkpoint that can't be used (missing/corrupt file, bad columns)."""


class SkippedFile(Exception):
    """A features file that can't be scored. Logged once, never fatal."""


# ---------- finding checkpoints ----------

def _log(message: str) -> None:
    print(f"[infer {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_sidecar(path: Path) -> dict | None:
    """Sidecar contents, or None if unreadable. Same rule as train's latest_published."""
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        int(sidecar["rows_total"])
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        return None
    return sidecar


def published_checkpoints(cfg: Config) -> list[Checkpoint]:
    """Every readable published version, newest (highest number) first.

    By version number, never modification time, so touching an old file can't
    make it "newest". A .joblib without a .json is half-published and ignored.
    """
    found = []
    for path in models_dir(cfg).glob("model_v*.json"):
        match = MODEL_SIDECAR_PATTERN.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    checkpoints = []
    for version, path in sorted(found, reverse=True):
        sidecar = read_sidecar(path)
        if sidecar is not None:
            checkpoints.append(Checkpoint(version, path, sidecar))
    return checkpoints


def unreadable_sidecars(cfg: Config) -> list[int]:
    """Versions whose model_v<NNNN>.json exists but can't be read."""
    return sorted(
        int(m.group(1)) for p in models_dir(cfg).glob("model_v*.json")
        if (m := MODEL_SIDECAR_PATTERN.fullmatch(p.name)) and read_sidecar(p) is None
    )


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    """Published versions never change, so any change here means a different artifact."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def newest_checkpoint(cfg: Config) -> Checkpoint | None:
    checkpoints = published_checkpoints(cfg)
    return checkpoints[0] if checkpoints else None


def load_checkpoint(ckpt: Checkpoint) -> Loaded:
    """Load the model a sidecar names. Raises ``CannotLoad`` with a short reason.

    joblib.load runs code from the file, so it only ever loads
    ``model_v<NNNN>.joblib`` sitting next to its own sidecar in data/models/.
    """
    expected_file = f"{ckpt.checkpoint_id}.joblib"
    if ckpt.sidecar.get("model_file", expected_file) != expected_file:
        raise CannotLoad(f"sidecar names {ckpt.sidecar.get('model_file')!r}, "
                         f"expected {expected_file}")
    columns = ckpt.sidecar.get("feature_columns")
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(c, str) and c for c in columns)):
        raise CannotLoad("sidecar has no usable feature_columns")

    model_path = ckpt.sidecar_path.with_name(expected_file)
    if not model_path.is_file():
        raise CannotLoad(f"{expected_file} is missing")
    try:
        payload = joblib.load(model_path)
    except Exception as exc:  # corrupt, truncated, wrong format, incompatible pickle
        raise CannotLoad(f"{expected_file} is unreadable ({type(exc).__name__})") from None

    if not isinstance(payload, dict) or "model" not in payload:
        raise CannotLoad(f"{expected_file} doesn't hold a model")
    if list(payload.get("feature_columns") or []) != columns:
        raise CannotLoad(f"feature_columns differ between {expected_file} and its sidecar")
    if not hasattr(payload["model"], "predict_proba"):
        raise CannotLoad(f"{expected_file} holds no predict_proba model")
    # Try one prediction now, so a model that can't score with these columns
    # (e.g. it expects a different number of inputs) is rejected here and the
    # current model stays in use, instead of failing on every file later.
    try:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            probe = payload["model"].predict_proba(np.zeros((1, len(columns))))
        usable = np.shape(probe) == (1, 2) and np.isfinite(probe).all()
    except Exception as exc:
        raise CannotLoad(f"{expected_file} can't score {len(columns)} feature columns "
                         f"({type(exc).__name__})") from None
    if not usable:
        raise CannotLoad(f"{expected_file} gives no usable late probability")
    return Loaded(ckpt, payload["model"], tuple(columns))


def _describe(loaded: Loaded) -> str:
    sidecar = loaded.checkpoint.sidecar
    accuracy = sidecar.get("accuracy")
    accuracy = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "n/a"
    return f"accuracy {accuracy}, trained on {int(sidecar['rows_total']):,} rows"


# ---------- scoring ----------

def batch_key(features_path: Path) -> str:
    """``features_<stamp>_<seq>.csv`` -> ``<stamp>_<seq>``."""
    return features_path.name[len("features_"):-len(".csv")]


def predictions_path_for(features_path: Path, cfg: Config) -> Path:
    return predictions_dir(cfg) / f"predictions_{batch_key(features_path)}.csv"


def pending_files(cfg: Config) -> list[Path]:
    """Feature files with no predictions file yet, oldest first. The folders are the state."""
    return sorted(
        (p for p in features_dir(cfg).glob(FEATURES_GLOB)
         if p.is_file() and not predictions_path_for(p, cfg).exists()),
        key=lambda p: p.name,
    )


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def score_file(path: Path, loaded: Loaded, cfg: Config) -> BatchResult:
    """Score one feature file with one checkpoint and write its predictions file."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            missing = [c for c in ("order_id", *loaded.feature_columns) if c not in header]
            if missing:
                raise SkippedFile(f"missing column {', '.join(missing)}")
            records = list(reader)
    except FileNotFoundError:
        raise SkippedFile("file disappeared") from None
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SkippedFile(f"unreadable ({type(exc).__name__})") from None

    matrix = []
    for line_no, record in enumerate(records, start=2):
        row = []
        for column in loaded.feature_columns:
            try:
                value = float(record[column])
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                raise SkippedFile(f"non-numeric {column} on line {line_no}")
            row.append(value)
        matrix.append(row)

    rows: list[dict[str, str]] = []
    if matrix:
        # Same spurious NumPy 2.0 / macOS Accelerate warnings as in training;
        # silence them, then check the numbers so a real problem still stands out.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            probabilities = loaded.model.predict_proba(np.array(matrix))[:, 1]
        if not np.isfinite(probabilities).all():
            raise SkippedFile(f"{loaded.checkpoint_id} produced non-finite probabilities")
        for record, probability in zip(records, probabilities):
            rounded = round(float(probability), 3)
            rows.append({
                "order_id": record["order_id"],
                "late_probability": f"{rounded:.3f}",
                # Decide on the written value, so the file always agrees with itself.
                "predicted_late": "1" if rounded >= PREDICT_THRESHOLD else "0",
                "checkpoint_id": loaded.checkpoint_id,
            })

    out = predictions_path_for(path, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(out, rows)
    return BatchResult(out, len(rows), sum(r["predicted_late"] == "1" for r in rows))


# ---------- loop ----------

class _State:
    """What the loop remembers between polls. Everything else comes from disk."""

    def __init__(self) -> None:
        self.loaded: Loaded | None = None
        self.loaded_identity: tuple[int, int, int] | None = None  # its sidecar, at load time
        self.failed_versions: set[int] = set()  # each broken version is logged once
        self.skipped_files: set[str] = set()  # each unusable features file logged once
        self.waiting_logged = False


def _refresh_model(cfg: Config, state: _State) -> None:
    """Switch to the newest checkpoint that loads, if it's newer than the one in use."""
    if state.loaded is not None and (
        _file_identity(state.loaded.checkpoint.sidecar_path) != state.loaded_identity
    ):
        # The artifact we're scoring with is gone or was replaced, e.g. data/
        # was reset while infer ran and train started again from v0001. Keep
        # using it and every new row would be labelled with a checkpoint_id that
        # now names a different model, so start over from what's on disk.
        _log(f"{state.loaded.checkpoint_id} is no longer the file on disk "
             f"(was data/models reset?); looking for the newest checkpoint again")
        state.loaded = state.loaded_identity = None
        state.failed_versions.clear()  # version numbers now name different files
        state.waiting_logged = False

    for version in unreadable_sidecars(cfg):
        if version not in state.failed_versions:
            state.failed_versions.add(version)
            _log(f"cannot load model_v{version:04d}: unreadable sidecar")

    current = state.loaded.checkpoint.version if state.loaded else 0
    candidates = [c for c in published_checkpoints(cfg)
                  if c.version > current and c.version not in state.failed_versions]
    for ckpt in candidates:  # newest first; fall back down if one won't load
        try:
            loaded = load_checkpoint(ckpt)
        except CannotLoad as reason:
            state.failed_versions.add(ckpt.version)
            _log(f"cannot load {ckpt.checkpoint_id}: {reason}")
            continue
        if state.loaded is None:
            _log(f"found {loaded.checkpoint_id}")
            _log(f"loaded {loaded.checkpoint_id} ({_describe(loaded)})")
        else:
            _log(f"loaded {loaded.checkpoint_id} ({_describe(loaded)}) "
                 f"— replacing {state.loaded.checkpoint_id}")
        state.loaded = loaded
        state.loaded_identity = _file_identity(ckpt.sidecar_path)
        return

    if state.loaded is None and not state.waiting_logged:
        state.waiting_logged = True
        if state.failed_versions:
            _log("waiting for a checkpoint: none of the published models could be loaded")
        else:
            _log(f"waiting for a checkpoint: no model_v*.json in "
                 f"{display_path(models_dir(cfg))} yet (run make train)")


def run(cfg: Config, poll_seconds: float, once: bool = False) -> tuple[int, int, str | None]:
    """Score pending feature files with the newest checkpoint, then poll until Ctrl+C.

    Returns ``(files, orders, checkpoint_id)``; checkpoint_id is None if none loaded.
    """
    ensure_data_dirs(cfg)
    state = _State()
    files = orders = 0
    last_error = None
    try:
        while True:
            try:
                _refresh_model(cfg, state)
                if state.loaded is not None:
                    for path in pending_files(cfg):
                        if path.name in state.skipped_files:
                            continue
                        try:
                            result = score_file(path, state.loaded, cfg)
                        except SkippedFile as reason:
                            state.skipped_files.add(path.name)
                            _log(f"skipped {path.name}: {reason}")
                            continue
                        files += 1
                        orders += result.orders
                        _log(
                            f"{path.name} → {result.predictions_path.name}: {result.orders} "
                            f"orders scored with {state.loaded.checkpoint_id} "
                            f"({result.predicted_late} predicted late) | total {orders}"
                        )
                last_error = None
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # one bad poll never kills the loop
                message = f"inference failed: {type(exc).__name__}: {exc}"
                if message != last_error:  # same failure every poll is logged once
                    _log(message)
                    last_error = message

            if once:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        pass

    checkpoint_id = state.loaded.checkpoint_id if state.loaded else None
    last = f"last checkpoint {checkpoint_id}" if checkpoint_id else "no checkpoint loaded"
    print(f"[infer] stopped after scoring {files} batches ({orders} orders), {last}", flush=True)
    return files, orders, checkpoint_id


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

    how = "(one pass)" if cfg.infer_once else "(Ctrl+C to stop)"
    print(
        f"[infer] watching {display_path(models_dir(cfg))} for the newest checkpoint and "
        f"{display_path(features_dir(cfg))} every {cfg.infer_poll_seconds:g}s → writing "
        f"{display_path(predictions_dir(cfg))} {how}",
        flush=True,
    )
    run(cfg, poll_seconds=cfg.infer_poll_seconds, once=cfg.infer_once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
