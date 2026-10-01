"""Shared DashBite settings.

Defaults live on ``Config``; environment variables are the only override.
Run ``python -m pipeline.config`` (via ``make config``) to print the resolved
config and create or confirm the data dirs.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional

# The DashByte-Demo folder (the one holding pipeline/), found from this file's
# location. Not the git root and not the shell's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

ENV_TRAIN_EVERY_N_EVENTS = "TRAIN_EVERY_N_EVENTS"
ENV_BATCH_SIZE = "BATCH_SIZE"
ENV_DATA_DIR = "DASHBITE_DATA_DIR"
ENV_RUN_DIR = "DASHBITE_RUN_DIR"
ENV_SIM_INTERVAL_SECONDS = "SIM_INTERVAL_SECONDS"
ENV_SIM_MAX_BATCHES = "SIM_MAX_BATCHES"
ENV_SIM_SEED = "SIM_SEED"
ENV_SIM_MESSY_RATE = "SIM_MESSY_RATE"
ENV_SIM_LATE_THRESHOLD_MINUTES = "SIM_LATE_THRESHOLD_MINUTES"
ENV_SIM_CLOCK_SPEED = "SIM_CLOCK_SPEED"
ENV_SIM_PEAK_DELAY_MINUTES = "SIM_PEAK_DELAY_MINUTES"
ENV_PREPROCESS_POLL_SECONDS = "PREPROCESS_POLL_SECONDS"
ENV_PREPROCESS_ONCE = "PREPROCESS_ONCE"
ENV_TRAIN_POLL_SECONDS = "TRAIN_POLL_SECONDS"
ENV_TRAIN_ONCE = "TRAIN_ONCE"
ENV_INFER_POLL_SECONDS = "INFER_POLL_SECONDS"
ENV_INFER_ONCE = "INFER_ONCE"
ENV_DASHBOARD_WINDOW_MINUTES = "DASHBOARD_WINDOW_MINUTES"
ENV_DASHBOARD_BUCKET_MINUTES = "DASHBOARD_BUCKET_MINUTES"
ENV_DASHBOARD_REFRESH_SECONDS = "DASHBOARD_REFRESH_SECONDS"
ENV_DASHBOARD_PORT = "DASHBOARD_PORT"
ENV_POLL_INTERVAL_SECONDS = "POLL_INTERVAL_SECONDS"

MAX_DASHBOARD_BUCKETS = 288

# Every variable load_config() reads. Tests strip these so a shell env can't leak in.
ENV_VARS = (
    ENV_TRAIN_EVERY_N_EVENTS,
    ENV_BATCH_SIZE,
    ENV_DATA_DIR,
    ENV_RUN_DIR,
    ENV_SIM_INTERVAL_SECONDS,
    ENV_SIM_MAX_BATCHES,
    ENV_SIM_SEED,
    ENV_SIM_MESSY_RATE,
    ENV_SIM_LATE_THRESHOLD_MINUTES,
    ENV_SIM_CLOCK_SPEED,
    ENV_SIM_PEAK_DELAY_MINUTES,
    ENV_PREPROCESS_POLL_SECONDS,
    ENV_PREPROCESS_ONCE,
    ENV_TRAIN_POLL_SECONDS,
    ENV_TRAIN_ONCE,
    ENV_INFER_POLL_SECONDS,
    ENV_INFER_ONCE,
    ENV_DASHBOARD_WINDOW_MINUTES,
    ENV_DASHBOARD_BUCKET_MINUTES,
    ENV_DASHBOARD_REFRESH_SECONDS,
    ENV_DASHBOARD_PORT,
    ENV_POLL_INTERVAL_SECONDS,
)


@dataclass(frozen=True)
class Config:
    # Retrain after this many new labeled rows. ~1 minute of simulator output.
    train_every_n_events: int = 400
    batch_size: int = 20
    data_dir: Path = field(default=DEFAULT_DATA_DIR)
    # Where make run / make stop keep .run/ (PID files) and logs/.
    run_dir: Path = field(default=PROJECT_ROOT)
    sim_interval_seconds: float = 3.0
    sim_max_batches: Optional[int] = None  # None = run until stopped
    sim_seed: Optional[int] = None  # None = fresh random seed each run
    sim_messy_rate: float = 0.05
    # An order is late when prep + 3 min/km + noise exceeds this many minutes.
    sim_late_threshold_minutes: float = 30.0
    # Simulated seconds per real second. 300 = one simulated day in ~4.8 real minutes; 1 = real time.
    sim_clock_speed: float = 300.0
    # Extra minutes added to the delivery estimate for orders placed at rush hour. 0 = off.
    sim_peak_delay_minutes: float = 8.0
    # The classroom poll cadence. It's the default for every *_POLL_SECONDS and for
    # the dashboard refresh; a stage-specific variable still overrides it.
    poll_interval_seconds: float = 15.0
    preprocess_poll_seconds: float = 15.0
    preprocess_once: bool = False  # True = one pass over data/raw, then exit
    train_poll_seconds: float = 15.0
    train_once: bool = False  # True = check once, publish if due, then exit
    infer_poll_seconds: float = 15.0
    infer_once: bool = False  # True = one pass: load newest checkpoint, score pending, exit
    # Model Pulse: window and bucket are minutes of the order clock.
    dashboard_window_minutes: int = 60
    dashboard_bucket_minutes: int = 1
    dashboard_refresh_seconds: float = 15.0
    dashboard_port: int = 8501


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _optional_positive_int(env: Mapping[str, str], name: str) -> int | None:
    if env.get(name) is None:
        return None
    return _positive_int(env, name, 0)


def _optional_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _float(env, name, default)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {env.get(name)!r}")
    return value


def _non_negative_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _float(env, name, default)
    if value < 0:
        raise ValueError(f"{name} must be zero or more, got {env.get(name)!r}")
    return value


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    if raw.strip() not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw.strip() == "1"


def _port(env: Mapping[str, str], name: str, default: int) -> int:
    value = _positive_int(env, name, default)
    if not 1024 <= value <= 65535:
        raise ValueError(f"{name} must be a port from 1024 to 65535, got {env.get(name)!r}")
    return value


def _bucket(env: Mapping[str, str], window: int, default: int) -> int:
    name = ENV_DASHBOARD_BUCKET_MINUTES
    bucket = _positive_int(env, name, default)
    if window % bucket:
        raise ValueError(f"{name} must divide the {window}-minute window evenly, got {bucket}")
    if window // bucket > MAX_DASHBOARD_BUCKETS:
        raise ValueError(f"{name}={bucket} gives {window // bucket} buckets over the "
                         f"{window}-minute window; the most is {MAX_DASHBOARD_BUCKETS}")
    return bucket


def _fraction(env: Mapping[str, str], name: str, default: float) -> float:
    value = _float(env, name, default)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {env.get(name)!r}")
    return value


def _dir(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw = env.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        raise ValueError(f"{name} must be a directory path, got an empty value")
    path = Path(raw).expanduser()
    # Relative paths resolve from PROJECT_ROOT, wherever the project folder lives.
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_run_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve only DASHBITE_RUN_DIR, so `make stop` works even if other vars are bad."""
    return _dir(os.environ if env is None else env, ENV_RUN_DIR, Config().run_dir)


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from ``env`` (defaults to ``os.environ``).

    Unset variables keep their defaults. Bad values raise ``ValueError``
    naming the variable; there is no silent fallback.
    """
    if env is None:
        env = os.environ
    defaults = Config()
    poll = _positive_float(env, ENV_POLL_INTERVAL_SECONDS, defaults.poll_interval_seconds)
    return Config(
        train_every_n_events=_positive_int(
            env, ENV_TRAIN_EVERY_N_EVENTS, defaults.train_every_n_events
        ),
        batch_size=_positive_int(env, ENV_BATCH_SIZE, defaults.batch_size),
        data_dir=_dir(env, ENV_DATA_DIR, defaults.data_dir),
        run_dir=_dir(env, ENV_RUN_DIR, defaults.run_dir),
        sim_interval_seconds=_positive_float(
            env, ENV_SIM_INTERVAL_SECONDS, defaults.sim_interval_seconds
        ),
        sim_max_batches=_optional_positive_int(env, ENV_SIM_MAX_BATCHES),
        sim_seed=_optional_int(env, ENV_SIM_SEED),
        sim_messy_rate=_fraction(env, ENV_SIM_MESSY_RATE, defaults.sim_messy_rate),
        sim_late_threshold_minutes=_positive_float(
            env, ENV_SIM_LATE_THRESHOLD_MINUTES, defaults.sim_late_threshold_minutes
        ),
        sim_clock_speed=_positive_float(env, ENV_SIM_CLOCK_SPEED, defaults.sim_clock_speed),
        sim_peak_delay_minutes=_non_negative_float(
            env, ENV_SIM_PEAK_DELAY_MINUTES, defaults.sim_peak_delay_minutes
        ),
        poll_interval_seconds=poll,
        preprocess_poll_seconds=_positive_float(env, ENV_PREPROCESS_POLL_SECONDS, poll),
        preprocess_once=_flag(env, ENV_PREPROCESS_ONCE, defaults.preprocess_once),
        train_poll_seconds=_positive_float(env, ENV_TRAIN_POLL_SECONDS, poll),
        train_once=_flag(env, ENV_TRAIN_ONCE, defaults.train_once),
        infer_poll_seconds=_positive_float(env, ENV_INFER_POLL_SECONDS, poll),
        infer_once=_flag(env, ENV_INFER_ONCE, defaults.infer_once),
        **_dashboard_settings(env, defaults, poll),
    )


def _dashboard_settings(env: Mapping[str, str], defaults: Config, poll: float) -> dict:
    window = _positive_int(env, ENV_DASHBOARD_WINDOW_MINUTES, defaults.dashboard_window_minutes)
    return {
        "dashboard_window_minutes": window,
        "dashboard_bucket_minutes": _bucket(env, window, defaults.dashboard_bucket_minutes),
        "dashboard_refresh_seconds": _positive_float(env, ENV_DASHBOARD_REFRESH_SECONDS, poll),
        "dashboard_port": _port(env, ENV_DASHBOARD_PORT, defaults.dashboard_port),
    }


def main(argv: list[str] | None = None) -> int:
    from pipeline.paths import ensure_data_dirs

    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    # `python -m pipeline.config get <field>` prints one validated value and
    # touches nothing; the Makefile uses it to resolve DASHBOARD_PORT.
    if argv[:1] == ["get"]:
        names = {f.name for f in fields(cfg)}
        if len(argv) != 2 or argv[1] not in names:
            print(f"usage: python -m pipeline.config get <{'|'.join(sorted(names))}>",
                  file=sys.stderr)
            return 2
        print(getattr(cfg, argv[1]))
        return 0

    for f in fields(cfg):
        print(f"{f.name} = {getattr(cfg, f.name)}")
    print()
    for name, (path, created) in ensure_data_dirs(cfg).items():
        status = "created" if created else "exists"
        print(f"{status:<8} {name:<12} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
