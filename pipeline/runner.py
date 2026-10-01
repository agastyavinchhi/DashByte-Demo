"""Background launcher behind ``make run`` / ``make stop`` / ``make status``.

It owns process lifecycle only: start each long-running stage as
``python -u -m …``, track its PID in ``.run/<name>.pid``, append its output to
``logs/<name>.log``, and stop it with SIGTERM. It never touches data/ and never
imports a stage's code.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pipeline.config import PROJECT_ROOT, load_config, load_run_dir
from pipeline.paths import display_path

# Everything `make run` starts, in start order; `make stop` goes in reverse, so
# consumers stop before their producers. The last arg is each process's
# identity: `ps` must show it for a PID file to count as ours.
PROCESSES = {
    "simulator": ["-m", "pipeline.simulator"],
    "preprocess": ["-m", "pipeline.preprocess"],
    "train": ["-m", "pipeline.train"],
    "infer": ["-m", "pipeline.infer"],
    "dashboard": ["-m", "pipeline.dashboard_server"],
}

STOP_TIMEOUT_SECONDS = 10.0
STARTUP_CHECK_SECONDS = 0.5


def pid_dir(run_dir: Path) -> Path:
    return run_dir / ".run"


def logs_dir(run_dir: Path) -> Path:
    return run_dir / "logs"


def pid_file(run_dir: Path, name: str) -> Path:
    return pid_dir(run_dir) / f"{name}.pid"


def log_file(run_dir: Path, name: str) -> Path:
    return logs_dir(run_dir) / f"{name}.log"


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _is_ours(pid: int, name: str) -> bool:
    """Guard against PID reuse: the live process must still be this stage.

    If ``ps`` can't answer, trust the PID file rather than refuse to act.
    """
    module = PROCESSES[name][-1]
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if out.returncode != 0:
        return False
    return module in out.stdout


def running_pid(run_dir: Path, name: str) -> int | None:
    """PID of the live process recorded for ``name``, or None if not running.

    A stale PID file (dead process, reused PID or garbage contents) is removed.
    """
    path = pid_file(run_dir, name)
    try:
        pid = int(path.read_text().strip())
    except FileNotFoundError:
        return None
    except ValueError:
        path.unlink(missing_ok=True)
        return None
    if is_alive(pid) and _is_ours(pid, name):
        return pid
    path.unlink(missing_ok=True)
    return None


def start(run_dir: Path) -> int:
    try:
        cfg = load_config()  # fail here, loudly, rather than start a process that dies
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    pid_dir(run_dir).mkdir(parents=True, exist_ok=True)
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")  # logs contain →

    failed = False
    for name, args in PROCESSES.items():
        pid = running_pid(run_dir, name)
        if pid is not None:
            print(f"{name} already running (pid {pid})")
            continue

        log = log_file(run_dir, name)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-u", *args],
                cwd=PROJECT_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # survives the terminal; Ctrl+C there won't hit it
            )
        pid_file(run_dir, name).write_text(f"{proc.pid}\n")

        time.sleep(STARTUP_CHECK_SECONDS)
        code = proc.poll()
        if code is not None and code != 0:
            pid_file(run_dir, name).unlink(missing_ok=True)
            print(f"{name} exited right away (code {code}): {_last_line(log)}\n"
                  f"  see {display_path(log)}", file=sys.stderr)
            failed = True
            continue
        print(f"started {name} (pid {proc.pid}) → {display_path(log)}")
        if name == "dashboard":
            print(f"  open it:  http://localhost:{cfg.dashboard_port}")
    print(f"logs: make logs (or tail -f {display_path(logs_dir(run_dir))}/<name>.log) · "
          f"status: make status · stop: make stop")
    return 1 if failed else 0


def _last_line(log: Path) -> str:
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1].strip() if lines else "(no output)"


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    return not is_alive(pid)


def stop(run_dir: Path) -> int:
    stopped_any = False
    for name in reversed(list(PROCESSES)):
        pid = running_pid(run_dir, name)
        if pid is None:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            if not _wait_for_exit(pid, STOP_TIMEOUT_SECONDS):
                print(f"{name} (pid {pid}) ignored SIGTERM for {STOP_TIMEOUT_SECONDS:g}s; killing it")
                os.kill(pid, signal.SIGKILL)
                _wait_for_exit(pid, 2.0)
        except ProcessLookupError:
            pass  # exited on its own in the meantime
        pid_file(run_dir, name).unlink(missing_ok=True)
        print(f"stopped {name} (pid {pid})")
        stopped_any = True
    if not stopped_any:
        print("nothing running")
    return 0


def status(run_dir: Path) -> int:
    """One line per process, then a count."""
    running = 0
    for name in PROCESSES:
        pid = running_pid(run_dir, name)
        log = display_path(log_file(run_dir, name))
        if pid is None:
            print(f"{name:<11} stopped")
        else:
            running += 1
            print(f"{name:<11} running (pid {pid})  log: {log}")
    print(f"{running}/{len(PROCESSES)} running")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Progress goes to stdout and failures to stderr. When both are piped (CI,
    # `make run | tee`, a container's log) stdout is block-buffered, so a startup
    # failure would print *before* the "started …" lines that preceded it.
    # Flushing every line keeps the output in the order things happened.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="python -m pipeline.runner")
    parser.add_argument("command", choices=["start", "stop", "status", "logs-dir"])
    args = parser.parse_args(argv)
    try:
        run_dir = load_run_dir()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    if args.command == "logs-dir":
        print(logs_dir(run_dir))
        return 0
    return {"start": start, "stop": stop, "status": status}[args.command](run_dir)


if __name__ == "__main__":
    sys.exit(main())
