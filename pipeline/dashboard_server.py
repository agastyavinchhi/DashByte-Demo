"""Launch the Model Pulse dashboard: ``python -m pipeline.dashboard_server``.

One entry point for both ``make dashboard`` (foreground) and ``make run``
(background, via the runner). Streamlit runs inside this process, so the
command line the runner tracks stays ``-m pipeline.dashboard_server`` and
``make stop``'s SIGTERM reaches Streamlit's own clean shutdown.

Nothing leaves the classroom machine: headless (no first-run email prompt, no
browser launch), usage stats off, and bound to localhost, which also stops
Streamlit looking up this machine's public IP. No file watcher: the page needs
no hot reload.
"""
from __future__ import annotations

import os
import socket
import sys
import warnings

from pipeline.config import PROJECT_ROOT, load_config

APP = PROJECT_ROOT / "pipeline" / "dashboard.py"
HOST = os.environ.get("DASHBOARD_HOST", "localhost")


def port_in_use(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def streamlit_argv(port: int) -> list[str]:
    return [
        "streamlit", "run", str(APP),
        "--server.headless", "true",
        "--server.port", str(port),
        "--server.address", HOST,
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]


def main() -> int:
    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    # Checked before importing Streamlit, so a busy port fails in well under a
    # second and `make run`'s startup check catches it.
    if port_in_use(cfg.dashboard_port):
        print(f"dashboard: port {cfg.dashboard_port} is already in use. Stop whatever is "
              f"using it, or pick another with DASHBOARD_PORT=<port>", file=sys.stderr)
        return 1

    print(f"[dashboard] Model Pulse on http://{HOST}:{cfg.dashboard_port} "
          f"(reads data/ every {cfg.dashboard_refresh_seconds:g}s)", flush=True)

    # A harmless LibreSSL notice from macOS's system Python.
    warnings.filterwarnings("ignore", module="urllib3")
    from streamlit.web import cli as streamlit_cli

    sys.argv = streamlit_argv(cfg.dashboard_port)
    try:
        streamlit_cli.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
