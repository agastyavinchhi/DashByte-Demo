# DashBite demo pipeline

Food-delivery orders come in, and a model predicts whether each will be late.
Every stage is its own process, and stages talk **only through folders under
`data/`**. The full design, stage by stage, is in [docs/plan.md](docs/plan.md).

```
simulator → data/raw → preprocess → data/features → train → data/models
                                         │                      │
                                         └──────── infer ◀──────┘ → data/predictions
                                    (Model Pulse dashboard only reads data/)
```

## Setup

Python 3.9+. Everything runs through `make`, using `.venv/bin/python`, so
there's no virtualenv to activate.

```bash
make install
make test
```

## Run each stage separately

Use one terminal per stage. Stop each one with **Ctrl+C**. A stage can start and
stop on its own: the others keep going, and it catches up from the files on disk
when it comes back.

| Terminal | Command | Reads | Writes |
|---|---|---|---|
| 1 | `make simulator` | — | `data/raw/orders_*.csv` |
| 2 | `make preprocess` | `data/raw/` | `data/features/`, rejects in `data/quality/` |
| 3 | `make train` | `data/features/` | `data/models/model_v*.joblib` + `.json` |
| 4 | `make infer` | `data/models/`, `data/features/` | `data/predictions/` |
| 5 | `make dashboard` | `data/features/`, `data/predictions/`, `data/quality/`, model `.json` sidecars (read-only) | nothing; open http://localhost:8501 |

- **Order doesn't matter.** Infer waits cleanly until the first model exists,
  and train waits until enough rows have arrived.
- **Training doesn't need to be running.** Once a model is on disk, stop
  `make train`; `make infer` keeps scoring with the newest checkpoint.
- **One pass instead of a loop:** `PREPROCESS_ONCE=1 make preprocess`,
  `TRAIN_ONCE=1 make train`, or `INFER_ONCE=1 make infer`.

## Run everything in the background

```bash
make run      # starts simulator, preprocess, train, infer and the dashboard
make status   # which of the five are running
make logs     # follow every log at once (Ctrl+C stops following, not the stack)
make stop     # stops them all, dashboard first, simulator last; safe to run twice
```

- Each process runs in its own session, so it keeps going after you close the
  terminal that ran `make run`.
- Logs go to `logs/<stage>.log` and PID files to `.run/<stage>.pid`.
- Model Pulse is at http://localhost:8501 (`DASHBOARD_PORT` changes it).
- Running `make run` again only starts what isn't already running.
- If the dashboard's port is busy, `make run` says so, and the other four keep
  running. Free the port, then run `make run` again.

Foreground (`make simulator`, `make dashboard`, …) and background (`make run`)
use the same code. Pick one style at a time, so two copies of a stage don't run
against the same `data/`.

## Settings

Every setting is an environment variable that you put in front of the command.
`make config` prints the resolved values.

```bash
SIM_CLOCK_SPEED=300 make simulator         # fast clock: a simulated day in ~5 min (rush-hour demo)
SIM_INTERVAL_SECONDS=1 make simulator      # a batch every second
TRAIN_EVERY_N_EVENTS=200 make train        # retrain more often
DASHBOARD_PORT=8502 make dashboard
POLL_INTERVAL_SECONDS=5 make run           # check for new files every 5s instead of 15s
```

`POLL_INTERVAL_SECONDS` (default **15**) is how often preprocess, train and
infer check for new files, and how often the dashboard refreshes.
`PREPROCESS_POLL_SECONDS`, `TRAIN_POLL_SECONDS`, `INFER_POLL_SECONDS` and
`DASHBOARD_REFRESH_SECONDS` override it for one stage. The simulator still
writes a batch every 3s (`SIM_INTERVAL_SECONDS`), so each poll picks up about
5 batches.

The simulator's clock defaults to **real time (1×)**, which is what Model Pulse's
per-minute charts expect. For the Stage 3 rush-hour demo, where `hour` and
`is_peak` sweep through a whole day, use `SIM_CLOCK_SPEED=300`. The dashboard
shows a warning if it detects that fast clock. To go back, run `make stop`,
then `make clean-data`, then `make run`.

## Reset

```bash
make stop
make clean-data   # removes data/ only; leaves code, docs/, logs/ and .run/ alone
```
