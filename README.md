# DashBite: a containerized ML pipeline

**Student:** Agastya Vinchhi · **NetID:** av351 · **Option:** 1 (Extend and Containerize DashBite)
**Plan:** [docs/plan.md](docs/plan.md) · **AI transcripts:** [docs/transcripts/](docs/transcripts/)

---

## 1. Which option I selected

**Option 1: Extend and Containerize DashBite.** I built the DashBite pipeline stage by stage with three AI roles: an Architect, a Builder and a Tester. Then I packaged the whole stack into one Docker image.

The extension is the Docker setup:

- [`Dockerfile`](Dockerfile): a `python:3.9-slim` image that runs all five processes. Its dependencies install in their own layer, so editing code doesn't reinstall them.
- [`.dockerignore`](.dockerignore): keeps the host's `.venv/`, `data/`, `logs/`, `.run/` and `.git/` out of the image. The build context is about 800 KB.
- [`docker-compose.yml`](docker-compose.yml): one command builds and starts the stack.

On top of those, I made these container-readiness improvements:

- **Persistent data storage.** Compose mounts a named volume, `dashbite-data`, at `/app/data`. Models and predictions survive `docker compose down` and `up`, and the pipeline resumes from the files on disk.
- **A health check.** Compose polls Streamlit's `/_stcore/health` endpoint every 30s, so `docker compose ps` shows `(healthy)` or `(unhealthy)`.
- **Graceful shutdown.** When the container gets a stop signal, it runs `make stop`, so each stage stops in order and logs its summary. The container then exits with code 0 instead of the processes being killed.
- **Runtime configuration.** The dashboard used to be hard-coded to `localhost`. Inside a container that address isn't reachable from your browser. It now reads `DASHBOARD_HOST`, which the image sets to `0.0.0.0`. On the host it still defaults to `localhost`, so nothing is exposed there ([pipeline/dashboard_server.py:23](pipeline/dashboard_server.py)).
- **Containerized tests.** The full test suite also runs inside the image (see 3.4).

## 2. Purpose of the project

Food-delivery orders come in, and a model predicts whether each order will be **late**. The project demonstrates a modular ML pipeline. Each stage is its own process, and stages communicate **only through folders under `data/`**, never by importing each other.

```
simulator → data/raw → preprocess → data/features → train → data/models
                                         │                      │
                                         └──────── infer ◀──────┘ → data/predictions
                                    (Model Pulse dashboard only reads data/)
```

| Stage | What it does |
|---|---|
| `simulator` | Writes a batch of 20 synthetic orders every 3s, including some deliberately messy rows. |
| `preprocess` | Rejects invalid rows (with a reason code), derives `hour` and `is_peak`, and writes model-ready features. |
| `train` | Publishes a new versioned LogisticRegression checkpoint every 400 new rows. |
| `infer` | Scores new features with the newest checkpoint on disk. It keeps working if training is stopped. |
| `dashboard` | **Model Pulse**, a read-only Streamlit page showing order volume, drop rate and late-flag rate over time. |

---

## 3. How to install, run and test it

There are three ways to run DashBite: **with Make** on your machine, **with Docker Compose**, or **with plain Docker**. Use one at a time, because both serve the dashboard on port 8501.

### 3.1 Run with Make (on your machine)

**Requirements:** Python 3.9+ and `make`. Make always uses `.venv/bin/python`, so you never need to activate the virtualenv.

```bash
make install      # create .venv/ and install requirements.txt
make run          # start simulator, preprocess, train, infer and the dashboard in the background
make status       # check that all five are running
make logs         # follow every stage's log (Ctrl+C stops following, not the stack)
```

Open **http://localhost:8501** to see Model Pulse. The first model is published after about 1 minute, and predictions start right after that.

```bash
make stop         # stop all five processes
make clean-data   # delete data/ to start fresh (code and docs are never touched)
```

To run each stage in its own terminal instead (Ctrl+C stops each one):

| Terminal | Command | Reads | Writes |
|---|---|---|---|
| 1 | `make simulator` | — | `data/raw/` |
| 2 | `make preprocess` | `data/raw/` | `data/features/`, plus rejects in `data/quality/` |
| 3 | `make train` | `data/features/` | `data/models/` |
| 4 | `make infer` | `data/models/`, `data/features/` | `data/predictions/` |
| 5 | `make dashboard` | `data/` (read-only) | nothing; open http://localhost:8501 |

### 3.2 Run with Docker Compose (recommended for Docker)

**Requirements:** Docker Desktop (or Docker Engine) running, and port 8501 free. If the Make stack is running, run `make stop` first.

**1. Build and start the stack in the background.** The first build takes about 40 seconds.
```bash
docker compose up -d --build
```

**2. Check that it's running.** After about 30 seconds the status shows `(healthy)`.
```bash
docker compose ps
```

**3. Open http://localhost:8501** for Model Pulse. The first model is published after about 1 minute.

**4. Watch the logs, and look at the data.** Ctrl+C stops following the logs, not the stack.
```bash
docker compose logs -f
```
```bash
docker exec dashbite ls data/models data/predictions
```

**5. Stop it.** The data stays in the `dashbite-data` volume, so the next `up` resumes where it left off.
```bash
docker compose down
```

**To reset all data**, remove the volume too:
```bash
docker compose down -v
```

### 3.3 Run with plain Docker (without Compose)

This runs the same image without the volume, so each run starts with empty data.

**1. Build the image.**
```bash
docker build -t dashbite .
```

**2. Run the whole stack in one container.** Every stage's log streams to this terminal.
```bash
docker run --rm --init -p 8501:8501 --name dashbite dashbite
```

**3. Open http://localhost:8501** for Model Pulse.

**4. Look at the data the container produces.** Run these in a second terminal:
```bash
docker exec dashbite ls data/raw data/features data/models data/predictions
```
```bash
docker exec dashbite make status
```

**5. Stop it.** Press **Ctrl+C** in the first terminal, or run:
```bash
docker stop dashbite
```

`--rm` deletes the container when it stops. `--init` passes Ctrl+C on to the container, which then stops every stage cleanly.

### 3.4 Run the tests

**On your machine:**
```bash
make test                 # full suite: 544 tests
make test-unit            # or a single layer: test-unit, test-regression, test-integration
```

**Inside the Docker image:**
```bash
docker run --rm -e DASHBOARD_HOST=localhost dashbite make test
```
Two tests check that the dashboard binds to `localhost` by default. The image sets `DASHBOARD_HOST=0.0.0.0`, so `-e DASHBOARD_HOST=localhost` restores the default for the test run.

---

## 4. Result of my manual smoke test

**Attempt 1: a 300× simulator clock (didn't work well).** I first smoke-tested with the simulator clock at 300×, which runs a simulated day in about 5 minutes. I wanted to speed up the demo, so edited some parameters with the help of the AI agent. It made the dashboard misleading: orders were stamped hours in the future, so Model Pulse's per-minute charts didn't match real time, and the page showed a fast-clock warning.

**Attempt 2: the default real-time clock (works).** I reverted the default to real time (1×), cleared the data, and ran the full stack again. I let it run for about **10 minutes**:

- All five processes started, and `make status` showed them all running.
- The simulator wrote a batch every 3s, and preprocess rejected the messy rows with reasons such as `out_of_range`, `duplicate_id` and `not_a_number`.
- Train published `model_v0001` after about a minute, and then a new version roughly every minute after that.
- **Infer picked up each new model and wrote predictions** to `data/predictions/`.
- Model Pulse at http://localhost:8501 showed orders per minute, the drop rate and the late-flag rate updating live.

**Docker smoke test:**

| Check | Result |
|---|---|
| `docker build -t dashbite .` | Succeeded in 37s, with an 800 KB build context, so `.dockerignore` kept the local data out |
| `docker run …` | All five processes started inside the container, and every stage's log streamed to the terminal |
| Data flowing | The simulator wrote batches, preprocess kept and rejected rows, and train counted toward 400 rows |
| Dashboard | Reachable from the host browser at http://localhost:8501, which confirms the `DASHBOARD_HOST` fix |
| `docker stop dashbite` | Stopped cleanly, and `--rm` removed the container |
| `docker compose up -d --build` | After 90s the container was `(healthy)`, `model_v0001` was published (accuracy 0.85 vs a 0.64 baseline), and 26 prediction files were written |
| Persistence: `docker compose down`, then `up -d` | `model_v0001` was still there, infer loaded it straight away, and predictions kept climbing (26 → 38) |
| Graceful shutdown: `docker compose stop` | All five stages stopped in reverse order, each logged its `stopped after …` summary, and the container exited with code 0 |
| Tests in the image | 543 passed, 1 skipped |

**A problem my smoke test caught:** at first, `docker compose down` finished in 0.26s. That was too fast: the stage processes were being killed instead of shutting down. I changed the container's start command to run `make stop` when it gets a stop signal, then re-tested. All five stages now stop cleanly.

---

## 5. How each AI role contributed

All three roles used Claude Opus 5.5 in Claude Code, each in a separate conversation. The full conversations are in [docs/transcripts/](docs/transcripts/).

### Architect ([av351_architect.txt](docs/transcripts/av351_architect.txt))
- Drew the pipeline diagram and wrote the living plan in [docs/plan.md](docs/plan.md), one stage at a time (Stages 1–7). Each stage covers its goal, files, boundaries, automated tests and a manual smoke test.
- Kept the plan in sync with what the Builder actually built, when I asked it to.
- Gave design recommendations, such as lowering how often training runs (see section 6).

### Builder ([av351_builder.txt](docs/transcripts/av351_builder.txt))
- Implemented Stages 1–7: config and paths, simulator, preprocess, train, infer, the Model Pulse dashboard, and the `make run`/`make stop` background stack. It wrote tests alongside the code.
- Explained its code when I asked, for example `clean_row` and how `train_model`'s logistic regression converges.
- Proposed new simulator settings, which I selected from (see sections 6 and 7).

### Tester ([av351_tester.txt](docs/transcripts/av351_tester.txt))
- Reviewed each stage as a PR it hadn't written, checked it against `docs/plan.md`, and ran the full suite.
- **Found a serious bug:** run from a folder whose path contains a space, `make clean-data` would have run `rm -rf` on the user's **whole Desktop folder**. Make now refuses to run from such a path.
- Added tests I asked for: zero or negative config values fail with a clear error, prediction files keep an exact format, and `make config` uses the right clock default.

### Docker
I did the Docker step manually with AI help, keeping it to a minimal `Dockerfile`, `.dockerignore` and `docker-compose.yml`. I made and checked the decisions myself: one container, a named volume for the data, a health check, keeping local state out with `.dockerignore`, the `DASHBOARD_HOST` fix, and the graceful-shutdown fix after my smoke test caught the problem.

---

## 6. AI recommendations I accepted

1. **Train every 400 rows instead of 2000 (Architect).** I first asked whether this would make training much slower. The Architect explained that it changes how *often* training runs, not how long each run takes, and the first model would appear after about 1 minute instead of 5. I accepted. The default is now `TRAIN_EVERY_N_EVENTS=400`.
2. **A `wrong_field_count` reject reason (Builder).** I noticed that rows with extra fields were silently accepted, with the extra value dropped. The Builder recommended a new reason code, so these rows are rejected instead of slipping through. I accepted it, and the Tester then verified the fix with new tests.
3. **The `make clean-data` path guard (Tester).** I had the Tester show me the exact `rm -rf` command before and after its fix. Seeing that it would have deleted my Desktop folder, I kept the fix.

## 7. AI recommendations I changed or rejected

1. **The simulator clock: I overruled the AI, then reverted after experimenting (Builder).** In Stage 6 the Builder changed the simulator's default clock from 300× to real time (1×), so Model Pulse's per-minute charts would match the clock. I overruled it and kept **300×**, because I wanted the demo to move faster. My smoke test showed that this didn't work well. At 300× the orders are stamped hours in the future, so the "last 60 minutes" windows and per-minute charts were misleading, and the dashboard showed a fast-clock warning. I **reverted the default to 1×**, kept 300× as an opt-in (`SIM_CLOCK_SPEED=300`) for the rush-hour demo, and re-ran the stack. Experimenting taught me a lot:
   - Speeding up the clock doesn't make the pipeline faster. It only changes the timestamps the orders carry.
   - A dashboard built on real-time windows needs real-time data.
   - The real lever for a quicker demo was the training threshold: 400 rows gives a first model in about a minute.
2. **Simulator settings (Builder), partly rejected and partly changed.** I asked for its top 3 new settings. It suggested a lateness threshold (default 35 min), a label-noise setting, and `SIM_DISTANCE_MEDIAN_KM`.
   - I **rejected** `SIM_DISTANCE_MEDIAN_KM` and didn't take the noise setting.
   - I **accepted** only the lateness threshold, and **changed its default from 35 to 30 minutes**.
3. **`make run` / `make stop`: build them now, not later (Architect).** In Stage 2 the Architect wrote these targets into the plan as *deferred*, to be added "in the first stage that has a second long-running process." I rejected that. I had the plan tell the Implementer to build them in that stage, with the word "deferred" removed, so the one-command start and stop wouldn't keep getting pushed to a later stage.
4. **Keep the original raw line in the rejects file (Tester).** After verifying the `wrong_field_count` fix, the Tester pointed out that the rejects file loses the actual bad values, for example the `EXTRA` in `…,0,EXTRA`. It chose to leave that as is, because the rejects file's columns are pinned by tests. I changed that decision. I had it add a `raw_line` column holding the exact original text, so anyone debugging a reject can see exactly what the bad line looked like.

---

## 8. How I independently verified the final result

- **I ran it myself.** I ran the full stack on my machine for about 10 minutes and watched predictions appear (section 4). I built and ran the Docker image and opened the dashboard from my browser.
- **Tests.** `make test` passes on the host (544 tests), and the same suite passes inside the Docker image (543 passed, 1 skipped).
- **I questioned the AI instead of trusting it.** For example:
  - I had the Builder explain `clean_row` and `train_model`.
  - I asked whether the default L2 penalty shrinks the distance and minutes coefficients unequally.
  - I made the Tester show the exact `rm -rf` command before and after its fix.
  - I asked the Builder to cross-check whether `make config` was overriding the clock default.
- **I caught gaps myself.** I spotted that extra-field rows were silently kept, and that the clock default behaved badly at 300×. I had both fixed and re-tested.

---

## Reference

### Settings
Every setting is an environment variable that you put in front of the command. `make config` prints the resolved values.

```bash
TRAIN_EVERY_N_EVENTS=200 make run   # retrain more often
POLL_INTERVAL_SECONDS=5 make run    # check for new files every 5s instead of 15s
DASHBOARD_PORT=8502 make dashboard  # use another port
SIM_CLOCK_SPEED=300 make simulator  # opt-in fast clock for the rush-hour demo (not with the dashboard)
```

With Docker, pass settings with `-e`. For example:
```bash
docker run --rm --init -p 8501:8501 -e TRAIN_EVERY_N_EVENTS=200 --name dashbite dashbite
```

### Reset
```bash
make stop
make clean-data   # removes data/ only
docker compose down -v   # stop the container and delete its data volume
docker rmi dashbite      # remove the Docker image
```
