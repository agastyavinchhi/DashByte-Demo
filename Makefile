# DashBite — the Makefile is the interface. Every stage runs through a target here.

ROOT   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV   := $(ROOT)/.venv
PYTHON := $(VENV)/bin/python

# Make splits paths on spaces, so ROOT would be wrong and clean-data's rm -rf
# could hit a parent folder. Refuse to run at all rather than guess.
ifneq ($(words $(ROOT)),1)
$(error DashBite cannot run from a path containing spaces. Move or rename the project folder)
endif
ifeq ($(wildcard $(ROOT)/pipeline/config.py),)
$(error Could not locate the DashBite project folder (got $(ROOT)); check its path for spaces)
endif

.PHONY: help install test test-unit test-regression test-integration config clean-data \
        simulator preprocess train infer dashboard run stop status logs

help:
	@echo "make install           create .venv/ and install requirements.txt"
	@echo "make test              run all tests"
	@echo "make test-unit         run unit tests only"
	@echo "make test-regression   run regression tests only"
	@echo "make test-integration  run integration tests only"
	@echo "make config            print resolved config and create/confirm data/ dirs"
	@echo "make clean-data        remove data/ (never touches code or docs/)"
	@echo ""
	@echo "Foreground, one stage per terminal (Ctrl+C to stop):"
	@echo "make simulator         write a batch of orders to data/raw/ every few seconds (Ctrl+C to stop)"
	@echo "make preprocess        clean each raw batch into data/features/ (rejects → data/quality/)"
	@echo "make train             publish a new model to data/models/ every TRAIN_EVERY_N_EVENTS new rows"
	@echo "make infer             score new feature batches with the newest model into data/predictions/"
	@echo "make dashboard         Model Pulse: a read-only ML health page at http://localhost:8501"
	@echo ""
	@echo "Background stack (survives closing the terminal):"
	@echo "make run               start simulator, preprocess, train, infer and the dashboard"
	@echo "make status            show which background processes are running"
	@echo "make logs              follow every background log (Ctrl+C stops following only)"
	@echo "make stop              stop everything make run started"
	@echo ""
	@echo "Poll cadence: POLL_INTERVAL_SECONDS (default 15) for preprocess, train, infer and the dashboard."

install:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --quiet --upgrade pip
	$(PYTHON) -m pip install --quiet -r $(ROOT)/requirements.txt
	@echo "installed into $(VENV)"

test:
	cd $(ROOT) && $(PYTHON) -m pytest

test-unit:
	cd $(ROOT) && $(PYTHON) -m pytest -m unit

test-regression:
	cd $(ROOT) && $(PYTHON) -m pytest -m regression

test-integration:
	cd $(ROOT) && $(PYTHON) -m pytest -m integration

config:
	@cd $(ROOT) && $(PYTHON) -m pipeline.config

# -u keeps the log unbuffered so each batch shows up the moment it lands.
simulator:
	@cd $(ROOT) && $(PYTHON) -u -m pipeline.simulator

preprocess:
	@cd $(ROOT) && $(PYTHON) -u -m pipeline.preprocess

train:
	@cd $(ROOT) && $(PYTHON) -u -m pipeline.train

infer:
	@cd $(ROOT) && $(PYTHON) -u -m pipeline.infer

run:
	@cd $(ROOT) && $(PYTHON) -m pipeline.runner start

stop:
	@cd $(ROOT) && $(PYTHON) -m pipeline.runner stop

# Foreground Model Pulse (Ctrl+C to stop). `make run` starts the same launcher
# in the background. pipeline.dashboard_server validates DASHBOARD_PORT (a bad
# value fails here, naming the variable), refuses a busy port, and runs
# Streamlit headless on localhost with usage stats off.
# The urllib3 filter hides a harmless LibreSSL warning from macOS's Python.
dashboard:
	@cd $(ROOT) && PYTHONWARNINGS="ignore::Warning:urllib3" \
	exec $(PYTHON) -u -m pipeline.dashboard_server

status:
	@cd $(ROOT) && $(PYTHON) -m pipeline.runner status

# Follow every background log at once (Ctrl+C stops following, not the stack).
logs:
	@cd $(ROOT) && RUN_DIR=$$($(PYTHON) -m pipeline.runner logs-dir) && \
	ls $$RUN_DIR/*.log >/dev/null 2>&1 || { echo "no logs yet: start the stack with make run"; exit 0; }; \
	exec tail -n 5 -F $$RUN_DIR/*.log

# Only ever removes $(ROOT)/data. A custom DASHBITE_DATA_DIR is not ours to delete.
clean-data:
ifneq ($(strip $(DASHBITE_DATA_DIR)),)
	$(error DASHBITE_DATA_DIR is set to $(DASHBITE_DATA_DIR); clean-data only removes $(ROOT)/data. Unset it, or delete that dir yourself)
endif
	rm -rf "$(ROOT)/data"
	@echo "removed $(ROOT)/data"
