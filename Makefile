# One-command reproducible pipeline.
.PHONY: setup run refresh watch snapshot test lint fmt clean

VENV   := .venv
PYTHON ?= python3      # base interpreter (>=3.11); override: make setup PYTHON=python3.11
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
BIN    := $(VENV)/bin

setup:  ## create venv and install deps (pyproject.toml)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

run:  ## full pipeline in live mode: download -> recondition -> simulate -> figures
	$(PY) scripts/run_pipeline.py --config config/config.yaml

refresh: run  ## re-simulate with the latest results (live mode; alias for run)

watch:  ## loop: re-run every N seconds while matches are on (use during the tournament)
	$(PY) scripts/run_pipeline.py --config config/config.yaml --watch --interval 600

test:  ## pytest
	$(BIN)/pytest

lint:  ## ruff + black --check + mypy
	$(BIN)/ruff check src tests scripts app
	$(BIN)/black --check src tests scripts app
	$(BIN)/mypy src tests scripts app

fmt:  ## black + ruff --fix
	$(BIN)/black src tests scripts app
	$(BIN)/ruff check --fix src tests scripts app

clean:  ## remove venv and caches (NEVER touches data/raw)
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
