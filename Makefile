# Pipeline reproducible de 1 comando. Ver PROJECT.md §8.
.PHONY: setup run refresh watch snapshot test lint fmt clean

VENV   := .venv
PYTHON ?= python3      # intérprete base (>=3.11); override: make setup PYTHON=python3.11
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
BIN    := $(VENV)/bin

setup:  ## crea venv e instala deps (pyproject.toml)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

run:  ## pipeline completo en modo live: download -> recondiciona -> simula -> figuras
	$(PY) scripts/run_pipeline.py --config config/config.yaml

refresh:  ## baja nuevo snapshot y re-simula solo lo pendiente
	$(PY) scripts/run_pipeline.py --config config/config.yaml --refresh

watch:  ## loop: re-corre cada N segundos mientras hay partidos (uso durante el torneo)
	$(PY) scripts/run_pipeline.py --config config/config.yaml --watch --interval 600

test:  ## pytest
	$(BIN)/pytest

lint:  ## ruff + black --check + mypy
	$(BIN)/ruff check src tests
	$(BIN)/black --check src tests
	$(BIN)/mypy src

fmt:  ## black + ruff --fix
	$(BIN)/black src tests
	$(BIN)/ruff check --fix src tests

clean:  ## borra venv y cachés (NUNCA borra data/raw)
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
