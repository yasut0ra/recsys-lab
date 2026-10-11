PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: init run modern check docs-check clean

init:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PY) -m recsys_lab.run

modern:
	$(PY) -m recsys_lab.run_modern

docs-check:
	$(PY) scripts/check_materials.py

check:
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf __pycache__ recsys_lab/__pycache__ tests/__pycache__
