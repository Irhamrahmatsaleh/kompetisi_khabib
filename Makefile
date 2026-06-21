PYTHON ?= python3
VENV ?= .venv
RUNPY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install serve clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

serve:
	$(RUNPY) analyzer.py

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
