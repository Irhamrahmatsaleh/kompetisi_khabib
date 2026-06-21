PYTHON ?= python3
VENV ?= .venv
RUNPY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install serve core clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

serve:
	$(RUNPY) competition_scanner.py --symbols ALL --top 20

core:
	$(RUNPY) competition_scanner.py --symbols CORE --top 12

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
