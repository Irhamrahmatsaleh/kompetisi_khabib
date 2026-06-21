PYTHON ?= python3
PIP ?= pip3
SYMBOL ?= BTC-USD
PERIOD ?= 6mo
INTERVAL ?= 1d

.PHONY: install serve

install:
	$(PIP) install -r requirements.txt

serve:
	$(PYTHON) analyzer.py --symbol $(SYMBOL) --period $(PERIOD) --interval $(INTERVAL)
