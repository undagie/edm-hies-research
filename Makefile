# Reproducibility runbook for the HIES benchmark.
#
# Targets:
#   make install      # install pinned Python dependencies
#   make smoke        # fast end-to-end check (~5 minutes)
#   make reproduce    # full benchmark + all analyses (~2-3 hours)
#   make analyses     # rerun downstream analyses only (no benchmark)
#   make test         # pytest unit + integration tests
#   make clean        # remove only the generated artifact directories
#
# On Windows without GNU make, run the commands directly:
#   python -m pip install -r requirements.txt
#   python -m src.main --config configs/smoke.yaml
#   python scripts/reproduce_full.py
#   python scripts/reproduce_full.py --skip-benchmark
#   python -m pytest tests/

PYTHON ?= python

.PHONY: install smoke reproduce analyses test clean

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) -m src.main --config configs/smoke.yaml

reproduce:
	$(PYTHON) scripts/reproduce_full.py

analyses:
	$(PYTHON) scripts/reproduce_full.py --skip-benchmark

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf artifacts/results artifacts/tables artifacts/figures \
		artifacts/manifests artifacts/logs artifacts/phase1_analysis \
		artifacts_smoke
