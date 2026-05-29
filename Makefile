PYTHON ?= python
THREADS ?= $(shell $(PYTHON) -c 'import os; print(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)')
LOG_ROOT ?= ../logs

.PHONY: phase0 smoke test tree clean

phase0:
	@mkdir -p $(LOG_ROOT)
	@RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ); \
	mkdir -p "$(LOG_ROOT)/manual_phase0_$${RUN_ID}"; \
	PNA_THREADS=$(THREADS) PYTHONPATH=src $(PYTHON) scripts/wrds_schema_discovery.py \
		--project-root . \
		--out-dir artifacts/schema_discovery \
		--log-dir "$(LOG_ROOT)/manual_phase0_$${RUN_ID}" \
		--threads $${PNA_METADATA_THREADS:-8} \
		2>&1 | tee "$(LOG_ROOT)/manual_phase0_$${RUN_ID}/wrds_schema_discovery.console.log"

smoke:
	@PYTHONPATH=src $(PYTHON) -m compileall -q src scripts
	@PYTHONPATH=src $(PYTHON) -c "import production_network_alpha; print('import ok')"

test:
	@PYTHONPATH=src $(PYTHON) -m pytest -q

tree:
	@find . -maxdepth 3 -type d | sort

clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
