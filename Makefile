# ellip2 — developer entry points.
# The repo uses uv + a project .venv (no system pip). Bootstrap once with `make install`,
# then every target runs through .venv/bin/python.
PY := .venv/bin/python

.PHONY: help install demo check-numbers verify test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install the package + dev extra
	uv venv .venv
	uv pip install --python $(PY) -e '.[dev]'

demo:  ## Keyless end-to-end demo on a synthetic toy graph (CPU, <2 min, no credentials)
	$(PY) scripts/demo.py

check-numbers:  ## Assert README/RESULTS numbers match facts.json (drift-check)
	$(PY) -m pytest tests/test_published_numbers.py -q

verify:  ## Full gate: pytest + ruff + mypy (same as CI)
	bash scripts/verify.sh

test:  ## Run the test suite
	$(PY) -m pytest -q
