# ellip2 — developer entry points.
# The repo uses uv + a project .venv (no system pip). Bootstrap once with `make install`,
# then every target runs through .venv/bin/python.
PY := .venv/bin/python

.PHONY: help install demo check-numbers verify test kaggle-push

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

# Requires `uv pip install kaggle` (2.x). Reads a token from KAGGLE_API_TOKEN, or from a
# gitignored .env (KAGGLE_API_TOKEN, else the new-style token stored as KAGGLE_KEY). The `id`
# in notebooks/kernel-metadata.json must start with YOUR Kaggle username. Never run in CI.
kaggle-push:  ## Publish notebooks/ as a Kaggle notebook (needs kaggle 2.x + token in .env or KAGGLE_API_TOKEN)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	KAGGLE_API_TOKEN="$${KAGGLE_API_TOKEN:-$$KAGGLE_KEY}" $(PY) -m kaggle kernels push -p notebooks/
