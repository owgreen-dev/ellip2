#!/usr/bin/env bash
# The Ralph loop's single verification gate. Must be ONE command: the loop runs
# $VERIFY_COMMAND via unquoted shell expansion (ralph-fresh.sh / stop-hook.sh), so a
# "pytest && ruff && mypy" string would break — hence this wrapper.
#
# Runs everything through the project .venv (the env has no system pip; deps live in
# .venv). Exits non-zero on the first failing stage so Ralph re-prompts to fix it.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

echo "== pytest =="
"$PY" -m pytest -q || exit 1

echo "== ruff =="
"$PY" -m ruff check src tests || exit 1

echo "== mypy =="
"$PY" -m mypy src || exit 1

echo "verify.sh: all gates passed"
