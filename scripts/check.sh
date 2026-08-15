#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/3] Compile Python files"
uv run python -m py_compile app/*.py experiments/*.py tools/*.py

echo "[2/3] Check required files"
for path in \
  README.md \
  pyproject.toml \
  prompt.txt \
  prompt_twins.txt \
  data/rag/rag_best.csv \
  data/motions/best \
  media/demos \
  unitree_rl_gym; do
  if [[ ! -e "$path" ]]; then
    echo "Missing: $path" >&2
    exit 1
  fi
done

echo "[3/3] Check main entrypoints"
for path in \
  app/any_robots.py \
  app/any_ag_micro.py \
  app/any_3h1_2.py \
  app/any_3h1_2_grab.py \
  app/any_robots_map.py; do
  if [[ ! -f "$path" ]]; then
    echo "Missing: $path" >&2
    exit 1
  fi
done

echo "Repository check passed."
