#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/4] Compile Python files"
uv run python -m py_compile app/*.py legacy/*.py utilities/*.py

echo "[2/4] Check repository files"
for path in \
  README.md \
  pyproject.toml \
  prompt.txt \
  prompt_twins.txt \
  data/rag/rag_best.csv \
  data/motions/best \
  media/demos; do
  if [[ ! -e "$path" ]]; then
    echo "Missing: $path" >&2
    exit 1
  fi
done

echo "[3/4] Check runtime assets"
for path in \
  unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml \
  unitree_rl_gym/deploy/deploy_mujoco/configs/h1.yaml \
  unitree_rl_gym/deploy/deploy_mujoco/configs/h1_2.yaml \
  unitree_rl_gym/deploy/pre_train/g1/motion.pt \
  unitree_rl_gym/deploy/pre_train/h1/motion.pt \
  unitree_rl_gym/deploy/pre_train/h1_2/motion.pt \
  unitree_rl_gym/resources/robots/g1 \
  unitree_rl_gym/resources/robots/h1 \
  unitree_rl_gym/resources/robots/h1_2; do
  if [[ ! -e "$path" ]]; then
    echo "Missing runtime asset: $path" >&2
    exit 1
  fi
done

echo "[4/4] Check main entrypoints"
for path in \
  app/any_robots.py \
  app/any_ag_micro.py \
  app/any_3h1_2.py \
  app/any_3h1_2_grab.py \
  app/any_robots_map.py; do
  if [[ ! -f "$path" ]]; then
    echo "Missing entrypoint: $path" >&2
    exit 1
  fi
done

echo "Repository check passed."
