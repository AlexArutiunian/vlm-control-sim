# dataset_player.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional
import argparse
import importlib



def load_commands_csv(csv_path: str | Path) -> List[Dict[str, str]]:
    """Загрузить commands.csv -> [{'id': str, 'command': str}, ...]."""
    p = Path(csv_path)
    if not p.exists():
        print(f"[WARN] commands.csv not found at: {p.resolve()}")
        return []

    rows: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[WARN] commands.csv has no header.")
            return []

        # нормализуем заголовки
        fmap = {k.lower().strip(): k for k in reader.fieldnames}
        id_col = fmap.get("id")
        cmd_col = fmap.get("command") or fmap.get("cmd")
        if not id_col or not cmd_col:
            print("[WARN] commands.csv must contain columns: id, command")
            return []

        for row in reader:
            cid = str(row.get(id_col, "")).strip()
            cmd = str(row.get(cmd_col, "")).strip()
            if cid and cmd:
                rows.append({"id": cid, "command": cmd})
    return rows


def load_actions_json(json_dir: str | Path, command_id: str) -> Optional[List[Dict[str, Any]]]:
    """Загрузить actions из data_action/<id>.json. Вернёт None, если файла нет/ошибка."""
    p = Path(json_dir) / f"{command_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
            print(f"[WARN] {p} — ожидался список объектов действий.")
            return None
        return data
    except Exception as e:
        print(f"[WARN] failed to read {p.name}: {e}")
        return None


def run_dataset_player(
    move_joints_by_name: Callable[[List[Dict[str, Any]]], None],
    csv_path: str | Path = "commands.csv",
    json_dir: str | Path = "data_action",
    pause_between: float = 0.0,
) -> None:
    """
    Для каждой строки из commands.csv:
      - читает data_action/<id>.json
      - печатает id, команду и JSON массив действий
      - проигрывает через move_joints_by_name
    """
    import time

    commands = load_commands_csv(csv_path)
    if not commands:
        print("[INFO] nothing to play.")
        return

    print(f"[INFO] loaded {len(commands)} rows from {Path(csv_path).resolve()}")
    print(f"[INFO] actions dir: {Path(json_dir).resolve()}\n")

    for row in commands:
        cid = row["id"]
        cmd = row["command"]

        actions = load_actions_json(json_dir, cid)
        if actions is None:
            print(f"[SKIP] id={cid} — no JSON found, command='{cmd}'")
            continue

        # выводим аккуратно
        print("=" * 80)
        print(f"id: {cid}")
        print(f"command: {cmd}")
        print("actions:")
        try:
            print(json.dumps(actions, ensure_ascii=False, indent=2))
        except Exception:
            print(str(actions))

        # проигрываем
        try:
            move_joints_by_name(actions)
            print(f"[OK] executed id={cid}\n")
        except Exception as e:
            print(f"[ERR] execution failed for id={cid}: {e}\n")

        if pause_between > 0:
            time.sleep(pause_between)


def _cli() -> None:
    """
    CLI-режим.
    Пример:
      python dataset_player.py --csv commands.csv --json-dir data_action \
          --control-module main_viewer_module --pause 0.5

    Ожидается, что в --control-module есть функция move_joints_by_name(actions: list[dict]) -> None.
    """
    parser = argparse.ArgumentParser(description="Play robot actions from CSV + JSON dataset.")
    parser.add_argument("--csv", dest="csv_path", default="commands.csv", help="Path to commands.csv")
    parser.add_argument("--json-dir", dest="json_dir", default="data_action", help="Directory with <id>.json files")
    parser.add_argument("--control-module", dest="control_module", default=None,
                        help="Python module that provides move_joints_by_name(actions)")
    parser.add_argument("--pause", dest="pause", type=float, default=0.0,
                        help="Pause (seconds) between samples")

    args = parser.parse_args()

    

    if args.control_module:
        mod = importlib.import_module(args.control_module)
        if not hasattr(mod, "move_joints_by_name"):
            raise RuntimeError(f"Module '{args.control_module}' must export move_joints_by_name(actions).")
        move_fn = getattr(mod, "move_joints_by_name")
    else:
        # Если модуль управления не указан — подсказываем и выходим.
        raise SystemExit(
            "Please provide --control-module <module_with_move_joints_by_name> "
            "or import run_dataset_player() and pass your move_joints_by_name explicitly."
        )
    mod = importlib.import_module(args.control_module)
# инициализация среды, если предусмотрена
    if hasattr(mod, "ensure_initialized"):
        mod.ensure_initialized()
    if not hasattr(mod, "move_joints_by_name"):
        raise RuntimeError(...)
    move_fn = getattr(mod, "move_joints_by_name")
    run_dataset_player(
        move_joints_by_name=move_fn,
        csv_path=args.csv_path,
        json_dir=args.json_dir,
        pause_between=args.pause,
    )


if __name__ == "__main__":
    _cli()
