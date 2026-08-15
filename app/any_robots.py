# =========================
# Imports
# =========================
import time
import math
import json
import ast
import re
from typing import List, Dict, Any
from pathlib import Path
import json
import math
import numpy as np
import imageio.v2 as imageio

import numpy as np
import yaml
import torch
import tkinter as tk
import csv
import uuid
from pathlib import Path

import mujoco as mj
from mujoco.glfw import glfw

from tkinter import filedialog

from llm_providers import build_llm
import settings

# =========================
# Imports (extended)
# =========================
# ... existing import ...
from demo_player import load_commands_csv, load_actions_json  # Add functions from demo_player.py

# --- Few-shot via cosine similarity (new) ---
from sim_phrases import load_csv as load_cmds_csv, compute_similarities, DEFAULT_MODEL  # noqa

# --- UPPER-BODY PD control (replaces the old ARM_* block) ---
UPPER_HOLD_ZERO = True   # <— ключ: держать нули по умолчанию

UPPER_JOINTS: list[str] = []     # имена суставов верхней части (динамически)
ARM_NAME_TO_IDX = {}
ARM_JIDS = []
ARM_ACT_IDS = []
ARM_QPOS_ADDRS = []
ARM_QVEL_ADDRS = []
ARM_TARGETS = None
ARM_KPS = None
ARM_KDS = None
ARM_KIS = None
ARM_IERR = None
ARM_I_CLAMP = 0.7  # рад*с, ограничение интеграла

def log_zeroed_upper():
    if not ARM_NAME_TO_IDX:
        print("[UPPER] no joints bound")
        return
    names = sorted(ARM_NAME_TO_IDX.keys())
    print(f"[UPPER] holding ZERO on {len(names)} joints:")
    for nm in names:
        idx = ARM_NAME_TO_IDX[nm]
        adr = ARM_QPOS_ADDRS[idx]
        cur_deg = math.degrees(d.qpos[adr])
        print(f"  - {nm:30s} target=0.0 rad (was {cur_deg:+6.1f}°)")

def _is_upper_joint(name: str) -> bool:
    return (
        name == "torso_joint" or
        name.startswith((
            "waist_",
            "left_shoulder_", "right_shoulder_",
            "left_elbow", "right_elbow",
            "left_wrist_", "right_wrist_",
            "left_hand_", "right_hand_",    # ← ВАЖНО: кисть и пальцы
            "L_", "R_"
        ))
    )


def _suggest_kp(name: str) -> float:
    # чуть жёстче плечо/торс, мягче кисть, ещё мягче пальцы
    if name == "torso_joint": return 40.0
    if "shoulder" in name:   return 40.0
    if "elbow" in name:      return 30.0
    if "wrist" in name:      return 12.0
    # пальцы (L_/R_*)
    return 4.0

def setup_arm_pd(zero_pose: bool = True):
    """Привязать ВСЕ суставы верхней части и инициализировать PD-цели.
    Если у актуатора нет name==joint_name, ищем по trnid (target=этот joint)."""
    global UPPER_JOINTS, ARM_NAME_TO_IDX, ARM_JIDS, ARM_ACT_IDS
    global ARM_QPOS_ADDRS, ARM_QVEL_ADDRS, ARM_TARGETS, ARM_KPS, ARM_KDS
    global ARM_KIS, ARM_IERR
    ARM_NAME_TO_IDX.clear()
    ARM_JIDS.clear(); ARM_ACT_IDS.clear(); ARM_QPOS_ADDRS.clear(); ARM_QVEL_ADDRS.clear()

    # 1) собрать имена верхних hinge-суставов
    UPPER_JOINTS = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
        if _is_upper_joint(nm):
            UPPER_JOINTS.append(nm)

    # 2) построить фолбэк-карту: joint_id -> actuator_id (для actuator с JOINT target)
    trn  = np.array(m.actuator_trnid).reshape(m.nu, 2) if m.nu > 0 else np.zeros((0, 2), dtype=int)
    trnt = np.array(m.actuator_trntype) if m.nu > 0 else np.zeros((0,), dtype=int)
    jointid_to_act = {}
    for i in range(m.nu):
        if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
            jid_target = int(trn[i, 0])
            # первый найденный считаем основным
            jointid_to_act.setdefault(jid_target, i)

    # 3) связать joint ↔ actuator (сначала по имени, иначе по trnid)
    names_in_order = []
    for name in UPPER_JOINTS:
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"[UPPER] joint '{name}' not found — skip")
            continue

        # пробуем имя
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)
        # если имени нет — берём по целевому joint'у
        if aid < 0:
            aid = jointid_to_act.get(jid, -1)

        if aid < 0:
            print(f"[UPPER] actuator for '{name}' not found (jid={jid}) — skip")
            continue

        idx = len(ARM_JIDS)
        ARM_NAME_TO_IDX[name] = idx
        ARM_JIDS.append(jid)
        ARM_ACT_IDS.append(aid)
        ARM_QPOS_ADDRS.append(m.jnt_qposadr[jid])
        ARM_QVEL_ADDRS.append(m.jnt_dofadr[jid])
        names_in_order.append(name)

    n = len(ARM_JIDS)
    if n == 0:
        print("[UPPER] no upper-body joints bound")
        ARM_TARGETS = np.zeros(0, dtype=np.float32)
        ARM_KPS = np.zeros(0, dtype=np.float32)
        ARM_KDS = np.zeros(0, dtype=np.float32)
        return

    # 4) цели: либо нули, либо текущие значения
    if zero_pose:
        ARM_TARGETS = np.zeros(n, dtype=np.float32)
        log_zeroed_upper()
    else:
        ARM_TARGETS = np.array([d.qpos[i] for i in ARM_QPOS_ADDRS], dtype=np.float32)

    # 5) Kp/Kd в порядке индексов
    ARM_KPS = np.array([_suggest_kp(nm) for nm in names_in_order], dtype=np.float32)
    ARM_KDS = np.clip(ARM_KPS * 0.05, 0.2, None).astype(np.float32)
    ARM_KIS = np.clip(ARM_KPS * 0.02, 0.0, 1.0).astype(np.float32)  # мягкий интеграл
    ARM_IERR = np.zeros_like(ARM_KPS, dtype=np.float32)

    print(f"[UPPER] PD enabled for {n} joints:", names_in_order)


def print_gait_policy_io():
    na = int(num_actions)

    # === входы политики (наблюдения) ===
    qpos_start = 7
    qvel_start = 6

    names_pos = []
    names_vel = []
    for j in range(m.njnt):
        jt = m.jnt_type[j]
        if jt in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            adr_q = m.jnt_qposadr[j]
            adr_v = m.jnt_dofadr[j]
            if qpos_start <= adr_q < qpos_start + na:
                names_pos.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j))
            if qvel_start <= adr_v < qvel_start + na:
                names_vel.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j))

    print(f"[POLICY IN qpos 7:{7+na}] joints: {', '.join(names_pos)}")
    print(f"[POLICY IN qvel 6:{6+na}] joints: {', '.join(names_vel)}")

    # === выходы политики (актуаторы, куда пишем d.ctrl[:na]) ===
    names_act = []
    trn = np.array(m.actuator_trnid).reshape(m.nu, 2)
    trntype = np.array(m.actuator_trntype)
    for i in range(min(na, m.nu)):
        if trntype[i] == mj.mjtTrn.mjTRN_JOINT:
            jid = int(trn[i, 0])
            names_act.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, jid))
        else:
            names_act.append(f"act{i}: non-joint target")
    print(f"[POLICY OUT d.ctrl[:{na}]] actuators -> joints: {', '.join(names_act)}")

def _load_top_sim_examples(
    user_cmd: str,
    csv_path: str = "1000.csv",
    json_dir: str | Path = "data_action",
    k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[tuple[str, dict]]:
    """
    Returns a list of pairs (command_text, example_json) up to length k,
    sorted by descending cosine similarity to user_cmd.
    Skips examples without a JSON file.
    Safely returns [] on any error (missing model/files, etc.).
    """
    try:
        df = load_cmds_csv(csv_path)
        ranked = compute_similarities(user_cmd, df, model_name)
    except Exception as e:
        print(f"[few-shot] skip (similarity failed): {e}")
        return []

    out: list[tuple[str, dict]] = []
    json_dir = Path(json_dir)
    for _, row in ranked.head(k).iterrows():
        cid = str(row["id"]).strip()
        cmd_text = str(row["command"]).strip()
        jf = json_dir / f"{cid}.json"
        if not jf.exists():
            continue
        try:
            with jf.open("r", encoding="utf-8") as f:
                ex = json.load(f)
            # normalize to list (as expected by your parser)
            if isinstance(ex, dict):
                ex = [ex]
            if not isinstance(ex, list):
                continue
        except Exception as e:
            print(f"[few-shot] bad json for id={cid}: {e}")
            continue
        out.append((cmd_text, ex))
    return out

def run_walk_blocking(num: int, dir_deg: float, spd: float):
    """Start walking and block until steps finish; keep arms with PD targets."""
    global steps_done, steps_needed, counter, target_dof_pos, old_sin_phase
    global action, obs, cmd, default_angles
    global m, d, opt, cam, scene, context, window
    global ARM_IERR
    start_new_steps(num, dir_deg, spd)

    while steps_done < steps_needed and not glfw.window_should_close(window):
        if _should_abort():
            break

        # --- legs (gait) PD control from RL target_dof_pos ---
        na = num_actions
        tau = pd_control(
            target_dof_pos,
            d.qpos[7:7 + na],
            kps,
            np.zeros_like(kds),
            d.qvel[6:6 + na],
            kds
        )
        for k, aid in enumerate(GAIT_ACT_IDS):
            d.ctrl[aid] = float(tau[k])

        # --- arms: hold PD targets to prevent drifting/crossing ---
        if ARM_ACT_IDS:
            q  = d.qpos[ARM_QPOS_ADDRS]
            dq = d.qvel[ARM_QVEL_ADDRS]
            # ошибка и интеграл
            err = ARM_TARGETS - q
            ARM_IERR += err * m.opt.timestep
            ARM_IERR = np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP)

            # компенсация тяжести/пассивки для этих dof'ов
            bias = d.qfrc_bias[ARM_QVEL_ADDRS]  # размер = len(ARM_ACT_IDS)

            # итоговый момент
            tau_arm = bias + ARM_KPS * err - ARM_KDS * dq + ARM_KIS * ARM_IERR

            # извлечь gear для actuator'ов (для JOINT берём компоненту [0])
            act_gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else np.zeros((0,6))

            for k, act_id in enumerate(ARM_ACT_IDS):
                g = float(act_gears[act_id, 0]) if act_gears.size else 1.0
                lo, hi = m.actuator_ctrlrange[act_id]
                u = tau_arm[k] / max(g, 1e-6)        # масштаб: ctrl = τ / gear
                d.ctrl[act_id] = float(np.clip(u, lo, hi))

        mj.mj_step(m, d)
        counter += 1

        # --- RL policy & step-phase bookkeeping on decimated ticks ---
        if counter % control_decimation == 0:
            qj = d.qpos[7:7 + na]
            dqj = d.qvel[6:6 + na]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_scaled = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = 0.8
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = qj_scaled
            obs[9 + na: 9 + 2 * na] = dqj_scaled
            obs[9 + 2 * na: 9 + 3 * na] = action
            obs[9 + 3 * na: 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action[:] = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

            # step counter: rising zero-crossing of sin_phase
            if sin_phase >= 0 and old_sin_phase < 0:
                if STANCE_HOLD:
                    _stance_local += 1
                    if _stance_local >= STANCE_STEPS_EACH:
                        _stance_local = 0
                        _stance_dir *= -1
                        v = _stance_dir * STANCE_V
                        if STANCE_AXIS == "x":
                            cmd[:] = [v, 0.0, 0.0]
                        else:
                            cmd[:] = [0.0, v, 0.0]
                else:
                    steps_done += 1
                    print(f"Got step #{steps_done} / {steps_needed}")
                    if steps_done >= steps_needed:
                        cmd[:] = 0.0
                        target_dof_pos = default_angles.copy()

            old_sin_phase = sin_phase

        # --- render ---
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    # stop commands at exit
    cmd[:] = 0.0
    d.ctrl[:] = 0.0

def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw around +Z (MuJoCo: Z-up)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def run_turn_blocking(angle_deg: float, spd_deg_s: float = 45.0):
    """
    Smoothly rotate around the body Z axis by the given angle (in degrees).
    Positive — counterclockwise when viewed from above (standard +Z).
    Blocking, like run_walk_blocking: keep running PD/gait until the turn completes.
    """
    global ARM_IERR
    stop_stance_hold()
    global m, d, policy, counter, cmd, action, target_dof_pos
    global default_angles, obs, old_sin_phase
    # turn speed (rad/s) with the correct sign
    direction = 1.0 if angle_deg >= 0 else -1.0
    yaw_rate = direction * math.radians(abs(spd_deg_s))
    target_abs = math.radians(abs(angle_deg))

    # zero linear speed, set angular speed
    cmd[:] = [0.0, 0.0, yaw_rate]

    # initial yaw and accumulator with wraparound across -pi..pi
    qw, qx, qy, qz = d.qpos[3:7]
    last_yaw = _quat_yaw(qw, qx, qy, qz)
    acc = 0.0

    while (acc < target_abs) and not glfw.window_should_close(window) and not RUN_ABORT:
        if _should_abort():
            break   # === same as in your main loop ===
        na = num_actions  # fixed to 10 in your script
        tau = pd_control(
            target_dof_pos,
            d.qpos[7:7 + na],
            kps,
            np.zeros_like(kds),
            d.qvel[6:6 + na],
            kds
        )
        # ноги: применяем только к актуаторам походки (плечи не трогаем)
        for k, aid in enumerate(GAIT_ACT_IDS):
            d.ctrl[aid] = float(tau[k])

        # руки: держим PD-контролем, чтобы RL не «подхватывал» плечи
        if ARM_ACT_IDS:
            q  = d.qpos[ARM_QPOS_ADDRS]
            dq = d.qvel[ARM_QVEL_ADDRS]
            # ошибка и интеграл
            err = ARM_TARGETS - q
            ARM_IERR += err * m.opt.timestep
            ARM_IERR = np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP)

            # компенсация тяжести/пассивки для этих dof'ов
            bias = d.qfrc_bias[ARM_QVEL_ADDRS]  # размер = len(ARM_ACT_IDS)

            # итоговый момент
            tau_arm = bias + ARM_KPS * err - ARM_KDS * dq + ARM_KIS * ARM_IERR

            # извлечь gear для actuator'ов (для JOINT берём компоненту [0])
            act_gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else np.zeros((0,6))

            for k, act_id in enumerate(ARM_ACT_IDS):
                g = float(act_gears[act_id, 0]) if act_gears.size else 1.0
                lo, hi = m.actuator_ctrlrange[act_id]
                u = tau_arm[k] / max(g, 1e-6)        # масштаб: ctrl = τ / gear
                d.ctrl[act_id] = float(np.clip(u, lo, hi))

        mj.mj_step(m, d)
        counter += 1

        if counter % control_decimation == 0:
            qj = d.qpos[7:7 + na]
            dqj = d.qvel[6:6 + na]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_scaled = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = 0.8
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = qj_scaled
            obs[9 + na: 9 + 2 * na] = dqj_scaled
            obs[9 + 2 * na: 9 + 3 * na] = action
            obs[9 + 3 * na: 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action[:] = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        # render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        # update accumulated rotation
        qw, qx, qy, qz = d.qpos[3:7]
        yaw_now = _quat_yaw(qw, qx, qy, qz)
        inc = _wrap_to_pi(yaw_now - last_yaw)
        acc += abs(inc)
        last_yaw = yaw_now

    # stop rotation
    cmd[:] = 0.0
    d.ctrl[:] = 0.0


# =========================
# Global variables (extended)
# =========================
# ... existing globals ...
demo_pause_between = 3  # Pause between demos (configurable)
_initialized = False

# =========================
# Demo function
# =========================
def run_demo(csv_path="1000.csv", json_dir="data_action"):
    """
    Plays all actions from data_action using 1000.csv.
    """
    ensure_initialized()

    commands = load_commands_csv(csv_path)
    if not commands:
        print("[INFO] Nothing to play. No commands found in 1000.csv.")
        return

    print(f"[INFO] Starting demo: {len(commands)} commands")
    for row in commands:
        cid = row["id"]
        cmd = row["command"]
        actions = load_actions_json(json_dir, cid)
        if actions is None:
            print(f"[SKIP] id={cid} — no JSON found, command='{cmd}'")
            continue

        print("=" * 80)
        print(f"id: {cid}")
        print(f"command: {cmd}")

        try:
            move_joints_by_name(actions)
            print(f"[OK] executed id={cid}\n")
        except Exception as e:
            print(f"[ERR] execution failed for id={cid}: {e}\n")

        if demo_pause_between > 0:
            time.sleep(demo_pause_between)

    print("[DONE] Demo finished.")


# Linting-time placeholders so mypy/linters don’t complain at import time
m = d = policy = None
window = cam = opt = scene = context = None
joint_map = {}
joint_index_map = {}
INIT_QPOS = None
simulation_dt = control_decimation = None
kps = kds = default_angles = None
ang_vel_scale = dof_pos_scale = dof_vel_scale = action_scale = None
cmd_scale = None
num_actions = num_obs = None
movement = {}

# =========================
# Saving helpers
# =========================

def _ensure_list_of_dicts(actions: Any) -> list[dict]:
    """Normalize actions to a list of objects."""
    if actions is None:
        return []
    if isinstance(actions, dict):
        return [actions]
    if isinstance(actions, list):
        # accept list of lists (old frame format); still save as-is
        return actions
    raise ValueError("Actions must be list or dict")


def save_actions_json(
    actions: Any,
    command_id: str | None = None,
    command_text: str | None = None,
    out_dir: str | Path = "captured",
    filename: str | None = None
) -> Path:
    """
    Save an array of actions to JSON in the captured/ directory.
    Additionally, append a record to data_captured.csv (id, command, json_file).
    """
    actions_norm = _ensure_list_of_dicts(actions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"capture_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    p = out_dir / filename

    with p.open("w", encoding="utf-8") as f:
        json.dump(actions_norm, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {p}")

    # write to csv (id, command, json_file)
    log_capture_to_csv(command_id, command_text, filename)

    return p


def log_capture_to_csv(
    command_id: str | None,
    command_text: str | None,
    json_filename: str,
    csv_path: str | Path = "data_captured.csv"
) -> None:
    """
    Append a record about the saved json to data_captured.csv.
    Format: id,command,json_file
    """
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "command", "json_file"])
        writer.writerow([
            command_id or "",
            command_text or "",
            json_filename
        ])


def save_actions_json_by_id(actions: Any, command_id: str, out_dir: str | Path = "data_action") -> Path:
    """
    Save an array of actions to file data_action/<id>.json (overwriting existing file).
    """
    actions_norm = _ensure_list_of_dicts(actions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{command_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(actions_norm, f, ensure_ascii=False, indent=2)
    print(f"[SAVE-ID] {p}")
    return p


def generate_from_csv_and_play(
    csv_path: str | Path = "1000.csv",
    out_dir: str | Path = "data_action",
    start_index: int = 0,
    max_count: int | None = None,
    interval_sec: float = 0.0,
    preview_only: bool = False,
) -> None:
    """
    Read a CSV (columns: id, command), for each row:
      - generate actions via parse_command(command),
      - play them on the robot (move_joints_by_name),
      - save to data_action/<id>.json (if preview_only=False).

    Params:
      start_index  — which row to start from (0-based)
      max_count    — how many rows to process (None = until end)
      interval_sec — pause between examples (may be 0)
      preview_only — if True, show but do NOT save by id
    """
    ensure_initialized()

    p = Path(csv_path)
    if not p.exists():
        print(f"[ERR] CSV not found: {p.resolve()}")
        return

    rows: list[dict] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[ERR] CSV has no header")
            return
        fmap = {k.lower().strip(): k for k in reader.fieldnames}
        id_col = fmap.get("id")
        cmd_col = fmap.get("command") or fmap.get("cmd")
        if not id_col or not cmd_col:
            print("[ERR] CSV must contain columns: id, command")
            return
        for row in reader:
            cid = str(row.get(id_col, "")).strip()
            cmd = str(row.get(cmd_col, "")).strip()
            if cid and cmd:
                rows.append({"id": cid, "command": cmd})

    if start_index < 0 or start_index >= len(rows):
        print(f"[INFO] start_index out of range (0..{len(rows)-1})")
        return

    end_index = len(rows) if max_count is None else min(len(rows), start_index + max_count)

    print(f"[INFO] Generating {end_index - start_index} samples from {p.resolve()}")
    for i in range(start_index, end_index):
        cid = rows[i]["id"]
        cmd_text = rows[i]["command"]
        print("=" * 80)
        print(f"id: {cid}")
        print(f"command: {cmd_text}")
        print("Thinking...")
        try:
            actions = parse_command(cmd_text)  # LLM → list of action objects
        except Exception as e:
            print(f"[ERR] parse failed for id={cid}: {e}")
            continue

        try:
            # Play on the robot
            move_joints_by_name(actions)
        except Exception as e:
            print(f"[ERR] play failed for id={cid}: {e}")
            # even if playing failed, save anyway (if desired)
            if not preview_only:
                save_actions_json_by_id(actions, cid, out_dir=out_dir)
            continue

        # Save <id>.json
        if not preview_only:
            save_actions_json_by_id(actions, cid, out_dir=out_dir)
            log_capture_to_csv(cid, cmd_text, p.name)
        # Pause between examples
        if interval_sec > 0:
            time.sleep(interval_sec)

    print("[DONE] CSV generation completed.")

def _deg(x: float) -> float:
    return float(np.degrees(x))

def _axis_letter(ax: np.ndarray) -> str:
    if np.linalg.norm(ax) < 1e-8:
        return "-"
    return "XYZ"[int(np.argmax(np.abs(ax)))]

def _format_joint_range(j: int) -> str:
    jtype = m.jnt_type[j]
    limited = bool(m.jnt_limited[j])
    rng = m.jnt_range[j]
    if jtype == mj.mjtJoint.mjJNT_HINGE:
        return f"{np.degrees(rng[0]):7.2f}..{np.degrees(rng[1]):7.2f} deg" if limited else "unlimited"
    if jtype == mj.mjtJoint.mjJNT_SLIDE:
        return f"{rng[0]:7.3f}..{rng[1]:7.3f} m" if limited else "unlimited"
    if jtype == mj.mjtJoint.mjJNT_BALL:
        return "quat (no range)"
    return "free (6-DoF)"

def _current_angle_str(j: int) -> str:
    jtype = m.jnt_type[j]
    adr = m.jnt_qposadr[j]
    if jtype == mj.mjtJoint.mjJNT_HINGE:
        return f"{_deg(d.qpos[adr]):.2f} deg"
    if jtype == mj.mjtJoint.mjJNT_SLIDE:
        return f"{d.qpos[adr]:.4f} m"
    return "-"  # BALL/FREE

def build_current_pose_tables() -> str:
    # мягкая проверка готовности модели/данных
    if m is None or d is None:
        return "[WARN] current pose unavailable: model/data not initialized"

    lines = []
    lines.append("Joints overview (current):")
    lines.append(f"{'id':>3} {'name':<28} {'range':>21} {'axis':>4} {'curr':>10}")
    lines.append("-" * 70)
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        rng = _format_joint_range(j)
        jtype = m.jnt_type[j]
        ax = _axis_letter(m.jnt_axis[j]) if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE) else "-"
        curr = _current_angle_str(j)
        lines.append(f"{j:3d} {name:<28} {rng:>21} {ax:>4} {curr:>10}")

    upper = [
        ("left_shoulder_pitch",  "Y", "-164.4 to +164.4",  " "),
        ("left_shoulder_roll",   "X", "-19.5 to +178.2",   "Outward(+)/inward(-)"),
        ("left_shoulder_yaw",    "Z", "-74.5 to +254.9",   "Arm rotation"),
        ("left_elbow",           "Y", "-71.6 to +149.6",   "0°=90° bend, +90°=straight"),
        ("right_shoulder_pitch", "Y", "-164.4 to +164.4",  " "),
        ("right_shoulder_roll",  "X", "-178.2 to +19.5",   "Outward(-)/inward(+)"),
        ("right_shoulder_yaw",   "Z", "-254.9 to +74.5",   "Arm rotation"),
        ("right_elbow",          "Y", "-71.6 to +149.6",   "0°=90° bend, +90°=straight"),
    ]
    md = []
    md.append("")
    md.append("| Joint Name | Axis | Range (deg) | Current (deg) | Description |")
    md.append("| ---------- | ---- | ----------- | ------------- | ----------- |")
    for n, ax, rng, desc in upper:
        if joint_map and n in joint_map:
            curr = _deg(d.qpos[joint_map[n]])
            md.append(f"| {n} | {ax} | {rng} | {curr:.2f} | {desc} |")
        else:
            md.append(f"| {n} | {ax} | {rng} | - | {desc} |")

    return "\n".join(lines + md)


def _append_current_pose_tables_to_prompt(txt: str) -> str:
 
    try:
        block = build_current_pose_tables()
        print(block)
        return (txt or "") + "\n\n" + block + "\n"
    except Exception as e:
        print(f"[WARN] can't build current pose tables: {e}")
        return txt
# --- GAIT (policy) actuator mapping ---
GAIT_ACT_IDS = []

def build_gait_act_ids():
    na = num_actions
    # какие суставы читает политика из qpos[7:7+na] (в нужном порядке)
    gait_joint_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)
                        for j in range(m.njnt)
                        if m.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)
                        and 7 <= m.jnt_qposadr[j] < 7 + na]
    # joint -> actuator
    j2a = {}
    trn  = np.array(m.actuator_trnid).reshape(m.nu, 2)
    trnt = np.array(m.actuator_trntype)
    for i in range(m.nu):
        if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
            jid = int(trn[i, 0])
            jname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, jid)
            j2a[jname] = i
    return [j2a[n] for n in gait_joint_names]


def ensure_initialized():
    global _initialized, m, d, policy, INIT_QPOS
    global simulation_dt, control_decimation, kps, kds, default_angles
    global ang_vel_scale, dof_pos_scale, dof_vel_scale, action_scale, cmd_scale
    global num_actions, num_obs
    global joint_map, joint_index_map, ALLOWED_JOINTS
    global window, cam, opt, scene, context

    if _initialized:
        return

    # ---- file paths (as in __main__) ----
   
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    #xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene.xml"
    xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene.xml"

    # ---- load config ----
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    # ---- MuJoCo model ----
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt

    m.vis.quality.shadowclip = 10.0  

    

    opt = mj.MjvOption()
    opt.flags[mj.mjtVisFlag.mjVIS_SHADOW] = True

    # ---- allowed joints map ----
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)

    # ---- IMPORTANT: joint_map first, then joint_index_map ----
    joint_map = {}
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]

    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]


    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    # ---- reset and store initial pose ----
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    # ---- policy ----
    policy = torch.jit.load(policy_path)

    # ---- use the already created viewer window ----
    if window is None:
        raise RuntimeError("GLFW window is not initialized. Run from main viewer first.")
    # ensure context is active (just in case)
    try:
        cur_ctx = glfw.get_current_context()
    except Exception:
        cur_ctx = None
    if not cur_ctx:
        glfw.make_context_current(window)

    # cam/opt/scene/context are created in __main__; if missing — create here
    if cam is None:
        cam = mj.MjvCamera()
        cam.azimuth = 180
        cam.elevation = -15
        cam.distance = 4.0
        cam.lookat = np.array([0.0, 0.0, 0.8])
    if opt is None:
        opt = mj.MjvOption()
    if scene is None:
        scene = mj.MjvScene(m, maxgeom=10000)
    if context is None:
        context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    _initialized = True
    print("[whole_body] initialized for dataset player (using main viewer window)")


# =========================
# Global constants & aliases
# =========================

def build_allowed_joints_from_model(m: mj.MjModel) -> set[str]:
    """
    Collect allowed joint names from the model.
    We include HINGE and SLIDE joints and exclude FREE/BALL joints.
    """
    allowed = set()
    for j in range(m.njnt):
        jtype = m.jnt_type[j]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            allowed.add(name)
    return allowed


def build_joint_index_map(joint_map: Dict[str, int],
                          preferred_order: List[str] | None = None) -> Dict[int, str]:
    """
    Build the frame joint order from the model.
    By default, we keep the canonical 8 arm joints in this order if present.
    """
    if preferred_order is None:
        preferred_order = [
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        ]

    names = [n for n in preferred_order if n in joint_map]
    missing = [n for n in preferred_order if n not in joint_map]
    if missing:
        print("Warning: missing joints for frame control:", ", ".join(missing))

    return {i: n for i, n in enumerate(names)}


# Simple aliases the LLM might produce
JOINT_ALIASES = {
    "right_arm": "right_shoulder_pitch",
    "left_arm": "left_shoulder_pitch",
    "right_hand": "right_elbow",
    "left_hand": "left_elbow",
}

# Cache for parsed commands
command_cache: Dict[str, Any] = {}

# Initialize LLM once
LLM = build_llm()


# =========================
# LLM helpers
# =========================

def _llm_chat(messages: List[Dict[str, str]], max_tokens: int | None = None, temperature: float | None = None) -> str:
    """Single entry point into the LLM with default settings from `settings`."""
    return LLM.chat(
        messages,
        # max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
    )


def _vlm_chat(user_text: str, image_path: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
    return LLM.chat_vision(
        user_text,
        image_path,
        # max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
    )


def open_cmd_with_image_window():
    root = tk.Tk()
    root.title("Command + Image")
    root.geometry("640x240")

    state = {"path": None}

    frm = tk.Frame(root)
    frm.pack(fill="both", expand=True, padx=10, pady=10)

    tk.Label(frm, text="Natural-language command:").pack(anchor="w")
    txt = tk.Text(frm, width=80, height=6)
    txt.pack(fill="both", expand=True, pady=(0, 8))

    path_lbl = tk.Label(frm, text="No image selected")
    path_lbl.pack(anchor="w")

    def choose_file():
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if path:
            state["path"] = path
            path_lbl.config(text=f"Image: {path}")

    btns = tk.Frame(frm)
    btns.pack(fill="x")
    tk.Button(btns, text="Choose image…", command=choose_file).pack(side="left")

    result = {"text": None, "image_path": None}

    def apply_and_close():
        t = txt.get("1.0", "end").strip()
        if not t or not state["path"]:
            path_lbl.config(text="Need both: command text AND an image file.")
            return
        result["text"] = t
        result["image_path"] = state["path"]
        root.destroy()

    def cancel():
        root.destroy()

    tk.Button(btns, text="Apply", command=apply_and_close).pack(side="right", padx=6)
    tk.Button(btns, text="Cancel", command=cancel).pack(side="right")

    root.mainloop()
    return result if result["text"] and result["image_path"] else None


def _read_system_prompt() -> str:
    """Load the base system prompt from settings.PROMPT_PATH."""
    with open(settings.PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# JSON cleaning & parsing
# =========================

def _clean_json_block(s: str) -> str:
    """
    Remove provider markers, code fences, and single-line comments; return raw JSON-ish text.
    Does not parse — only cleans.
    """
    t = s.strip()

    # 1) Remove provider-style markers like <|start|>assistant<|...|>
    t = re.sub(r"<\|.*?\|>", "", t)

    # 2) Remove Markdown code fences: ```json ... ``` or ```
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.DOTALL)

    # 3) Remove single-line comments starting with // ...
    t = re.sub(r"//.*?(?=\n|$)", "", t)

    return t.strip()


def _extract_first_json(s: str) -> str:
    """
    Try to extract the first valid JSON object/array from text.
    Returns a JSON string or raises ValueError.
    """
    txt = _clean_json_block(s)

    # Quick path: the whole string is JSON
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # Otherwise, scan for {...} or [...] candidates and test them
    candidates = re.findall(r"(\{.*\}|\[.*\])", txt, flags=re.DOTALL)
    for cand in candidates:
        cand_stripped = cand.strip()
        try:
            json.loads(cand_stripped)
            return cand_stripped
        except Exception:
            continue

    # Last resort: literal_eval for cases with single quotes
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, (list, dict)):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    raise ValueError(f"Could not extract JSON from: {s}")


# =========================
# Natural-language command parsing
# =========================

def parse_command(command: str, rag_k: int = 5, rag_csv: str = "1000.csv", rag_json: str = "data_action"):
    """
    Convert a natural-language user command into a normalized list of JSON actions.
    New features:
      • Inject few-shot from top-5 similar examples (cosine similarity) with gold JSON answers.
      • Carefully extract the first valid JSON from the model's response.
      • Soft normalization of action fields (type, numeric values, joint names).

    Environment requirements (optional — safe fallbacks are used if absent):
      _read_system_prompt() -> str
      _llm_chat(messages: list[dict], max_tokens: int, temperature: float) -> str
      _extract_first_json(text: str) -> str
      settings.LLM_TEMPERATURE : float
      ALLOWED_JOINTS : set[str]
      JOINT_ALIASES : dict[str, str]
      command_cache : dict[str, Any]
      _load_top_sim_examples(user_cmd, csv_path, json_dir, k, model_name) -> list[tuple[str, list|dict]]
      DEFAULT_MODEL : str
    """
    import json
    from pathlib import Path

    # ---------- safe defaults/globals ----------
    try:
        temp = float(getattr(settings, "LLM_TEMPERATURE", 0.2))
    except Exception:
        temp = 0.2

    # Command cache (if not defined above)
    global command_cache
    if "command_cache" not in globals() or not isinstance(globals().get("command_cache"), dict):
        command_cache = {}

    # Normative lists (use module-level ones if available)
    ALLOWED = set(globals().get("ALLOWED_JOINTS", []))
    ALIASES = dict(globals().get("JOINT_ALIASES", {}))

    def _read_prompt_safe() -> str:
        try:
            return str(_read_system_prompt()).strip()
        except Exception:
            # minimal system prompt by default
            return (
                "You are a strict JSON planner. "
                "Given a human 'Command', respond with JSON ONLY: a list of actions, "
                "each action being an object. No prose. Use unicode, keep keys in snake_case."
            )

    def _few_shots(command_text: str, k: int = 5, csv_path="1000.csv", json_dir="data_action") -> list[tuple[str, list]]:
        """
        Get pairs (example_command, gold_JSON_as_list).
        Safely returns [] if something is missing.
        """
        try:
            examples = _load_top_sim_examples(
                user_cmd=command_text,
                csv_path=csv_path,
                json_dir=json_dir,
                k=k,
                model_name=globals().get("DEFAULT_MODEL", None),
            )
        except Exception:
            examples = []

        # normalize JSON to list
        out: list[tuple[str, list]] = []
        for ex_cmd, ex_json in examples:
            if isinstance(ex_json, dict):
                ex_json = [ex_json]
            if isinstance(ex_json, list):
                out.append((str(ex_cmd), ex_json))
        return out

    def _normalize_joint_name(name: str) -> str:
        """Normalize joint name via aliases and validate if lists are provided."""
        j = str(name).strip()
        if ALLOWED:
            if j not in ALLOWED and j in ALIASES:
                j = ALIASES[j]
            if j not in ALLOWED:
                # if a control list is present and the name is unknown — raise
                raise ValueError(f"Unknown joint name '{name}'. Allowed: {sorted(ALLOWED)}")
        return j

    def _normalize_action(action: dict) -> dict:
        """
        Soft-normalize a single action:
          • keys → snake_case/lower
          • strings — strip whitespace
          • numbers — cast to float/int where appropriate
          • fields 'joint'/'joints' validated via ALLOWED/ALIASES (if provided)
        """
        if not isinstance(action, dict):
            raise ValueError("Each action must be an object.")

        def _snake(s: str) -> str:
            return str(s).strip().replace(" ", "_").replace("-", "_").lower()

        # 1) keys to snake_case + copy
        norm = {}
        for k, v in action.items():
            norm[_snake(k)] = v

        # 2) action type (help model keep type/action)
        if "type" not in norm and "action" in norm:
            norm["type"] = norm.pop("action")

        # 3) string fields — trim
        for k, v in list(norm.items()):
            if isinstance(v, str):
                norm[k] = v.strip()

        # 4) numeric conversions (softly)
        def _maybe_num(x):
            try:
                if isinstance(x, bool):
                    return x
                if isinstance(x, (int, float)):
                    return x
                if isinstance(x, str) and x:
                    # int if no dot; else float
                    if x.isdigit() or (x[0] in "+-" and x[1:].isdigit()):
                        return int(x)
                    return float(x)
            except Exception:
                pass
            return x

        for k, v in list(norm.items()):
            if k in {"angle", "speed", "duration", "x", "y", "z", "roll", "pitch", "yaw", "value"}:
                norm[k] = _maybe_num(v)

        # 5) joint normalization
        if "joint" in norm:
            norm["joint"] = _normalize_joint_name(norm["joint"])
        if "joints" in norm and isinstance(norm["joints"], list):
            norm["joints"] = [_normalize_joint_name(j) for j in norm["joints"]]

        return norm

    # ---------- quick cache ----------
    if command in command_cache:
        return command_cache[command]

    # ---------- build messages for the LLM ----------
  
    base_prompt = _read_prompt_safe()
    base_prompt = _append_current_pose_tables_to_prompt(base_prompt)  # <<<
    messages = [{"role": "system", "content": base_prompt}]


    few_shots = _few_shots(command, k=rag_k, csv_path=rag_csv, json_dir=rag_json)
    show_examples = True

    if show_examples:
        print("by commands examples:")
        if few_shots:
            for i, (ex_cmd, _) in enumerate(few_shots, 1):
                print(f"{i}) {ex_cmd}")
        else:
            # fallback for printing only
            try:
                df = load_cmds_csv(args.csv)
                ranked = compute_similarities(command, csv_path=args.csv, json_dir=args.json_dir)
                for i, (_, row) in enumerate(ranked.head(5).iterrows(), 1):
                    print(f"{i}) {row['command']}  [no json]")
            except Exception as e:
                print(f"  [none] ({e})")

    if few_shots:
        messages.append({
            "role": "system",
            "content": ("""
Below are guidelines: pairs of “command example → target JSON”. Strictly follow the response format from the examples. Reply with JSON ONLY."""
                        ),
        })
        for ex_cmd, ex_json in few_shots:
            messages.append({"role": "user", "content": f'Command: "{ex_cmd}". Output JSON only.'})
            try:
                messages.append({"role": "assistant", "content": json.dumps(ex_json, ensure_ascii=False)})
            except Exception:
                # if serialization fails — skip that particular example
                continue

    # final user request
    messages.append({"role": "user", "content": f'Command: "{command}". Output JSON only.'})

    # ---------- model call ----------
    try:
        print("\n=== PROMPT DUMP ===")
        for i, m in enumerate(messages, 1):
            role = m.get("role")
            content = m.get("content")
            print(f"[{i}] {role}:\n{content}\n")
        print("=== END PROMPT ===\n")
        raw = _llm_chat(messages, max_tokens=2048, temperature=temp)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    # ---------- extract JSON ----------
    try:
        json_text = _extract_first_json(raw)
    except Exception:
        # if no special extractor — naive attempt to find the first JSON block
        import re
        m = re.search(r"(\[.*\]|\{.*\})", raw, flags=re.S)
        json_text = m.group(1) if m else raw.strip()

    try:
        parsed = json.loads(json_text)
    except Exception:
        # developer diagnostics
        print("[parse_command] Raw LLM response:\n", raw)
        raise

    # ensure list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array of actions or a single JSON object.")
    print(parsed)

    # ---------- normalize the set of actions ----------
    normalized: list[dict] = []
    for item in parsed:
        norm = _normalize_action(item)
        normalized.append(norm)

    # ---------- cache and return ----------
    command_cache[command] = normalized
    return normalized


def parse_command_with_image(command: str, image_path: str):
    """
    Same as parse_command(), but with VLM.
    Supports 'frame', single joints and 'walk'.
    """
    cache_key = f"{command}||{image_path}"
    if cache_key in command_cache:
        return command_cache[cache_key]

    # parse_command_with_image()
    base_prompt = _read_system_prompt().strip()
    base_prompt = _append_current_pose_tables_to_prompt(base_prompt)  # <<<
    user_text = f"{base_prompt}Command: \"{command}\""


    raw = _vlm_chat(user_text, image_path)
    json_text = _extract_first_json(raw)
    try:
        parsed = json.loads(json_text)
    except Exception:
        print("[EXCEPTION CAUGHT] Original response:", raw)
        raise

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array with actions.")

    normalized: list[dict] = []

    def _normalize_joint_name(jname: str) -> str:
        jname = str(jname).strip()
        if jname not in ALLOWED_JOINTS and jname in JOINT_ALIASES:
            jname = JOINT_ALIASES[jname]
        if jname not in ALLOWED_JOINTS:
            raise ValueError(f"Unknown joint name '{jname}'. Allowed: {sorted(ALLOWED_JOINTS)}")
        return jname

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Action[{i}] must be an object, got {type(item)}")

        if "frame" in item:
            frame = item["frame"]
            if not isinstance(frame, list) or not frame:
                raise ValueError(f"Action[{i}].frame must be a non-empty list")
            norm_frame: list[dict] = []
            for j, sub in enumerate(frame):
                if not isinstance(sub, dict):
                    raise ValueError(f"Action[{i}].frame[{j}] must be an object")
                if "name" not in sub:
                    raise ValueError(f"Action[{i}].frame[{j}] missing 'name'")
                if "angle" not in sub:
                    raise ValueError(f"Action[{i}].frame[{j}] missing 'angle'")
                jname = _normalize_joint_name(sub["name"])
                angle = float(sub["angle"])
                norm_frame.append({"name": jname, "angle": angle})
            duration = float(item.get("duration", 0.3))
            normalized.append({"frame": norm_frame, "duration": duration})
            continue

        if item.get("name") == "walk":
            required_walk = {"num", "dir_deg", "spd"}
            missing = required_walk - set(item.keys())
            if missing:
                raise ValueError(f"'walk' action must include {required_walk}. Missing: {missing}. Got: {item}")
            normalized.append({
                "name": "walk",
                "num": int(item["num"]),
                "dir_deg": float(item["dir_deg"]),
                "spd": float(item["spd"]),
            })
            continue

        if "name" not in item:
            raise ValueError(f"Action at index {i} missing 'name'.")
        if "angle" not in item:
            raise ValueError(f"Action at index {i} missing 'angle'.")

        jname = _normalize_joint_name(item["name"])
        angle = float(item["angle"])
        duration = float(item.get("duration", 0.3))
        normalized.append({"name": jname, "angle": angle, "duration": duration})

    command_cache[cache_key] = normalized
    return normalized

# === STANCE (держим стойку, RL-ходьбу замораживаем) ===
# === STANCE: шаги на месте (RL включён) ===
STANCE_HOLD = False
STANCE_V = 0.12                  # целевая амплитуда по главной оси
STANCE_AXIS = "x"                # "x" или "y"
STANCE_STEPS_EACH = 1
_stance_dir = +1
_stance_local = 0

# Главная ось (ритм ±V + PI на ошибку позиции)
STANCE_KP = 1.0                  # м/(с·м)
STANCE_KD = 0.3                  # м/с на м/с
STANCE_KI = 0.2                  # м/(с·м·с) — интегральная убирает остаточный дрейф
STANCE_I_CLAMP = 0.3             # ограничение интегратора (в "эквиваленте" м)

# Поперечная ось (без ритма, только подавление дрейфа)
STANCE_ORTH_KP = 1.2
STANCE_ORTH_KD = 0.25

# Удержание курса (yaw → 0 относительно старта стойки)
STANCE_YAW_KP = 2.0              # рад/с на рад
STANCE_YAW_KD = 0.2              # рад/с на рад/с

STANCE_HOME = None               # np.array([x0, y0])
STANCE_YAW_HOME = None           # float (рад)

# внутренние состояния контроллера
STANCE_I_MAIN = 0.0              # интегральная сумма по главной оси

def start_stance_hold(v: float = 0.12, axis: str = "x", steps_each: int = 1):
    """Включить стойку: ритм ±V + PI по главной оси, подавление дрейфа по поперечной оси, удержание yaw."""
    global STANCE_HOLD, STANCE_V, STANCE_AXIS, STANCE_STEPS_EACH
    global _stance_dir, _stance_local, steps_needed, steps_done
    global STANCE_HOME, STANCE_YAW_HOME, STANCE_I_MAIN

    STANCE_HOLD = True
    STANCE_V = float(v)
    STANCE_AXIS = axis.lower()
    STANCE_STEPS_EACH = int(steps_each)
    _stance_dir = +1
    _stance_local = 0
    steps_needed = 0
    steps_done = 0

    # Запоминаем «дом»
    STANCE_HOME = np.array([float(d.qpos[0]), float(d.qpos[1])], dtype=np.float32)

    # Запоминаем курс в старте стойки
    qw, qx, qy, qz = d.qpos[3:7]
    STANCE_YAW_HOME = _quat_yaw(qw, qx, qy, qz)

    # Сбрасываем интегратор
    STANCE_I_MAIN = 0.0

    # Стартовая команда — нулевая: контроллер задаст в цикле
    cmd[:] = 0.0

def stop_stance_hold():
    """Выключить стойку на месте."""
    global STANCE_HOLD
    STANCE_HOLD = False
    cmd[:] = 0.0



# =========================
# Small GUI helper
# =========================

def open_paste_window():
    """Open a modal window, return parsed list (from JSON or Python literal) or None."""
    root = tk.Tk()
    root.title("Paste motion array")

    txt = tk.Text(root, width=80, height=18)
    txt.pack(fill="both", expand=True, padx=10, pady=10)

    result = {"data": None}

    def apply_and_close():
        raw = txt.get("1.0", "end").strip()
        if not raw:
            root.destroy()
            return
        # Try JSON first, then Python literal
        try:
            data = json.loads(raw)
        except Exception:
            try:
                data = ast.literal_eval(raw)
            except Exception as e:
                print("Parse error:", e)
                return  # keep the window open for user to fix
        if isinstance(data, dict):
            data = [data]  # single object -> list
        result["data"] = data
        root.destroy()

    def cancel():
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=10, pady=(0, 10))
    tk.Button(btns, text="Apply", command=apply_and_close).pack(side="right", padx=4)
    tk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=4)

    # Blocking modal run
    root.mainloop()
    return result["data"]

def _try_vlm_multi(prompt_text: str, image_paths: list[Path]):

    try:
        func = globals().get("_vlm_chat_multi", None)
        if callable(func):
            return func(prompt_text, [str(p) for p in image_paths])
    except Exception as e:
        print(f"[INFO] _vlm_chat_multi does not work: {e}")
    return None


def make_mosaic(image_paths: list[Path], cols: int = 3, pad: int = 8, out_path: str = "captures/views_mosaic.png") -> Path:
   
    assert image_paths, "at least one picture"
    imgs = [imageio.imread(str(p)) for p in image_paths]
    h, w = imgs[0].shape[:2]

    n = len(imgs)
    rows = math.ceil(n / cols)
    H = rows * h + (rows - 1) * pad
    W = cols * w + (cols - 1) * pad

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y+h, x:x+w, :3] = img[:, :, :3]

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(outp, canvas)
    print(f"[MOSAIC] saved -> {outp}")
    return outp


def vlm_analyze_views_single_call(prompt_text: str, image_paths: list[Path], out_json: str = "captures/vlm_batch.json", mosaic_cols: int = 3) -> dict:

    # try 1: naive multi-input
    resp = _try_vlm_multi(prompt_text, image_paths)
    meta = {"mode": "multi_api", "images": [str(p) for p in image_paths]}
    if resp is None:
        # try 2: fallback by mosaic
        mosaic_path = make_mosaic(image_paths, cols=mosaic_cols, pad=8, out_path="captures/views_mosaic.png")
        try:
            resp = _vlm_chat(prompt_text, str(mosaic_path))
            meta = {"mode": "mosaic", "mosaic_path": str(mosaic_path), "images": [str(p) for p in image_paths]}
        except Exception as e:
            resp = f"[VLM error: {e}]"
            meta["error"] = str(e)

    result = {"prompt": prompt_text, "response": resp, **meta}
    try:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] batch VLM -> {outp}")
    except Exception as e:
        print(f"[WARN] can't save batch JSON: {e}")
    return result

# =========================
# Multi-view capture helpers
# =========================

def _render_and_read_pixels(width: int, height: int) -> np.ndarray:

    global m, d, opt, cam, scene, context
    viewport = mj.MjrRect(0, 0, width, height)
   
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    depth = np.empty((height, width), dtype=np.float32)
    mj.mjr_readPixels(rgb, depth, viewport, context)

    return np.flipud(rgb)


def capture_views(
    out_dir: str = "captures",
    prefix: str = "view",
    num: int = 6,
    azimuth_span_deg: float = 360.0,
    elevation_deg: float | None = None,
    distance: float | None = None,
    size: tuple[int, int] | None = None,
) -> list[Path]:

    global window, cam

 
    if window is None:
        raise RuntimeError("GLFW window is not initialized")
    try:
        cur_ctx = glfw.get_current_context()
        if not cur_ctx:
            glfw.make_context_current(window)
    except Exception:
        glfw.make_context_current(window)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    cam_backup = mj.MjvCamera()
    cam_backup.azimuth = cam.azimuth
    cam_backup.elevation = cam.elevation
    cam_backup.distance = cam.distance
    cam_backup.lookat = cam.lookat.copy()

    if size is None:
        fb_w, fb_h = glfw.get_framebuffer_size(window)
    else:
        fb_w, fb_h = int(size[0]), int(size[1])

    if elevation_deg is not None:
        cam.elevation = float(elevation_deg)
    if distance is not None:
        cam.distance = float(distance)

    start_az = float(cam.azimuth)
    step = float(azimuth_span_deg) / max(1, num)

    saved: list[Path] = []
    try:
        for i in range(num):
            cam.azimuth = start_az + i * step
            rgb = _render_and_read_pixels(fb_w, fb_h)
            path = outp / f"{prefix}_{i:03d}.png"
            imageio.imwrite(path, rgb)
            print(f"[CAPTURE] {path}")
            saved.append(path)

            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        cam.azimuth = cam_backup.azimuth
        cam.elevation = cam_backup.elevation
        cam.distance = cam_backup.distance
        cam.lookat = cam_backup.lookat

    return saved


def capture_and_query_vlm_batch():

    try:
        user_extra = input("What's object interaction?").strip()

        base_prompt = ""
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                base_prompt = f.read().strip()
        except Exception as e:
            print(f"[WARN] prompt.txt not found: {e}")
        base_prompt = _append_current_pose_tables_to_prompt(base_prompt)
        prompt_text = (base_prompt + ("\n\n" + user_extra if user_extra else "")).strip()
        n = input("How much views [6]? ").strip()
        n = int(n) if n else 6
        out_dir = input("Path to save [captures]: ").strip() or "captures"
        prefix = input("Prefics [view]: ").strip() or "view"
        elev = input("Fix elevation (deg) [Enter - current]: ").strip()
        elev = float(elev) if elev else None
        dist = input("FIx distance [Enter — current]: ").strip()
        dist = float(dist) if dist else None
        cols = input("How much rows [3]? ").strip()
        cols = int(cols) if cols else 3
    except Exception as e:
        print(f"Bad input: {e}")
        return

    paths = capture_views(
        out_dir=out_dir,
        prefix=prefix,
        num=n,
        azimuth_span_deg=360.0,
        elevation_deg=elev,
        distance=dist,
        size=None,
    )
    if not paths:
        print("No frames to VLM.")
        return

    vlm_analyze_views_single_call(
        prompt_text, 
        paths, 
        out_json=str(Path(out_dir) / "vlm_batch.json"), 
        mosaic_cols=cols
    )
def _build_prompt_with_extra(user_extra: str) -> str:
    
    base_prompt = ""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except Exception as e:
        print(f"[WARN] prompt.txt not found: {e}")
   
    base_prompt = _append_current_pose_tables_to_prompt(base_prompt)
    return (base_prompt + ("\n\n" + user_extra if user_extra else "")).strip()

def _parse_actions_from_text(raw_text: str) -> list[dict]:
   
    js = _extract_first_json(raw_text)
    data = json.loads(js)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("At least 1 element is expected")
    return data


def capture_and_execute_vlm_batch(default_views: int = 6, default_cols: int = 3) -> None:
 

    user_extra = input("Object action to prompt.txt (emp): ").strip()
    prompt_text = _build_prompt_with_extra(user_extra)


    try:
        n_raw = input(f"How much views [{default_views}]? ").strip()
        n = int(n_raw) if n_raw else default_views
    except Exception:
        n = default_views

    try:
        cols_raw = input(f"Columns in mosaic [{default_cols}]? ").strip()
        cols = int(cols_raw) if cols_raw else default_cols
    except Exception:
        cols = default_cols

    out_dir = f"robot_captures/vlm_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    prefix = "view"
    elev = None
    dist = None

 
    paths = capture_views(
        out_dir=out_dir,
        prefix=prefix,
        num=n,
        azimuth_span_deg=360.0,
        elevation_deg=elev,
        distance=dist,
        size=None,
    )
    if not paths:
        print("[VLM] No frames.")
        return
    
    print("Thinking...") 

   
    result = vlm_analyze_views_single_call(
        prompt_text,
        paths,
        out_json=str(Path(out_dir) / "vlm_batch.json"),
        mosaic_cols=cols
    )
    resp_text = result.get("response", "")

   
    try:
        actions = _parse_actions_from_text(resp_text)
    except Exception as e:
        print(f"[ERR] Cant json read from VLM: {e}")
        print("=== RAW RESPONSE START ===")
        print(resp_text)
        print("=== RAW RESPONSE END ===")
        return

    print("[ACTIONS]", actions)
   
    move_joints_by_name(actions)

  
    # ans = input("Save to JSON? (y/N): ").strip().lower()
    # if ans in ("y", "yes"):
    #    
    #     save_actions_json(actions, command_text=f"[VLM batch] extra='{user_extra}'", out_dir="data_action", filename=None)

# =========================
# Physics / control helpers
# =========================

def get_gravity_orientation(quaternion):
    """Compute an approximate gravity orientation vector from a quaternion."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

# === GLOBAL ABORT ===
RUN_ABORT = False
def clear_abort():
    """Сбрасываем аварийный флаг перед СТАРТОМ новых движений."""
    global RUN_ABORT
    RUN_ABORT = False

def _should_abort() -> bool:
    # общий быстрый чек во всех циклах
    return RUN_ABORT or (window is not None and glfw.window_should_close(window))

def abort_all_motion():
    """Запросить аварийную остановку всего, что сейчас крутится."""
    global RUN_ABORT, steps_needed, steps_done, cmd, target_dof_pos, action, short_extra_done, last_activity_time
    RUN_ABORT = True
    # обнуляем всё управление походкой/руками
    steps_needed = 0
    steps_done = 0
    cmd[:] = 0.0
    action[:] = 0.0
    if 'default_angles' in globals() and default_angles is not None:
        target_dof_pos = default_angles.copy()
    if ARM_TARGETS is not None and len(ARM_TARGETS) > 0:
        ARM_TARGETS[:] = 0.0
    # немедленно глушим актуаторы
    d.ctrl[:] = 0.0
    d.qacc[:] = 0.0
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    # чтобы не стартовал автобаланс
    short_extra_done = True
    last_activity_time = time.time()


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control law."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def hard_reset(reload_model: bool = False, reload_policy: bool = True):
    global m, d, scene, context, opt
    global policy, INIT_QPOS
    global joint_map, joint_index_map, ALLOWED_JOINTS, GAIT_ACT_IDS
    global steps_needed, steps_done, cmd, action, target_dof_pos
    global counter, old_sin_phase, short_extra_done

    # при желании перезагрузить XML
    if reload_model:
        m = mj.MjModel.from_xml_path(xml_path)
        d = mj.MjData(m)
        m.opt.timestep = simulation_dt
        # пересобираем визуальные структуры под новый model
        scene = mj.MjvScene(m, maxgeom=10000)
        context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)
        opt = mj.MjvOption()

    # перезагрузка политики
    if reload_policy:
        policy = torch.jit.load(policy_path)

    # пересобрать карты/списки под текущий model
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    joint_map = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}": m.jnt_qposadr[j]
                 for j in range(m.njnt)}
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    # PD рук (связывание актуаторов) и нули в целях
    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)
    auto_tune_upper_kp()
    # походочная часть: какие актуаторы трогаем
    GAIT_ACT_IDS = build_gait_act_ids()
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS)

    # полный физический сброс
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    # очистка всего управления/состояния контроллеров
    d.qvel[:] = 0; d.qacc[:] = 0; d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0; d.ctrl[:] = 0
    steps_needed = 0; steps_done = 0
    cmd[:] = 0.0
    action[:] = 0.0
    target_dof_pos = default_angles.copy()
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False

    # руки в ноль
    if ARM_TARGETS is not None and len(ARM_TARGETS) > 0:
        ARM_TARGETS[:] = 0.0

    short_extra_done = True              # не запускать "Balancing: 2 short steps ..."
    last_activity_time = time.time() 

    print("[HARD RESET] model:", "reloaded" if reload_model else "kept",
          "| policy:", "reloaded" if reload_policy else "kept")

def do_reset():
    """Return base to initial pose and set all joints to zero."""
    global steps_needed, steps_done, cmd, action, target_dof_pos
    global counter, old_sin_phase, short_extra_done, last_activity_time

    # Full reset of MuJoCo internal state
    mj.mj_resetData(m, d)

    # Base (free joint) — restore initial configuration (INIT_QPOS)
    d.qpos[:] = INIT_QPOS

    # Zero all joints: hinge/slide -> 0, ball -> unit quaternion
    for j in range(m.njnt):
        jtype = m.jnt_type[j]
        adr = m.jnt_qposadr[j]
        if jtype == mj.mjtJoint.mjJNT_HINGE or jtype == mj.mjtJoint.mjJNT_SLIDE:
            d.qpos[adr] = 0.0
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            d.qpos[adr:adr + 4] = np.array([1.0, 0.0, 0.0, 0.0])
        # mjJNT_FREE (base) stays as in INIT_QPOS

    # Clear dynamics and controls
    d.qvel[:] = 0
    d.qacc[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    d.ctrl[:] = 0

    # Update kinematics
    mj.mj_forward(m, d)

    # Reset internal controller targets/counters
    steps_needed = 0
    steps_done = 0
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False
    cmd[:] = 0.0
    action[:] = 0.0
    # PD target — all joint zeros by default
    target_dof_pos = default_angles.copy()

    last_activity_time = time.time()
    print("Reset done: base -> initial pose, joints -> zero.")

    if ARM_TARGETS is not None and len(ARM_TARGETS) > 0:
        ARM_TARGETS[:] = 0.0   # <— по умолчанию держим нуль

def _control_step_with_arms():
    global counter, target_dof_pos, action, obs, ARM_IERR
    if _should_abort():
        d.ctrl[:] = 0.0
        return

    na = num_actions

    # ноги: если стойка — держим STANCE_TARGET, иначе обычная цель target_dof_pos
    tgt = target_dof_pos
    tau_gait = pd_control(
        tgt,
        d.qpos[7:7 + na],
        kps,
        np.zeros_like(kds),
        d.qvel[6:6 + na],
        kds
    )
    for k, aid in enumerate(GAIT_ACT_IDS):
        d.ctrl[aid] = float(tau_gait[k])

    # руки PD
    if ARM_ACT_IDS:
        q  = d.qpos[ARM_QPOS_ADDRS]
        dq = d.qvel[ARM_QVEL_ADDRS]
        err = ARM_TARGETS - q

        # интеграл (можно включить anti-windup позже)
        ARM_IERR += err * m.opt.timestep
        ARM_IERR = np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP)

        bias = d.qfrc_bias[ARM_QVEL_ADDRS]
        tau_arm = bias + ARM_KPS * err - ARM_KDS * dq + ARM_KIS * ARM_IERR

        act_gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else np.zeros((0,6))

        for k, act_id in enumerate(ARM_ACT_IDS):
            g = float(act_gears[act_id, 0]) if act_gears.size else 1.0
            lo, hi = m.actuator_ctrlrange[act_id]
            u = tau_arm[k] / max(g, 1e-6)          # ВАЖНО: масштаб по gear
            d.ctrl[act_id] = float(np.clip(u, lo, hi))


    mj.mj_step(m, d)
    counter += 1



    if counter % control_decimation == 0:
        qj = d.qpos[7:7 + na]
        dqj = d.qvel[6:6 + na]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]

        qj_scaled = (qj - default_angles) * dof_pos_scale
        dqj_scaled = dqj * dof_vel_scale
        grav = get_gravity_orientation(quat)
        omega_scaled = omega * ang_vel_scale

        period = 0.8
        time_in_sim = counter * simulation_dt
        phase = (time_in_sim % period) / period
        sin_phase = math.sin(2 * math.pi * phase)
        cos_phase = math.cos(2 * math.pi * phase)

        obs[:3] = omega_scaled
        obs[3:6] = grav
        obs[6:9] = cmd * cmd_scale
        obs[9:9 + na] = qj_scaled
        obs[9 + na: 9 + 2 * na] = dqj_scaled
        obs[9 + 2 * na: 9 + 3 * na] = action
        obs[9 + 3 * na: 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action[:] = policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = action * action_scale + default_angles


# =========================
# Motion helpers
# =========================
def move_joints_by_name(joints_list, duration_per_frame=0.3, fps=60):
    """
    Execute motion in one of the formats:
      A) Frames of angles [[8 angles], ...] — as before (for 8 fixed joints).
      B) List of action objects:
         - {"name": "<joint>", "angle": <deg>, "duration": <sec optional>}  # single joint
         - {"frame": [{"name": "<joint>", "angle": <deg>}, ...], "duration": <sec optional>}  # SIMULTANEOUS
         - {"name": "walk", "num": <int>, "dir_deg": <float>, "spd": <float>}
    """
    global last_activity_time
    last_activity_time = time.time()

    if not joints_list:
        return

    if _should_abort():
        print("[ABORT] skip move_joints_by_name (RUN_ABORT)")
        return 
    wants_walk_or_turn = any(
        isinstance(it, dict) and (it.get("name") in ("walk", "turn", "rotate"))
        for it in joints_list
    )
    auto_stance = not wants_walk_or_turn
    if auto_stance:
        start_stance_hold()
    try:
        prev_ctrl = d.ctrl.copy()
        sim_dt = m.opt.timestep
        substeps_per_render = max(1, int((1.0 / fps) / sim_dt))    

        # ===== helpers =====
        def render_once():
            width, height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, width, height)
            mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
            mj.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()

        def smoothstep(t: float) -> float:
            # t in [0,1] -> smooth S-curve
            return t * t * (3 - 2 * t)

        fps_dt = 1.0 / fps

        # ===== Variant A: array of frames with 8 angles =====
        if isinstance(joints_list[0], list):
            print(f"Executing motion with {len(joints_list)} frames (8-angle frames)")
            for frame in joints_list:
                if _should_abort():
                    print("[ABORT] frames")
                    return
                if len(frame) != 8:
                    print(f"Invalid frame: {frame}. Skipping.")
                    continue

                target_angles = {}
                for joint_idx, angle_deg in enumerate(frame):
                    joint_name = joint_index_map.get(joint_idx)
                    if joint_name:
                        target_angles[joint_name] = math.radians(angle_deg)

                # store starting angles
                start_angles = {name: d.qpos[joint_map[name]] for name in target_angles.keys()}

                # spread time uniformly per frame
                target_total_duration = 4.0
                if len(joints_list) > 20:
                    target_total_duration = 4.0 * len(joints_list) / 15
                per_frame_duration = target_total_duration / len(joints_list)
                n_steps = max(1, int(per_frame_duration * fps))

                for step in range(n_steps):
                    if _should_abort():
                        print("[ABORT] frames")
                        return
                    t = smoothstep((step + 1) / n_steps)
                    for jname, target_rad in target_angles.items():
                        start_rad = start_angles[jname]
                        d.qpos[joint_map[jname]] = (1 - t) * start_rad + t * target_rad
                    mj.mj_forward(m, d)
                    render_once()
                    time.sleep(fps_dt)
            return

        # ===== Variant B: list of objects (incl. simultaneous frames via "frame") =====
        for item in joints_list:
            if _should_abort():
                print("[ABORT] action loop")
                return
            # --- NEW: repeat block ---
            if isinstance(item, dict) and "repeat" in item and "times" in item:
                subactions = item["repeat"]
                n_times = int(item["times"])
                if not isinstance(subactions, list) or n_times <= 0:
                    continue
                for _ in range(n_times):
                    if _should_abort():
                        print("[ABORT] repeat")
                        return
                    move_joints_by_name(subactions, duration_per_frame, fps)
                    if _should_abort():
                        print("[ABORT] repeat")
                        return
                continue

            # Walking — leave as-is
            if isinstance(item, dict) and item.get("name") == "walk":
                run_walk_blocking(int(item["num"]), float(item["dir_deg"]), float(item["spd"]))
                continue

            # turn / rotate action
            if isinstance(item, dict) and item.get("name") in ("turn", "rotate"):
                ang = float(item.get("deg") or item.get("angle") or item.get("yaw_deg") or 0.0)
                spd = float(item.get("spd_deg_s") or item.get("speed_deg_s") or 45.0)
                run_turn_blocking(ang, spd)
                continue

            # Frame with several joints — SIMULTANEOUS
            if isinstance(item, dict) and "frame" in item:
                joints = item["frame"]
                if not isinstance(joints, list) or not joints:
                    print(f"Empty or invalid 'frame' in item: {item}")
                    continue

                # Разделяем: цели для рук (через PD) и для остальных (прямо в qpos)
                arm_goals = {}
                direct_targets = {}
                for j in joints:
                    jname = j.get("name")
                    if jname is None:
                        continue
                    try:
                        goal_rad = math.radians(float(j["angle"]))
                    except Exception:
                        print(f"Invalid angle in {j}. Skipping.")
                        continue

                    if jname in ARM_NAME_TO_IDX:
                        arm_goals[jname] = goal_rad
                    elif jname in joint_map:
                        direct_targets[jname] = goal_rad
                    else:
                        print(f"Joint '{jname}' not found. Skipping.")

                if not arm_goals and not direct_targets:
                    continue

                duration = float(item.get("duration", duration_per_frame))
                n_steps = max(1, int(duration * fps))

                # Стартовые значения
                start_arm = {jn: float(ARM_TARGETS[ARM_NAME_TO_IDX[jn]]) for jn in arm_goals.keys()}
                start_angles = {jn: d.qpos[joint_map[jn]] for jn in direct_targets.keys()}

                for step in range(n_steps):
                    if _should_abort():
                        print("[ABORT] single-joint")
                        return
                    t = smoothstep((step + 1) / n_steps)

                    # РУКИ: плавно ведём PD-цели
                    for jname, goal in arm_goals.items():
                        idx = ARM_NAME_TO_IDX[jname]
                        ARM_TARGETS[idx] = (1 - t) * start_arm[jname] + t * goal

                    # ПРОЧЕЕ: как раньше — напрямую в qpos
                    for jname, target_rad in direct_targets.items():
                        start_rad = start_angles[jname]
                        d.qpos[joint_map[jname]] = (1 - t) * start_rad + t * target_rad

                    # --- дать поработать PD по рукам (мини-шаги физики) ---
                    for _ in range(substeps_per_render):
                        if _should_abort():
                            print("[ABORT] single-joint/substep")
                            return
                        _control_step_with_arms()
                    render_once()
                    time.sleep(fps_dt)
                continue


            # Regular single joint — backward compatible with previous format
            if isinstance(item, dict) and "name" in item and "angle" in item:
                jname = item["name"]
                target_angle_rad = math.radians(float(item["angle"]))
                duration = float(item.get("duration", duration_per_frame))
                n_steps = max(1, int(duration * fps))

                if jname in ARM_NAME_TO_IDX:
                    idx = ARM_NAME_TO_IDX[jname]
                    start = float(ARM_TARGETS[idx])
                    for step in range(n_steps):
                        t = smoothstep((step + 1) / n_steps)
                        ARM_TARGETS[idx] = (1 - t) * start + t * target_angle_rad

                        # даём PD-контролю рук поработать внутри каждого шага интерполяции
                        for _ in range(substeps_per_render):
                            _control_step_with_arms()
                        render_once()
                        time.sleep(fps_dt)
                    continue


                if jname not in joint_map:
                    print(f"Joint '{jname}' not found.")
                    continue

                qpos_idx = joint_map[jname]
                start_rad = d.qpos[qpos_idx]
                for step in range(n_steps):
                    t = smoothstep((step + 1) / n_steps)
                    d.qpos[qpos_idx] = (1 - t) * start_rad + t * target_angle_rad
                    mj.mj_forward(m, d)
                    render_once()
                    time.sleep(fps_dt)
                continue


            print(f"[WARN] Unknown item format: {item}")
    finally:
        if auto_stance:
            stop_stance_hold()    


def start_new_steps(num, dir_deg, speed, is_extra=False):
    """
    Start a new walking sequence:
      - num: number of steps
      - dir_deg: direction in degrees (0° = +X, 90° = +Y)
      - speed: linear speed (m/s)
      - is_extra: internal flag used when auto-balancing
    """
    stop_stance_hold()
    global last_activity_time, steps_needed, steps_done, cmd, old_sin_phase, short_extra_done
    last_activity_time = time.time()
    if not is_extra:
        short_extra_done = False

    steps_needed = num
    steps_done = 0
    old_sin_phase = 0.0

    dir_rad = math.radians(dir_deg)
    vx = speed * math.cos(dir_rad)
    vy = speed * math.sin(dir_rad)
    cmd[:] = [vx, vy, 0.0]
    print(f"Starting {steps_needed} steps for direction = {dir_deg}°, speed = {speed} m/s")


def _vlm_describe_image(image_path: str) -> str:
    """Ask the VLM to briefly describe the image for safe context."""
    try:
        return LLM.chat_vision_describe(image_path)
    except Exception as e:
        return f"[vision description failed: {e}]"


# =========================
# GLFW input callbacks
# =========================

def key_callback(window_, key, scancode, action, mods):
    pressed = (action == glfw.PRESS or action == glfw.REPEAT)
    if action == glfw.RELEASE:
        pressed = False

    # WASD
    if key == glfw.KEY_W:
        movement["forward"] = pressed
    elif key == glfw.KEY_S:
        movement["backward"] = pressed
    elif key == glfw.KEY_A:
        movement["left"] = pressed
    elif key == glfw.KEY_D:
        movement["right"] = pressed

    # Up/Down
    elif key == glfw.KEY_UP:
        movement["rise"] = pressed
    elif key == glfw.KEY_DOWN:
        movement["fall"] = pressed

    # ESC
    elif key == glfw.KEY_ESCAPE and pressed:
        glfw.set_window_should_close(window_, True)

        # views to VLM
    elif key == glfw.KEY_V and pressed:
        capture_and_execute_vlm_batch()
    

    # 'O' — example placeholder: do nothing (empty list)
    elif key == glfw.KEY_O and pressed:
        list_unit = []
        move_joints_by_name(list_unit)

    elif key == glfw.KEY_T and pressed:
        try:
            ang = float(input("Turn angle (degrees, +left / -right): "))
            spd = input("Speed (deg/s) [45]: ").strip()
            spd = float(spd) if spd else 45.0
        except Exception:
            print("Invalid input.")
            return
        print(f"Rotating by {ang}° at {spd}°/s…")
        run_turn_blocking(ang, spd)

    elif key == glfw.KEY_R and pressed:
        abort_all_motion()            # <- ВАЖНО: сначала просим остановку
        do_reset()

    elif key == glfw.KEY_H and pressed:
        abort_all_motion()            # <- сначала стоп всего
        hard_reset(reload_model=False, reload_policy=True)
        auto_tune_upper_kp()

    # 'P' — paste motions via modal window (frames or action objects)
    elif key == glfw.KEY_P and pressed:
        cmd_list = open_paste_window()
        if cmd_list:
            clear_abort()               # ← добавьте
            move_joints_by_name(cmd_list)
    # 'L' — natural language command -> parsed JSON actions -> execute
    elif key == glfw.KEY_L and pressed:
        user_cmd = input("Enter the command in natural language: ")
        print("Thinking...")
        parsed_moves = parse_command(
            user_cmd,
            rag_k=settings.RAG_K,
            rag_csv=settings.RAG_CSV,
            rag_json=settings.RAG_JSON
        )
        print(parsed_moves)
        clear_abort() 
        move_joints_by_name(parsed_moves)
        # Ask about saving
        ans = input("Save this motion to JSON? (y/N): ").strip().lower()
        if ans in ("y", "yes"):
            save_actions_json(parsed_moves, command_text=user_cmd, out_dir="data_action", filename=None)

    # 'C' — type walking parameters in console
    elif key == glfw.KEY_C and pressed:
        try:
            num = int(input("Number of steps (0 to cancel)? "))
        except Exception:
            print("Invalid number.")
            return

        if num <= 0:
            print("0 steps. Robot will stay.")
            return

        try:
            dir_deg = float(input("Direction (degrees, 0° = +X, 90° = +Y): "))
            spd = float(input("Speed (m/s): "))
        except Exception:
            print("Invalid direction/speed.")
            return
        clear_abort()
        start_new_steps(num, dir_deg, spd)

    # 'G' — Generate from CSV: LLM -> play -> save <id>.json
    elif key == glfw.KEY_G and pressed:
        try:
            path = input("CSV path [1000.csv]: ").strip() or "1000.csv"
            start_index = input("Start index (0-based) [0]: ").strip()
            start_index = int(start_index) if start_index else 0
            max_count = input("How many rows to process (blank=all): ").strip()
            max_count = int(max_count) if max_count else None
            interval = input("Interval between samples, seconds [0.0]: ").strip()
            interval = float(interval) if interval else 0.0
            preview = input("Preview only (y/N)? ").strip().lower() in ("y", "yes")
        except Exception as e:
            print(f"[ERR] invalid input: {e}")
            return

        generate_from_csv_and_play(
            csv_path=path,
            out_dir="data_action",
            start_index=start_index,
            max_count=max_count,
            interval_sec=interval,
            preview_only=preview,
        )

    elif key == glfw.KEY_Q and pressed:
        print("Starting demo...")
        csv_ = input("Enter CSV path for demo: ")
        jsons_ = input("Enter JSON directory for demo: ")
        run_demo(csv_path=csv_, json_dir=jsons_)

    elif key == glfw.KEY_F and pressed:
        data = open_cmd_with_image_window()
        if data:
            user_cmd = data["text"]
            image_path = data["image_path"]
            brief = _vlm_describe_image(image_path=image_path)
            print(brief)
            print("[/Vision brief]\n")

            print("Thinking with vision...")
            parsed_moves = parse_command_with_image(user_cmd, image_path)
            print(parsed_moves)
            move_joints_by_name(parsed_moves)
def auto_tune_upper_kp(eps_deg=1.0, safety=1.5):
    eps = math.radians(eps_deg)
    mj.mj_forward(m, d)
    bias = np.abs(d.qfrc_bias[ARM_QVEL_ADDRS])
    req_kp = safety * bias / max(eps, 1e-5)
    global ARM_KPS, ARM_KDS
    ARM_KPS = np.maximum(ARM_KPS, req_kp).astype(np.float32)
    ARM_KDS = np.clip(ARM_KPS * 0.05, 0.2, None).astype(np.float32)
    print("[UPPER] retuned Kp:", ARM_KPS)


def mouse_button_callback(window_, button, action, mods):
    global is_dragging, last_cursor_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_cursor_pos = glfw.get_cursor_pos(window_)
        elif action == glfw.RELEASE:
            is_dragging = False


def cursor_pos_callback(window_, xpos, ypos):
    global is_dragging, last_cursor_pos, cam
    if is_dragging:
        x0, y0 = last_cursor_pos
        dx = xpos - x0
        dy = ypos - y0
        sensitivity = 0.4
        cam.azimuth -= sensitivity * dx
        cam.elevation -= sensitivity * dy
        last_cursor_pos = (xpos, ypos)


def scroll_callback(window_, x_offset, y_offset):
    global cam
    zoom_factor = 1.05
    if y_offset < 0:
        cam.distance *= zoom_factor
    else:
        cam.distance /= zoom_factor
    cam.distance = max(0.05, min(100.0, cam.distance))


# =========================
# Main
# =========================
import argparse
if __name__ == "__main__":
    last_activity_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-k", type=int, default=5, help="number of examples to few-shot")
    parser.add_argument("--rag-csv", default="1000.csv", help="CSV with commands")
    parser.add_argument("--rag-json", default="data_action", help="dir with JSON")
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default="g1",
                        help="which robot to load (g1, h1, h1_2)")
    args = parser.parse_args()

    settings.RAG_K = args.rag_k
    settings.RAG_CSV = args.rag_csv
    settings.RAG_JSON = args.rag_json

    name_ = args.robot

    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    #xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene.xml" # with objects
    xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene.xml"

    # Load configuration
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

    # Initialize MuJoCo
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt

 
    GAIT_ACT_IDS = build_gait_act_ids()
    assert len(GAIT_ACT_IDS) == num_actions
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS)
    print("[GAIT] actuators used by walking policy:", GAIT_ACT_IDS)

    # (1) Build joint name -> qpos index map + compact overview
    joint_map = {}

    # Build dynamic ALLOWED_JOINTS from the model (hinge/slide only)
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    print(f"\nAllowed joints for commands ({len(ALLOWED_JOINTS)}):")
    print(", ".join(sorted(ALLOWED_JOINTS)))
    print()

    # Build dynamic frame joint order (index -> name)
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    print_gait_policy_io()
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)
    print(f"Frame joint order ({len(joint_index_map)}):")
    print(", ".join(joint_index_map[i] for i in range(len(joint_index_map))))
    print()

    # Reset to defaults and save initial base pose (free joint) from XML
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)
    auto_tune_upper_kp()
    INIT_QPOS = d.qpos.copy()

    def axis_letter(ax: np.ndarray) -> str:
        """Return X/Y/Z based on the largest component of the joint axis direction."""
        if np.linalg.norm(ax) < 1e-8:
            return "-"
        return "XYZ"[int(np.argmax(np.abs(ax)))]

    print("\nJoints overview (compact):")
    print(f"{'id':>3} {'name':<28} {'range':>21} {'axis':>4}")
    print("-" * 60)

    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]

        jtype = m.jnt_type[j]
        limited = bool(m.jnt_limited[j])
        rng = m.jnt_range[j]

        # Range string
        if jtype == mj.mjtJoint.mjJNT_HINGE:
            if limited:
                rng_str = f"{np.degrees(rng[0]):7.2f}..{np.degrees(rng[1]):7.2f} deg"
            else:
                rng_str = "unlimited"
        elif jtype == mj.mjtJoint.mjJNT_SLIDE:
            rng_str = f"{rng[0]:7.3f}..{rng[1]:7.3f} m" if limited else "unlimited"
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            rng_str = "quat (no range)"
        else:  # free
            rng_str = "free (6-DoF)"

        # Axis letter
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            ax_letter = axis_letter(m.jnt_axis[j])
        else:
            ax_letter = "-"

        print(f"{j:3d} {name:<28} {rng_str:>21} {ax_letter:>4}")
    print()

    # Load policy
    policy = torch.jit.load(policy_path)

    # Create GLFW window
    glfw.init()
    width, height = 1200, 900
    window = glfw.create_window(width, height, "MuJoCo Manual Viewer", None, None)
    glfw.make_context_current(window)

    # Rendering structures
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(m, maxgeom=10000)
    context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    # Camera params
    cam.azimuth = 180
    cam.elevation = -15
    cam.distance = 4.0
    cam.lookat = np.array([0.0, 0.0, 0.8])

    # Camera keys (WASD + Up/Down)
    camera_speed = 0.2
    movement = {"forward": False, "backward": False, "left": False, "right": False, "rise": False, "fall": False}

    # Register input callbacks
    glfw.set_key_callback(window, key_callback)

    is_dragging = False
    last_cursor_pos = (0, 0)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # =========================
    # Walking / balance control state
    # =========================
    steps_needed = 0
    steps_done = 0
    cmd = np.zeros(3, dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    old_sin_phase = 0.0

    short_extra_done = False
    last_dir_deg = 0.0

    # =========================
    # Main simulation loop
    # =========================
    while not glfw.window_should_close(window):
        azimuth_rad = math.radians(cam.azimuth)
        right_vec = np.array([math.sin(azimuth_rad), -math.cos(azimuth_rad), 0])
        forward_vec = np.array([right_vec[1], -right_vec[0], 0])

        if movement["forward"]:
            cam.lookat -= camera_speed * forward_vec
        if movement["backward"]:
            cam.lookat += camera_speed * forward_vec
        if movement["left"]:
            cam.lookat -= camera_speed * right_vec
        if movement["right"]:
            cam.lookat += camera_speed * right_vec
        if movement["rise"]:
            cam.lookat[2] += camera_speed
        if movement["fall"]:
            cam.lookat[2] -= camera_speed

   
    

        # NOTE: This overrides num_actions each loop as in the original script.
        # Keep as-is unless your model/config demands otherwise.
        num_actions = config["num_actions"] 

        tau = pd_control(
            target_dof_pos,
            d.qpos[7:7 + num_actions],
            kps,
            np.zeros_like(kds),
            d.qvel[6:6 + num_actions],
            kds
        )

        for k, aid in enumerate(GAIT_ACT_IDS):
            d.ctrl[aid] = float(tau[k])

        if ARM_ACT_IDS:
            q  = d.qpos[ARM_QPOS_ADDRS]
            dq = d.qvel[ARM_QVEL_ADDRS]
            # ошибка и интеграл
            err = ARM_TARGETS - q
            ARM_IERR += err * m.opt.timestep
            ARM_IERR = np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP)

            # компенсация тяжести/пассивки для этих dof'ов
            bias = d.qfrc_bias[ARM_QVEL_ADDRS]  # размер = len(ARM_ACT_IDS)

            # итоговый момент
            tau_arm = bias + ARM_KPS * err - ARM_KDS * dq + ARM_KIS * ARM_IERR

            # извлечь gear для actuator'ов (для JOINT берём компоненту [0])
            act_gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else np.zeros((0,6))

            for k, act_id in enumerate(ARM_ACT_IDS):
                g = float(act_gears[act_id, 0]) if act_gears.size else 1.0
                lo, hi = m.actuator_ctrlrange[act_id]
                u = tau_arm[k] / max(g, 1e-6)        # масштаб: ctrl = τ / gear
                d.ctrl[act_id] = float(np.clip(u, lo, hi))

        mj.mj_step(m, d)
        counter += 1

        if STANCE_HOLD or (steps_done < steps_needed):
            if counter % control_decimation == 0:
                qj = d.qpos[7:7 + num_actions]
                dqj = d.qvel[6:6 + num_actions]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                period = 0.8
                time_in_sim = counter * simulation_dt
                phase = (time_in_sim % period) / period
                sin_phase = math.sin(2 * math.pi * phase)
                cos_phase = math.cos(2 * math.pi * phase)

                obs[:3] = omega_scaled
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj_scaled
                obs[9 + num_actions: 9 + 2 * num_actions] = dqj_scaled
                obs[9 + 2 * num_actions: 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action[:] = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles
                if STANCE_HOLD:
                    # знак ритма по фазе походки
                    sgn = 1.0 if sin_phase >= 0.0 else -1.0

                    # индексы осей
                    axis = 0 if STANCE_AXIS == "x" else 1
                    oth  = 1 - axis

                    # --- главная ось: PI + ритм ---
                    pos = float(d.qpos[axis])
                    vel = float(d.qvel[axis])
                    home = float(STANCE_HOME[axis]) if STANCE_HOME is not None else pos
                    err = pos - home

                    # интегратор с анти-нарастанием
                    STANCE_I_MAIN += err * simulation_dt
                    if STANCE_I_MAIN >  STANCE_I_CLAMP: STANCE_I_MAIN =  STANCE_I_CLAMP
                    if STANCE_I_MAIN < -STANCE_I_CLAMP: STANCE_I_MAIN = -STANCE_I_CLAMP

                    v_main = (
                        sgn * STANCE_V          # ритм
                        - STANCE_KP * err       # позиционная коррекция
                        - STANCE_KD * vel       # демпфирование
                        - STANCE_KI * STANCE_I_MAIN  # интеграл
                    )
                    # ограничение амплитуды
                    if v_main >  STANCE_V: v_main =  STANCE_V
                    if v_main < -STANCE_V: v_main = -STANCE_V

                    # --- поперечная ось: PD к дому (без ритма) ---
                    pos_o = float(d.qpos[oth])
                    vel_o = float(d.qvel[oth])
                    home_o = float(STANCE_HOME[oth]) if STANCE_HOME is not None else pos_o
                    err_o = pos_o - home_o

                    v_orth = - STANCE_ORTH_KP * err_o - STANCE_ORTH_KD * vel_o
                    vmax_o = 0.5 * STANCE_V
                    if v_orth >  vmax_o: v_orth =  vmax_o
                    if v_orth < -vmax_o: v_orth = -vmax_o

                    # --- удержание курса (yaw-lock) ---
                    qw, qx, qy, qz = d.qpos[3:7]
                    yaw_now = _quat_yaw(qw, qx, qy, qz)
                    yaw_err = _wrap_to_pi(yaw_now - (STANCE_YAW_HOME if STANCE_YAW_HOME is not None else yaw_now))
                    yaw_rate = float(d.qvel[5])  # ω_z
                    yaw_cmd = - STANCE_YAW_KP * yaw_err - STANCE_YAW_KD * yaw_rate

                    # собрать команду в нужном порядке осей
                    if axis == 0:    # главная — X
                        cmd[:] = [v_main, v_orth, yaw_cmd]
                    else:            # главная — Y
                        cmd[:] = [v_orth, v_main, yaw_cmd]
                else:
                    # обычный счёт шагов при запрошенной ходьбе
                    if sin_phase >= 0 and old_sin_phase < 0:
                        steps_done += 1
                        print(f"Got step #{steps_done} / {steps_needed}")
                        if steps_done >= steps_needed:
                            cmd[:] = 0.0
                            target_dof_pos = default_angles.copy()

                old_sin_phase = sin_phase


        elif not STANCE_HOLD:
            # Stop motion
            d.qvel[:] = 0
            d.qacc[:] = 0
            time_to_balance = 300
         #   Do a couple of short balancing steps once, then periodically if idle for > 30s
            if not short_extra_done:
                short_extra_done = True
                print("Balancing: 2 short steps speed=0.1")
                start_new_steps(2, 0.0, 0.1, is_extra=True)
            
            elif ((time.time() - last_activity_time) > time_to_balance):
                print(f"No activity for {time_to_balance}s. Doing balance steps.")
                start_new_steps(2, 0.0, 0.1, is_extra=True)
                last_activity_time = time.time()

        # Render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
    print("Window closed, exiting.")
