# =========================
# Imports
# =========================
import time
import math
import json
import ast
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import yaml
import torch
import tkinter as tk

import mujoco as mj
from mujoco.glfw import glfw

def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw around +Z (MuJoCo: Z-up)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))




def _resolve_joint_name(jname: str) -> str | None:
    """Selects the name of the joint taking into account the suffix _joint and prefixes r1_/r2_."""
    if jname in joint_map:
        return jname
    # try without suffix "_joint"
    if jname.endswith("_joint"):
        base = jname[:-6]
        if base in joint_map:
            return base
    # if in JSON came without a prefix, but there is one in the scene
    for pref in ("r1_", "r2_", "r3_"):

        cand = pref + jname
        if cand in joint_map:
            return cand
        if jname.endswith("_joint"):
            base = jname[:-6]
            cand2 = pref + base
            if cand2 in joint_map:
                return cand2
    return None

# === PATCH: UPPER-BODY PD state (arms/hands/fingers) ===
UPPER_HOLD_ZERO = True  # keep zeros by default

# recognizing which joints belong to the top
import re as _re
_LEG_TOKENS    = ("hip", "knee", "ankle", "toe", "foot", "leg")
_FINGER_TOKENS = ("thumb","index","middle","ring","pinky","finger")
_UPPER_TOKENS  = ("shoulder","elbow","wrist","hand","torso","waist","spine","chest","neck","head") + _FINGER_TOKENS

def _is_upper_joint(name: str) -> bool:
    nm = name.lower()
    nm = _re.sub(r"^r\d+_", "", nm)        # cut r1_/r2_/r3_
    if nm.endswith("_joint"): nm = nm[:-6]
    if any(tok in nm for tok in _LEG_TOKENS):   # quickly cut off the legs
        return False
    if any(tok in nm for tok in _UPPER_TOKENS):
        return True
    if nm.startswith(("l_","r_")):
        rest = nm[2:]
        if any(tok in rest for tok in _FINGER_TOKENS + ("wrist","hand","elbow","shoulder")):
            return True
    return False

# indices/odds PD along the upper joints
ARM_NAME_TO_IDX: dict[str,int] = {}
ARM_JIDS:        list[int]     = []
ARM_ACT_IDS:     list[int]     = []
ARM_QPOS_ADDRS:  list[int]     = []
ARM_QVEL_ADDRS:  list[int]     = []
ARM_TARGETS:     np.ndarray | None = None
ARM_KPS:         np.ndarray | None = None
ARM_KDS:         np.ndarray | None = None
ARM_KIS:         np.ndarray | None = None
ARM_IERR:        np.ndarray | None = None
ARM_I_CLAMP = 0.7  # let's limit the integral
# === PATCH: bind all upper-body hinge joints to actuators and init PD ===
def _suggest_kp(name: str) -> float:
    n = name.lower()
    if "torso" in n:   return 40.0
    if "shoulder" in n:return 40.0
    if "elbow" in n:   return 30.0
    if "wrist" in n:   return 12.0
    return 4.0  # fingers

def setup_arm_pd(zero_pose: bool = True):
    global ARM_NAME_TO_IDX, ARM_JIDS, ARM_ACT_IDS, ARM_QPOS_ADDRS, ARM_QVEL_ADDRS
    global ARM_TARGETS, ARM_KPS, ARM_KDS, ARM_KIS, ARM_IERR

    ARM_NAME_TO_IDX.clear()
    ARM_JIDS.clear(); ARM_ACT_IDS.clear(); ARM_QPOS_ADDRS.clear(); ARM_QVEL_ADDRS.clear()

    # let's collect everything hinge-upper joints
    upper_joints: list[str] = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
        if _is_upper_joint(nm):
            upper_joints.append(nm)

    # fallback map: joint_id -> actuator_id (For actuator With target==this joint)
    trn  = np.array(m.actuator_trnid).reshape(m.nu, 2) if m.nu > 0 else np.zeros((0,2), int)
    trnt = np.array(m.actuator_trntype) if m.nu > 0 else np.zeros((0,), int)
    jid2act: dict[int,int] = {}
    for i in range(m.nu):
        if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
            jid2act.setdefault(int(trn[i,0]), i)

    names_in_order: list[str] = []
    for name in upper_joints:
        nm_low = name.lower()
        if "torso" in nm_low:   # <<< add this
            continue
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0: continue
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0: aid = jid2act.get(jid, -1)
        if aid < 0: 
            # no actuator - skip
            continue
        idx = len(ARM_JIDS)
        ARM_NAME_TO_IDX[name] = idx
        ARM_JIDS.append(jid); ARM_ACT_IDS.append(aid)
        ARM_QPOS_ADDRS.append(m.jnt_qposadr[jid])
        ARM_QVEL_ADDRS.append(m.jnt_dofadr[jid])
        names_in_order.append(name)

    n = len(ARM_JIDS)
    if n == 0:
        ARM_TARGETS = np.zeros(0, np.float32)
        ARM_KPS = ARM_KDS = ARM_KIS = ARM_IERR = ARM_TARGETS.copy()
        print("[UPPER] no upper-body joints bound")
        return

    if zero_pose:
        ARM_TARGETS = np.zeros(n, np.float32)
        print(f"[UPPER] zero-hold enabled on {n} joints")
    else:
        ARM_TARGETS = np.array([d.qpos[i] for i in ARM_QPOS_ADDRS], np.float32)

    ARM_KPS = np.array([_suggest_kp(nm) for nm in names_in_order], np.float32)
    ARM_KDS = np.clip(ARM_KPS * 0.05, 0.2, None).astype(np.float32)
    ARM_KIS = np.clip(ARM_KPS * 0.02, 0.0, 1.0).astype(np.float32)
    ARM_IERR= np.zeros_like(ARM_KPS, dtype=np.float32)
    print(f"[UPPER] PD enabled for {n} joints")

# === PATCH: animate a frame for *upper* joints by ramping ARM_TARGETS ===
def _arm_apply_frame(frame_items: list[dict], duration: float, fps: int = 60):
    if ARM_TARGETS is None or ARM_TARGETS.size == 0:
        return

    # 1) We collect targets only at the "upper" joints
    to_set: dict[int, float] = {}
    for j in frame_items:
        raw = str(j.get("name", ""))
        jname = _resolve_joint_name(raw) or raw
        if jname not in ARM_NAME_TO_IDX:
            # it is useful to highlight that the joint is not recognized as the upper one
            print(f"[UPPER/PD][skip] not an upper-body joint or no actuator: {jname}")
            continue
        idx = ARM_NAME_TO_IDX[jname]
        q = math.radians(float(j["angle"]))
        to_set[idx] = q
        # debug log - which SEPARATE upper joint is actually used
        _dbg_joint_apply_pd(jname, float(j["angle"]), idx, ARM_ACT_IDS[idx], ARM_QPOS_ADDRS[idx])

    if not to_set:
        return

    # 2) We are planning a smooth transition
    n_steps = max(1, int(duration * fps))
    start = ARM_TARGETS.copy()
    goal  = ARM_TARGETS.copy()
    for i, q in to_set.items():
        goal[i] = q

    def smoothstep(t: float) -> float:
        return t * t * (3 - 2 * t)

    # 3) Linearization in time with simulation ticks
    for k in range(n_steps):
        t = smoothstep((k + 1) / n_steps)
        ARM_TARGETS[:] = start * (1.0 - t) + goal * t

        # we don’t let the "freeze" turn on while the scripted animation is in progress
        globals()["last_activity_time"] = time.time()
        globals()["BUSY_SCRIPT"] = max(1, globals().get("BUSY_SCRIPT", 0))

        _one_tick(render=True)


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

R = {} 
POL = {}

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

def parse_robot_input(sel) -> list[str]:
    """
    Supports:
      - None / ""           -> [ACTIVE]
      - "all", "*", "both"  -> all available from R
      - "r1,r3" / "1,3"     -> ["r1_","r3_"]
      - ["r1_","r3_"]       -> as is, with filter by R
      - "r1_" / "r2" / 2    -> single values
    Returns a list of normalized prefixes present in R.
    """
    # 1) Empty → ACTIVE
    if sel is None or sel == "":
        return [ACTIVE]

    # 2) Sets: list/tuple/set → glue the result together
    if isinstance(sel, (list, tuple, set)):
        out = []
        for item in sel:
            out.extend(parse_robot_input(item))
        # unique in order of appearance and filtered by R
        uniq = []
        for p in out:
            if p in R and p not in uniq:
                uniq.append(p)
        return uniq

    # 3) Numbers → "rN_"
    if isinstance(sel, (int, float)):
        cand = f"r{int(sel)}_"
        return [cand] if cand in R else []

    # 4) Line: cases "all" and transfers
    s = str(sel).strip().lower()
    if s in ("all", "any", "*", "both", "r12", "r1+r2", "r1+r2+r3"):
        return [p for p in ("r1_", "r2_", "r3_") if p in R]

    # allow already-normalized tokens "r1_" / "r2_"
    tokens = []
    for raw in s.replace(";", ",").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok.isdigit():
            tok = f"r{tok}_"
        else:
            # if already "r1_" — let's leave it as is
            if not tok.endswith("_"):
                # If "r1" without underlining - add
                if tok.startswith("r") and tok[1:].isdigit():
                    tok = tok + "_"
                else:
                    tok = tok + "_" if not tok.startswith("r") else tok + "_"
        tokens.append(tok)

    out = []
    for t in tokens:
        if t in R and t not in out:
            out.append(t)
    # if nothing is recognized, let the calling party decide through "or [ACTIVE]"
    return out


def ask_robots(prompt_default_active: bool = True) -> list[str]:
    hint = "Robots [all | r1,r2,r3 | 1,3 | Enter=ACTIVE]: " if prompt_default_active \
           else "Robots [all | r1,r2,r3 | 1,3]: "
    s = input(hint)
    return parse_robot_input(s if prompt_default_active else (s or "all"))


# ==== ADD (global settings) ====
IDLE_AFTER = 5   # seconds of silence until «freeze»
IDLE_SLEEP = 0.05  # pause interval
BUSY_SCRIPT = 0  # >0 — scripted animation in progress/team

def _is_idle() -> bool:
    # 0) if we move the camera, we don't idle
    if _camera_active():
        return False
    # NEW: if there is script activity - no idle
    if BUSY_SCRIPT > 0:
        return False

    if manual_override:
        return False
    if any(WALK_UNTIL.get(p, 0.0) or TURN_TGT.get(p, 0.0) > 0.0 for p in robots):
        return False
    if any(ACTIVE_TASK.get(p) or TASKS.get(p) for p in robots):
        return False
    if any(np.any(cmd_vec.get(p, np.zeros(3))) for p in robots):
        return False
    return True



# somewhere close to the rest of the globals
is_dragging = False  # to be safely used in checks before main()

def _camera_active() -> bool:
    # is there any pressed camera movement
    return any(movement.get(k, False) for k in ("forward","backward","left","right","rise","fall")) or is_dragging


def _maybe_freeze() -> bool:
    global last_activity_time
    # if we move the camera, this is an activity: we don’t freeze
    if _camera_active():
        last_activity_time = time.time()
        return False

    if not _is_idle():
        last_activity_time = time.time()
        return False
    if time.time() - last_activity_time < IDLE_AFTER:
        return False

    # stop physics...
    d.ctrl[:] = 0; d.qvel[:] = 0; d.qacc[:] = 0; d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0

    # although we are not following physics - the camera can be updated
    _update_camera()

    width, height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, width, height)
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.wait_events_timeout(IDLE_SLEEP)
    return True


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
    if preferred_order is None:
        preferred_order = [
            "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow_pitch","left_elbow_roll",
            "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow_pitch","right_elbow_roll",
        ]

    names, missing = [], []
    for base in preferred_order:
        if base in joint_map:
            names.append(base); continue
        found = None
        for pref in ("r1_","r2_"):
            for cand in (f"{pref}{base}_joint", f"{pref}{base}"):
                if cand in joint_map:
                    found = cand; break
            if found: break
        if found:
            names.append(found)
        else:
            missing.append(base)

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

# ===== Upper-body zero-hold (for both robots) =====
HOLD_UPPER_ZERO = True
UPPER_BASENAMES = [
    # ===== LEFT ARM =====
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw",
    "left_elbow_pitch","left_elbow_roll",
    "left_wrist_pitch","left_wrist_yaw",
    # Left hand — thumb + fingers
    "L_thumb_proximal_yaw","L_thumb_proximal_pitch","L_thumb_intermediate","L_thumb_distal",
    "L_index_proximal","L_index_intermediate",
    "L_middle_proximal","L_middle_intermediate",
    "L_ring_proximal","L_ring_intermediate",
    "L_pinky_proximal","L_pinky_intermediate",

    # ===== RIGHT ARM =====
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw",
    "right_elbow_pitch","right_elbow_roll",
    "right_wrist_pitch","right_wrist_yaw",
    # Right hand — thumb + fingers
    "R_thumb_proximal_yaw","R_thumb_proximal_pitch","R_thumb_intermediate","R_thumb_distal",
    "R_index_proximal","R_index_intermediate",
    "R_middle_proximal","R_middle_intermediate",
    "R_ring_proximal","R_ring_intermediate",
    "R_pinky_proximal","R_pinky_intermediate",
]

HOLD_UPPER_QPOS: dict[str, np.ndarray] = {}   # by work -> indices qpos
HOLD_UPPER_DOF:  dict[str, np.ndarray] = {}   # by work -> indices dof (For qvel=0)

def _build_upper_hold_maps():
    """Collecting indices of the upper joints (qpos And dof) for each robot r1_/r2_."""
    global HOLD_UPPER_QPOS, HOLD_UPPER_DOF
    HOLD_UPPER_QPOS = {}
    HOLD_UPPER_DOF  = {}

    def _jid_by_candidates(cands: list[str]) -> int | None:
        for nm in cands:
            try:
                j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, nm)
                if j >= 0:
                    return j
            except Exception:
                pass
        return None

    for p in R.keys():  # "r1_", "r2_"
        qpos_idx = []
        dof_idx  = []
        for base in UPPER_BASENAMES:
            # we try both variants of names: with "_joint" and without, with the required prefix
            cands = [f"{p}{base}_joint", f"{p}{base}"]
            j = _jid_by_candidates(cands)
            if j is None:
                continue
            qpos_idx.append(int(m.jnt_qposadr[j]))
            dof_idx.append(int(m.jnt_dofadr[j]))
        if qpos_idx:
            HOLD_UPPER_QPOS[p] = np.array(sorted(set(qpos_idx)), dtype=int)
            HOLD_UPPER_DOF[p]  = np.array(sorted(set(dof_idx)),  dtype=int)
        else:
            HOLD_UPPER_QPOS[p] = np.array([], dtype=int)
            HOLD_UPPER_DOF[p]  = np.array([], dtype=int)
    print(f"[HOLD] upper joints collected:",
          {p: (len(HOLD_UPPER_QPOS[p]), len(HOLD_UPPER_DOF[p])) for p in R.keys()})


# --- Policy wrapper: stateless/state + soft reset ---
class PolicyWrapper:
    def __init__(self, model_or_path: str | torch.jit.ScriptModule):
        if isinstance(model_or_path, str):
            self.path = model_or_path
            self.model = torch.jit.load(self.path)
        else:
            self.path = None
            self.model = model_or_path
        self.state = None  # in case the model returns (act, new_state)

    def reset(self, hard: bool = False):
        """Resetting the internal state of the controller. At hard=True — reboot .pt."""
        self.state = None
        # if the model has .reset() — let's try to call
        if hasattr(self.model, "reset") and callable(getattr(self.model, "reset")):
            try:
                self.model.reset()
            except Exception:
                pass
        if hard and self.path:
            try:
                self.model = torch.jit.load(self.path)
            except Exception:
                pass

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """The challenge of politics. View output support act OR (act, new_state)."""
        x = torch.from_numpy(obs_np).unsqueeze(0)  # (1, num_obs)
        out = self.model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            act_t, new_state = out[0], out[1]
            # keep state, if suddenly the model is recurrent
            try:
                self.state = new_state
            except Exception:
                pass
        else:
            act_t = out
        return act_t.detach().numpy().squeeze()



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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control law."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def do_reset():
    """Hard reset = as when starting the program: exact restoration spawn-state."""
    global steps_needed, steps_done, counter, old_sin_phase, short_extra_done, last_activity_time
    global cmd, cmd_vec, action, target, robots
    global WALK_UNTIL, TURN_UNTIL, TURN_ACC, TURN_LAST, TURN_TGT
    global ACTIVE_TASK, TASKS

    # 1) Complete reset of physics and return exactly to INIT_QPOS (from XML)
    mj.mj_resetData(m, d)
    d.qpos[:] = INIT_QPOS        # <<< the key thing is not to grind ANYTHING after this
    d.qvel[:] = 0
    d.qacc[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    d.ctrl[:] = 0
    mj.mj_forward(m, d)
    


    # 2) Service counters/phases
    steps_needed = 0
    steps_done = 0
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False
    last_activity_time = time.time()

    # 3) Teams/policies to zero, goals as at the start (== current joints)
    try:
        cmd[:] = 0.0
    except Exception:
        pass

    for p in robots:
        WALK_GOAL[p]  = 0
        WALK_COUNT[p] = 0
        cmd_vec[p][:] = 0.0
        action[p][:] = 0.0
        # as in __main__: target = current position of the joints (without jerking default_angles)
        target[p] = d.qpos[R[p]["qpos_adrs"]].copy()

    # 4) Resetting Walk Timers/turning
    if 'WALK_UNTIL' in globals():
        for p in robots: WALK_UNTIL[p] = 0.0
    if 'TURN_UNTIL' in globals():
        for p in robots: TURN_UNTIL[p] = 0.0
    if 'TURN_TGT' in globals():
        for p in robots: TURN_TGT[p] = 0.0
    if 'TURN_ACC' in globals():
        for p in robots: TURN_ACC[p] = 0.0
    if 'TURN_LAST' in globals():
        # recalculate yaw in the current (spawn) condition
        for p in robots:
            mp = R[p]
            qw, qx, qy, qz = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
            TURN_LAST[p] = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    # 5) Clear pose queues
    if 'TASKS' in globals() and 'ACTIVE_TASK' in globals():
        for p in robots:
            TASKS[p].clear()
            ACTIVE_TASK[p] = None

            # --- IMPORTANT: Resetting the controller and policy inputs ---
    for p in robots:
        # completely «clean» policy entries
        try:
            obs[p][:] = 0.0
            action[p][:] = 0.0
            cmd_vec[p][:] = 0.0
            target[p] = d.qpos[R[p]["qpos_adrs"]].copy()
        except Exception:
            pass
        # soft reset policy (without reloading the file)
        if p in POL and hasattr(POL[p], "reset"):
            POL[p].reset(hard=True)
            
        # --- PATCH: reset PD hands
    if ARM_TARGETS is not None:
        if UPPER_HOLD_ZERO:
            ARM_TARGETS[:] = 0.0
        else:
            # maintain current pose after reset (spawn)
            for i, adr in enumerate(ARM_QPOS_ADDRS):
                ARM_TARGETS[i] = d.qpos[adr]
    if ARM_IERR is not None:
        ARM_IERR[:] = 0.0
        

    # (optional) align step counting phases/turning
    for p in robots:
        if 'TURN_ACC' in globals(): TURN_ACC[p] = 0.0
        if 'TURN_TGT' in globals(): TURN_TGT[p] = 0.0
        if 'WALK_UNTIL' in globals(): WALK_UNTIL[p] = 0.0
        if 'TURN_UNTIL' in globals(): TURN_UNTIL[p] = 0.0
        if 'old_sin' in globals(): old_sin[p] = 0.0
    for p in robots:
        idx = HOLD_UPPER_QPOS.get(p)
        if idx is not None and idx.size:
            HOLD_UPPER_KEEP[p] = d.qpos[idx].copy()

    print("Reset done: restored exact spawn state from INIT_QPOS (same as fresh start).")



# =========================
# Motion helpers
# =========================

def _make_pose_task(frame: list[dict], duration: float):
    # frame: [{'name': joint, 'angle': deg}, ...]
    start = {}
    target = {}
    for j in frame:
        jname = j['name']
        if jname not in joint_map:
            continue
        start[jname]  = float(d.qpos[joint_map[jname]])
        target[jname] = math.radians(float(j['angle']))
    t0 = time.time()
    t1 = t0 + max(0.01, float(duration))
    return {'type':'pose','t0':t0,'t1':t1,'start':start,'target':target}

def schedule_action(p: str, item: dict):
  
    if item.get('name') in ('turn','rotate'):
        ang = float(item.get('deg') or item.get('angle') or item.get('yaw_deg') or 0.0)
        spd = float(item.get('spd_deg_s') or item.get('speed_deg_s') or 45.0)
        start_turn_for(p, ang, spd)
        return
    if item.get('name') == 'walk':
        n, ang, v = _parse_walk_params(item)
        start_walk_for(p, n, ang, v)
        return

   
    if 'frame' in item:
        dur = float(item.get('duration', 0.3))
        TASKS[p].append(_make_pose_task(item['frame'], dur))
        return

def _parse_walk_params(obj: dict) -> tuple[int, float, float]:
    """
    Returns (num_steps, dir_deg, speed_mps), supporting synonyms:
      steps/num/count, dir_deg/dir/deg/angle/yaw_deg, spd/speed/vel/v
    Default: dir_deg=0.0, speed=0.5
    Also understands direction={'forward','back','left','right'}.
    """
    # steps
    num = obj.get("num")
    if num is None:
        num = obj.get("steps", obj.get("count", obj.get("step")))
    num = int(num) if num is not None else 0

    # direction
    dir_deg = obj.get("deg")
    if dir_deg is None:
        dir_deg = obj.get("dir", obj.get("deg", obj.get("angle", obj.get("yaw_deg"))))
    if dir_deg is None:
        # text labels (optional)
        direction = str(obj.get("direction", "")).strip().lower()
        if direction in ("fwd", "forward"):
            dir_deg = 0.0
        elif direction in ("back", "backward", "bwd"):
            dir_deg = 180.0
        elif direction in ("left", "l"):
            dir_deg = 90.0
        elif direction in ("right", "r"):
            dir_deg = -90.0
        else:
            dir_deg = 0.0
    dir_deg = float(dir_deg)

    # speed
    spd = obj.get("spd")
    if spd is None:
        spd = obj.get("speed", obj.get("vel", obj.get("v")))
    spd = float(spd) if spd is not None else 0.5

    return max(0, int(num)), float(dir_deg), float(spd)


def move_joints_by_name(joints_list, duration_per_frame=0.3, fps=60):
    """
    Execute motion in one of the formats:
      A) Frames of angles [[8 angles], ...] — as before (for 8 fixed joints).
      B) List of action objects:
         - {"name": "<joint>", "angle": <deg>, "duration": <sec optional>}  # single joint
         - {"frame": [{"name": "<joint>", "angle": <deg>}, ...], "duration": <sec optional>}  # SIMULTANEOUS
         - {"name": "walk", "num": <int>, "deg": <float>, "spd": <float>}
    """
    global last_activity_time, BUSY_SCRIPT
    last_activity_time = time.time()
    BUSY_SCRIPT += 1
    try:
        if not joints_list:
            return

        if isinstance(joints_list, list) and joints_list and isinstance(joints_list[0], dict):
            wants_async = any(it.get('async') for it in joints_list if isinstance(it, dict))
        # We are NOT enabling asynchronous mode just because of the presence of the field 'robot'
            if wants_async:
                for it in joints_list:
                    sel = parse_robot_input(it.get('robot') or it.get('robots')) or [ACTIVE]
                    for p in sel:
                        schedule_action(p, it)
                return


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

        # ===== Variant B (A): list of objects (incl. simultaneous frames via "frame") =====
        for item in joints_list:
            # --- NEW: repeat block ---
            if isinstance(item, dict) and "repeat" in item and "times" in item:
                subactions = item["repeat"]
                n_times = int(item["times"])
                if not isinstance(subactions, list) or n_times <= 0:
                    continue
                for _ in range(n_times):
                    move_joints_by_name(subactions, duration_per_frame, fps)
                continue
            
            # Walking
            if isinstance(item, dict) and item.get("name") == "walk":
                sel = parse_robot_input(item.get("robot") or item.get("robots")) or [ACTIVE]
                n, ang, v = _parse_walk_params(item)
                started = []
                for p in sel:
                    start_walk_for(p, n, ang, v)
                    started.append(p)
                _wait_walks_done(started)
                continue
            
            # Turning
            if isinstance(item, dict) and item.get("name") in ("turn","rotate"):
                sel = parse_robot_input(item.get("robot") or item.get("robots")) or [ACTIVE]

                ang = float(item.get("deg") or item.get("angle") or item.get("yaw_deg") or 0.0)
                spd = float(item.get("spd_deg_s") or item.get("speed_deg_s") or 45.0)
                started = []
                for p in sel:
                    start_turn_for(p, ang, spd)
                    started.append(p)
                _wait_turns_done(started)
                continue
            
            if isinstance(item, dict) and "parallel" in item and isinstance(item["parallel"], list):
            
                started_turn = set()
                started_walk = set()
                for sub in item["parallel"]:
                    sel = parse_robot_input(sub.get('robot') or sub.get('robots')) or [ACTIVE]

                    if sub.get("name") in ("turn", "rotate"):
                        ang = float(sub.get('deg') or sub.get('angle') or sub.get('yaw_deg') or 0.0)
                        spd = float(sub.get('spd_deg_s') or sub.get('speed_deg_s') or 45.0)
                        for p in sel:
                            start_turn_for(p, ang, spd)
                            started_turn.add(p)
                    elif sub.get("name") == "walk":
                        n, ang, v = _parse_walk_params(sub)
                        for p in sel:
                            start_walk_for(p, n, ang, v)
                            started_walk.add(p)

                    else:
                    
                        move_joints_by_name([sub], duration_per_frame, fps)
                if started_turn:
                    _wait_turns_done(sorted(started_turn))
                if started_walk:
                    _wait_walks_done(sorted(started_walk))
                continue

            # Frame with several joints — SIMULTANEOUS
            if isinstance(item, dict) and "frame" in item:
                joints = item["frame"]
                if not isinstance(joints, list) or not joints:
                    print(f"Empty or invalid 'frame' in item: {item}")
                    continue

                # divide by the top/rest
                upper_list, other_list = [], []
                for j in joints:
                    raw = str(j.get("name",""))
                    jname = _resolve_joint_name(raw) or raw
                    if jname in ARM_NAME_TO_IDX:
                        upper_list.append({"name": jname, "angle": j["angle"]})
                    else:
                        other_list.append({"name": jname, "angle": j["angle"]})

                duration = float(item.get("duration", duration_per_frame))

                # 4.1 top - through goals PD
                if upper_list:
                    _arm_apply_frame(upper_list, duration, fps=fps)

                # 4.2 «not top» - old direct write logic qpos
                if other_list:
                    target_angles = {}
                    for j in other_list:
                        jname = j["name"]
                        if jname not in joint_map:
                            print(f"[skip] joint name not in model: {jname}")
                            continue
                        deg = float(j["angle"])
                        target_angles[jname] = math.radians(deg)
                        _dbg_joint_apply_direct(jname, deg, joint_map[jname])

                    if target_angles:
                        n_steps = max(1, int(duration * fps))
                        start_angles = {name: d.qpos[joint_map[name]] for name in target_angles.keys()}
                        def smoothstep(t): return t*t*(3-2*t)
                        fps_dt = 1.0 / fps
                        for step in range(n_steps):
                            t = smoothstep((step + 1) / n_steps)
                            for jname, q1 in target_angles.items():
                                q0 = start_angles[jname]
                                d.qpos[joint_map[jname]] = (1 - t) * q0 + t * q1
                            mj.mj_forward(m, d)
                            render_once()
                            time.sleep(fps_dt)
                        _update_hold_for_all()
                continue

            # Regular single joint — backward compatible with previous format
            if isinstance(item, dict) and "name" in item and "angle" in item:
                
                jname = item["name"]
                if jname not in joint_map:
                    print(f"Joint '{jname}' not found.")
                    continue

                target_angle_rad = math.radians(float(item["angle"]))
                qpos_idx = joint_map[jname]
                start_rad = d.qpos[qpos_idx]
                
                print("# Regular single joint — backward compatible with previous format")
                print(jname, " ", target_angle_rad)

                duration = float(item.get("duration", duration_per_frame))
                n_steps = max(1, int(duration * fps))

                for step in range(n_steps):
                    t = smoothstep((step + 1) / n_steps)
                    d.qpos[qpos_idx] = (1 - t) * start_rad + t * target_angle_rad
                    mj.mj_forward(m, d)
                
                    render_once()
                    time.sleep(fps_dt)
                _update_hold_for_all()
        
                continue

            print(f"[WARN] Unknown item format: {item}")
    finally:
        BUSY_SCRIPT = max(0, BUSY_SCRIPT - 1)        

manual_override = False 
HOLD_UPPER_KEEP: dict[str, np.ndarray] = {}
def _dbg_joint_apply_pd(name: str, deg: float, idx_in_arm: int, aid: int, qpos_adr: int):
    print(f"[UPPER/PD] {name}: target={deg:.2f} deg  -> arm_idx={idx_in_arm}  act_id={aid}  qposadr={qpos_adr}")

def _dbg_joint_apply_direct(name: str, deg: float, qpos_adr: int):
    print(f"[DIRECT]  {name}: qpos <- {deg:.2f} deg  @ qposadr={qpos_adr}")

def _update_hold_from_current(p: str):
    idx = HOLD_UPPER_QPOS.get(p)
    if idx is not None and idx.size:
        HOLD_UPPER_KEEP[p] = d.qpos[idx].copy()

def _update_hold_for_all():
    for p in robots:
        _update_hold_from_current(p)
def _one_tick(render=True):
    """One simulation step + updating tasks/policies/states for ALL robots."""
    if _maybe_freeze():
        return
    global counter

    now = time.time()
    
    


    for p in robots:
        if ACTIVE_TASK[p] is None and TASKS[p]:
            ACTIVE_TASK[p] = TASKS[p].pop(0)
        task = ACTIVE_TASK[p]
        if task and task['type'] == 'pose':
            t0, t1 = task['t0'], task['t1']
            if t1 <= t0:
                alpha = 1.0
            else:
                u = (now - t0) / (t1 - t0)
                u = max(0.0, min(1.0, u))
                alpha = u*u*(3-2*u)
            for jname, q1 in task['target'].items():
                idx = joint_map.get(jname)
                if idx is not None:
                    q0 = task['start'][jname]
                    d.qpos[idx] = (1.0 - alpha) * q0 + alpha * q1
            d.qvel[:] = 0.0
            d.qacc[:] = 0.0
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            if now >= t1:
                _update_hold_from_current(p)   # remember the actual final pose of the top
                ACTIVE_TASK[p] = None

        # --- KEEP THE TOP AT ZERO IF THE ROBOT HAS NO ACTIVE POSE ---
    if HOLD_UPPER_ZERO:
        for p in robots:
            if (ACTIVE_TASK[p] is not None) or (BUSY_SCRIPT > 0):
                continue
            qpos_idx = HOLD_UPPER_QPOS.get(p)
            dof_idx  = HOLD_UPPER_DOF.get(p)
            if qpos_idx is not None and qpos_idx.size:
                # keep the last pose shown, and not 0
                if p in HOLD_UPPER_KEEP:
                    d.qpos[qpos_idx] = HOLD_UPPER_KEEP[p]
            if dof_idx is not None and dof_idx.size:
                d.qvel[dof_idx] = 0.0

            
    mj.mj_forward(m, d)

    # 2) PD joints
    # 2) PD joints (LEGS/policy corps)
    for p in robots:
        mp = R[p]
        if manual_override or (ACTIVE_TASK[p] and ACTIVE_TASK[p]['type'] == 'pose'):
            d.ctrl[mp["act_ids"]] = 0.0
            target[p] = d.qpos[mp["qpos_adrs"]].copy()
            continue
        qj  = d.qpos[mp["qpos_adrs"]]
        dqj = d.qvel[mp["dof_adrs"]]
        tau = pd_control(target[p], qj, kps, np.zeros_like(kds), dqj, kds)
        d.ctrl[mp["act_ids"]] = tau

    # === PD TOP controller - ONCE PER TIC (outside the cycle p) ===
    if ARM_TARGETS is not None and ARM_TARGETS.size:
        q  = np.array([d.qpos[i] for i in ARM_QPOS_ADDRS], dtype=np.float32)
        dq = np.array([d.qvel[i] for i in ARM_QVEL_ADDRS], dtype=np.float32)
        e  = (ARM_TARGETS - q)
        if ARM_IERR is not None and ARM_KIS is not None:
            ARM_IERR[:] += e * simulation_dt
            np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP, out=ARM_IERR)
            i_term = ARM_KIS * ARM_IERR
        else:
            i_term = 0.0
        tau_arm = ARM_KPS * e + ARM_KDS * (-dq) + i_term
        for k, aid in enumerate(ARM_ACT_IDS):
            d.ctrl[aid] = float(tau_arm[k])


    # 3) Step phys
    mj.mj_step(m, d)
    counter += 1

  # 4) Timers/flags of walking/turning on time
    now = time.time()
    for p in robots:
        if WALK_UNTIL[p] and now >= WALK_UNTIL[p]:
            WALK_UNTIL[p] = 0.0
            cmd_vec[p][:] = 0.0
        if TURN_UNTIL[p] and now >= TURN_UNTIL[p]:
            TURN_UNTIL[p] = 0.0
            cmd_vec[p][:] = 0.0
#5) Policy/observation update + Rotation angle accumulation
    if counter % control_decimation == 0:
        def _wrap_to_pi(a): return (a + math.pi) % (2*math.pi) - math.pi
        def _yaw(qw,qx,qy,qz): return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

        na = 12
        for p in robots:
            mp   = R[p]
            quat = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
            omega= d.qvel[mp["base_qvel_adr"]+3 : mp["base_qvel_adr"]+6]
            qj   = d.qpos[mp["qpos_adrs"]]
            dqj  = d.qvel[mp["dof_adrs"]]

          
            if TURN_TGT[p] > 0.0:
                inc = _wrap_to_pi(_yaw(*quat) - TURN_LAST[p])
                TURN_ACC[p] += abs(inc)
                TURN_LAST[p] = _yaw(*quat)
                if TURN_ACC[p] >= TURN_TGT[p]:
                    cmd_vec[p][:] = 0.0
                    TURN_TGT[p] = 0.0
                    TURN_ACC[p] = 0.0

            qj_scaled  = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav       = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = GAIT_PERIOD
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2*math.pi*phase); cos_phase = math.cos(2*math.pi*phase)
            # Completion strictly according to the number of steps (zero transition sin: - → +)
            if WALK_GOAL[p] > 0 and (sin_phase >= 0.0 and old_sin[p] < 0.0):
                WALK_COUNT[p] += 1
                print(f"[WALK] {p[:-1]} step {WALK_COUNT[p]} / {WALK_GOAL[p]} done")
                if WALK_COUNT[p] >= WALK_GOAL[p]:
                    print(f"[WALK] {p[:-1]} walk finished ({WALK_COUNT[p]} steps).")
                    cmd_vec[p][:] = 0.0          # stop walking command
                    WALK_GOAL[p]  = 0
                    WALK_COUNT[p] = 0
                    WALK_UNTIL[p] = 0.0          # just in case
            # update the tracking sine
            old_sin[p] = sin_phase
            o = obs[p]
            o[:3] = omega_scaled
            o[3:6] = grav
            o[6:9] = cmd_vec[p] * cmd_scale
            o[9:9+na] = qj_scaled
            o[9+na:9+2*na] = dqj_scaled
            o[9+2*na:9+3*na] = action[p]
            o[9+3*na:9+3*na+2] = np.array([sin_phase, cos_phase])
            if manual_override or (ACTIVE_TASK[p] and ACTIVE_TASK[p]['type'] == 'pose'):
                continue
            act = POL[p].act(o)
            action[p][:] = act
            target[p]    = act * action_scale + default_angles
    
    _update_camera()
    # 6) Render
    if render:
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()


def _wait_turns_done(sel, timeout=None):
    t0 = time.time()
    while True:
       
        if all(TURN_TGT.get(p, 0.0) == 0.0 for p in sel):
            return
        if timeout and (time.time() - t0) > timeout:
            print("[WARN] wait_turns timeout"); return
        _one_tick(render=True)  

def _wait_walks_done(sel, timeout=None):
    t0 = time.time()
    while True:
        now = time.time()
        if all(WALK_GOAL.get(p, 0) == 0 for p in sel):
            return
        if timeout and (now - t0) > timeout:
            print("[WARN] wait_walks timeout"); return
        _one_tick(render=True)




def start_turn_for(p: str, angle_deg: float, spd_deg_s: float = 45.0):
    global last_activity_time
    last_activity_time = time.time()
    direction = 1.0 if angle_deg >= 0 else -1.0
    yaw_rate = direction * math.radians(abs(spd_deg_s))


    cmd_vec[p][:] = [0.0, 0.0, yaw_rate]

 
    mp = R[p]
    qw, qx, qy, qz = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
    def _yaw(qw,qx,qy,qz):
        return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    TURN_LAST[p] = _yaw(qw,qx,qy,qz)
    TURN_ACC[p]  = 0.0
    TURN_TGT[p]  = abs(math.radians(angle_deg))



def start_walk_for(p: str, num: int, dir_deg: float, spd: float):
    global last_activity_time
    last_activity_time = time.time()
    dir_rad = math.radians(dir_deg)
    vx = spd * math.cos(dir_rad); vy = spd * math.sin(dir_rad)
    cmd_vec[p][:] = [vx, vy, 0.0]

    WALK_GOAL[p]  = max(1, int(num))
    WALK_COUNT[p] = 0
    WALK_UNTIL[p] = 0.0  # the timer is no longer needed; you can leave it as fail-safe, for example. *2*GAIT_PERIOD



def start_new_steps_for(p: str, num: int, dir_deg: float, speed: float):
    global last_activity_time
    last_activity_time = time.time()
    dir_rad = math.radians(dir_deg)
    vx = speed * math.cos(dir_rad)
    vy = speed * math.sin(dir_rad)
    cmd_vec[p][:] = [vx, vy, 0.0]

def run_walk_blocking_for(p: str, num: int, dir_deg: float, spd: float):
    start_walk_for(p, num, dir_deg, spd)
    _wait_walks_done([p])

def run_turn_blocking_for(p: str, angle_deg: float, spd_deg_s: float = 45.0):
    start_turn_for(p, angle_deg, spd_deg_s)
    _wait_turns_done([p])


def start_new_steps(num, dir_deg, speed, is_extra=False):
    """
    Start a new walking sequence:
      - num: number of steps
      - dir_deg: direction in degrees (0° = +X, 90° = +Y)
      - speed: linear speed (m/s)
      - is_extra: internal flag used when auto-balancing
    """
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




# =========================
# GLFW input callbacks
# =========================
ACTIVE = "r1_" 
def key_callback(window_, key, scancode, action, mods):
    pressed = (action == glfw.PRESS or action == glfw.REPEAT)
    global ACTIVE
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
    

    # 'O' — example placeholder: do nothing (empty list)
    elif key == glfw.KEY_O and pressed:
        list_unit = []
        move_joints_by_name(list_unit)


    elif key == glfw.KEY_T and pressed:
        try:
            ang = float(input("Turn angle deg (+left / -right): "))
            spd = input("Speed deg/s [45]: ").strip()
            spd = float(spd) if spd else 45.0
        except Exception:
            print("Invalid input."); return

        sel = ask_robots(prompt_default_active=True)
        # Let's update ACTIVE, if exactly one is selected
        if len(sel) == 1:
            globals()["ACTIVE"] = sel[0]

        for p in sel:
            start_turn_for(p, ang, spd)




    # 'R' — hard reset
    elif key == glfw.KEY_R and pressed:
        do_reset()

    elif key == glfw.KEY_P and pressed:
        sel = "all"
        cmd_list = open_paste_window()
        if not cmd_list:
            return

        # If there are no instructions inside "robot"/"robots", wrap it in parallel onto the selected ones
        def has_targeting(items):
            return any(isinstance(it, dict) and (("robot" in it) or ("robots" in it))
                    for it in items if isinstance(it, dict))

        if isinstance(cmd_list, list) and not has_targeting(cmd_list):
            wrapped = {"parallel": []}
            for p in sel:
                for it in cmd_list:
                    if isinstance(it, dict):
                        cp = dict(it)
                        cp["robots"] = [p]
                        wrapped["parallel"].append(cp)
                    else:
                        wrapped["parallel"].append(it)
            cmd_list = [wrapped]

        move_joints_by_name(cmd_list)

  
    elif key == glfw.KEY_C and pressed:
        try:
            num = int(input("Number of steps (0=cancel): "))
        except Exception:
            print("Invalid number."); return
        if num <= 0:
            print("0 steps. Robot will stay.")
            return

        try:
            dir_deg = float(input("Direction deg (0=+X, 90=+Y): "))
            spd = float(input("Speed m/s: "))
        except Exception:
            print("Invalid direction/speed."); return

        sel = ask_robots(prompt_default_active=True)
        if len(sel) == 1:
            globals()["ACTIVE"] = sel[0]

        for p in sel:
            start_walk_for(p, num, dir_deg, spd)


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

        
def _update_camera(dt=1/2):
    # speed can be taken from camera_speed, so as not to duplicate the number
    v = camera_speed if 'camera_speed' in globals() else 0.8

    a = np.deg2rad(cam.azimuth)

    # correct basis vectors in the plane XY
    fwd   = np.array([np.cos(a), np.sin(a), 0.0])       # where the «» camera is looking in azimuth
    right = np.array([np.sin(a), -np.cos(a), 0.0])      # to the right of the direction of view

    dv = np.zeros(3)
    if movement.get("forward"):  dv += fwd
    if movement.get("backward"): dv -= fwd
    if movement.get("left"):     dv -= right
    if movement.get("right"):    dv += right
    if movement.get("rise"):     dv += np.array([0.0, 0.0, 1.0])
    if movement.get("fall"):     dv -= np.array([0.0, 0.0, 1.0])

    if np.any(dv):
        cam.lookat = cam.lookat + (dv / np.linalg.norm(dv)) * v * dt

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
   
    parser.add_argument("--xml", default="scene_twins_3room", help="dir with JSON")
    args = parser.parse_args()

    scene = args.xml

    # File paths
    name_ = "h1_2"
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path = f"unitree_rl_gym/resources/robots/{name_}/{scene}.xml" # with objects
    #xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene_zero.xml"

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

    # (1) Build joint name -> qpos index map + compact overview
    joint_map = {}

    # Build dynamic ALLOWED_JOINTS from the model (hinge/slide only)
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    print(f"\nAllowed joints for commands ({len(ALLOWED_JOINTS)}):")
    print(", ".join(sorted(ALLOWED_JOINTS)))
    print()

    # Build dynamic frame joint order (index -> name)
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

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

    def _jid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)

    def _aid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)

    # FIX: 12 DOF legs with knees and correct order
    JOINTS = [
        "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]

    def build_robot_maps(prefix: str):
      
        base_candidates = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
                if nm.startswith(prefix):
                    base_candidates.append(j)
        assert base_candidates, f"FREE joint for prefix '{prefix}' not found"
        jid_base = base_candidates[0]

        base_qpos_adr = m.jnt_qposadr[jid_base]
        base_qvel_adr = m.jnt_dofadr[jid_base]

        jids      = np.array([_jid(prefix + j) for j in JOINTS], dtype=int)
        qpos_adrs = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        dof_adrs  = np.array([m.jnt_dofadr[j]  for j in jids], dtype=int)
        act_ids   = np.array([_aid(prefix + j) for j in JOINTS], dtype=int)

        return {
            "base_qpos_adr": base_qpos_adr,
            "base_qvel_adr": base_qvel_adr,
            "qpos_adrs": qpos_adrs,
            "dof_adrs":  dof_adrs,
            "act_ids":   act_ids,
        }

    R = {
        "r1_": build_robot_maps("r1_"),
        "r2_": build_robot_maps("r2_"),
        "r3_": build_robot_maps("r3_"),
    }
    _build_upper_hold_maps() 
    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)
    
    
    POLICY_PATHS = {
    "r1_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",
    "r2_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",
    "r3_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",
    }
    POL = {p: PolicyWrapper(POLICY_PATHS[p]) for p in ("r1_","r2_","r3_") if p in R}

    # =========================
    # Walking / balance control state
    # =========================
    steps_needed = 0
    steps_done = 0
    cmd = np.zeros(3, dtype=np.float32)

    robots = [p for p in ("r1_","r2_","r3_") if p in R]
    if not robots:
        robots = ["r1_"]

    na = 12
    action   = {p: np.zeros(na, np.float32) for p in robots}
    target   = {p: d.qpos[R[p]["qpos_adrs"]].copy() for p in robots}  
    obs      = {p: np.zeros(num_obs, np.float32) for p in robots}

    cmd_vec  = {p: np.zeros(3, np.float32) for p in robots}
    old_sin  = {p: 0.0 for p in robots}

    GAIT_PERIOD = 0.8
    WALK_UNTIL = {p: 0.0 for p in robots}
    TURN_UNTIL = {p: 0.0 for p in robots}

    TURN_ACC   = {p: 0.0 for p in robots}  
    TURN_LAST  = {p: 0.0 for p in robots} 
    TURN_TGT   = {p: 0.0 for p in robots}   

    WALK_GOAL  = {p: 0 for p in robots}   # how many steps are left
    WALK_COUNT = {p: 0 for p in robots}   # how much has been done
    TASKS       = {p: [] for p in robots}  
    ACTIVE_TASK = {p: None for p in robots} 


    short_extra_done = False
    last_dir_deg = 0.0

    counter = 0

    # =========================
    # Main simulation loop
    # =========================
    while not glfw.window_should_close(window):
        _one_tick(render=True)
    mj.mjv_freeScene(scene)
    mj.mjr_freeContext(context)
    glfw.terminate()
    print("Window closed, exiting.")
