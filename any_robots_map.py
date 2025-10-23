
# =========================
import time, math, json, ast, re
from typing import List, Dict, Any
from pathlib import Path
import argparse
import numpy as np
import yaml
import torch
import tkinter as tk
from tkinter import filedialog
import re
import mujoco as mj
from mujoco.glfw import glfw

from llm_providers import build_llm
import settings
from sim_phrases import load_csv as load_cmds_csv, compute_similarities, DEFAULT_MODEL
from collections import deque

# =========================

# --- UPPER-BODY PD control (replaces the old ARM_* block) ---
UPPER_HOLD_ZERO = True   # <— key: zero angles by default

UPPER_JOINTS: list[str] = []     # dynamic names on joints
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
ARM_I_CLAMP = 0.7  # integral limit

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


_LEG_TOKENS   = ("hip", "knee", "ankle", "toe", "foot", "leg")
_FINGER_TOKENS= ("thumb", "index", "middle", "ring", "pinky", "finger")
_UPPER_TOKENS = ("shoulder","elbow","wrist","hand","torso","waist","spine","chest","neck","head") + _FINGER_TOKENS

def _is_upper_joint(name: str) -> bool:
    """True for  upper h1/h1_2 (L_/R_*)."""
    nm = name.lower()

    # to zero prefix
    nm = re.sub(r"^r\d+_", "", nm)
    if nm.endswith("_joint"):
        nm = nm[:-6]

    # fast leg filter
    if any(tok in nm for tok in _LEG_TOKENS):
        return False

    # clear signs of the top (including thumb/index/…)
    if any(tok in nm for tok in _UPPER_TOKENS):
        return True

    if nm.startswith(("l_", "r_")):
        rest = nm[2:]
        if any(tok in rest for tok in _FINGER_TOKENS + ("wrist","hand","elbow","shoulder")):
            return True

    return False


def _suggest_kp(name: str) -> float:
    
    if name == "torso_joint": return 40.0
    if "shoulder" in name:   return 40.0
    if "elbow" in name:      return 30.0
    if "wrist" in name:      return 12.0
    
    return 4.0

def setup_arm_pd(zero_pose: bool = True):
    """Bind ALL top joints and initialize PD-goals.
    If the actuator does not have name==joint_name, searching by trnid (target=this joint)."""
    global UPPER_JOINTS, ARM_NAME_TO_IDX, ARM_JIDS, ARM_ACT_IDS
    global ARM_QPOS_ADDRS, ARM_QVEL_ADDRS, ARM_TARGETS, ARM_KPS, ARM_KDS
    global ARM_KIS, ARM_IERR
    ARM_NAME_TO_IDX.clear()
    ARM_JIDS.clear(); ARM_ACT_IDS.clear(); ARM_QPOS_ADDRS.clear(); ARM_QVEL_ADDRS.clear()

    # 1) collect the names of the top hinge-joints
    UPPER_JOINTS = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
        if _is_upper_joint(nm):
            UPPER_JOINTS.append(nm)

    # 2) build a fallback map: joint_id -> actuator_id (For actuator With JOINT target)
    trn  = np.array(m.actuator_trnid).reshape(m.nu, 2) if m.nu > 0 else np.zeros((0, 2), dtype=int)
    trnt = np.array(m.actuator_trntype) if m.nu > 0 else np.zeros((0,), dtype=int)
    jointid_to_act = {}
    for i in range(m.nu):
        if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
            jid_target = int(trn[i, 0])
            # The first one found is considered the main one
            jointid_to_act.setdefault(jid_target, i)

    # 3) tie joint ↔ actuator (first by name, otherwise by trnid)
    names_in_order = []
    for name in UPPER_JOINTS:
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"[UPPER] joint '{name}' not found — skip")
            continue

        # trying out the name
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)
        # if there is no name, take the target one joint'at
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

    # 4) targets: either zeros or current values
    if zero_pose:
        ARM_TARGETS = np.zeros(n, dtype=np.float32)
        log_zeroed_upper()
    else:
        ARM_TARGETS = np.array([d.qpos[i] for i in ARM_QPOS_ADDRS], dtype=np.float32)

    # 5) Kp/Kd in index order
    ARM_KPS = np.array([_suggest_kp(nm) for nm in names_in_order], dtype=np.float32)
    ARM_KDS = np.clip(ARM_KPS * 0.05, 0.2, None).astype(np.float32)
    ARM_KIS = np.clip(ARM_KPS * 0.02, 0.0, 1.0).astype(np.float32)  # soft integral
    ARM_IERR = np.zeros_like(ARM_KPS, dtype=np.float32)

    print(f"[UPPER] PD enabled for {n} joints:", names_in_order)


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
    global steps_done, steps_needed
    start_new_steps(num, dir_deg, spd)
    while steps_done < steps_needed and not glfw.window_should_close(window):
        if not step_once(render=True):
            break
        # front step detection sin(phase)
        if PHASE_SIN >= 0 and PHASE_PREV_SIN < 0:
            if STANCE_HOLD:
                # (how was it with you; you can leave the logic stance)
                pass
            else:
                steps_done += 1
                print(f"Got step #{steps_done} / {steps_needed}")
                if steps_done >= steps_needed:
                    cmd[:] = 0.0
                    target_dof_pos[:] = default_angles
    cmd[:] = 0.0
    d.ctrl[:] = 0.0


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw around +Z (MuJoCo: Z-up)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
def run_turn_blocking(angle_deg: float, spd_deg_s: float = 45.0):
    stop_stance_hold()
    direction  = 1.0 if angle_deg >= 0 else -1.0
    yaw_rate   = direction * math.radians(abs(spd_deg_s))
    target_abs = math.radians(abs(angle_deg))

    cmd[:] = [0.0, 0.0, yaw_rate]
    qw, qx, qy, qz = d.qpos[3:7]
    last_yaw = _quat_yaw(qw, qx, qy, qz)
    acc = 0.0

    while acc < target_abs and not glfw.window_should_close(window):
        if not step_once(render=True):
            break
        qw, qx, qy, qz = d.qpos[3:7]
        yaw_now = _quat_yaw(qw, qx, qy, qz)
        acc += abs(_wrap_to_pi(yaw_now - last_yaw))
        last_yaw = yaw_now

    cmd[:] = 0.0
    d.ctrl[:] = 0.0



# =========================
# Global variables (extended)
# =========================


# Linting-time placeholders so mypy/linters don’t complain at import time
m = d = policy = None
is_dragging = False
last_cursor_pos = (0.0, 0.0)
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
# ==== MINIMAP (real time, top view) ====
MAP_CELL_M    = 0.05          # 5 cm per pixel (leave as is)
MAP_RADIUS_M  = 3.0           # radius of the visible zone around the robot, m
MAP_SIZE_M    = MAP_RADIUS_M * 2.0   # map window - diameter 6 m
MINIMAP_PX    = 220
MAP_BG        = 230
MAP_OBJ_COL   = 80
MAP_ROBOT_COL = (220, 40, 40)
MAP_ZONE_COL  = (120, 180, 240)  # zone ring color (3m)
MAP_BORDER    = 40

MAP_FOLLOW_YAW = True  # the map is rotated with a robot (False — «north up»)
# === Reactive obstacle avoidance (applies to ANY walking) ===
NAV_AVOID_ENABLE    = True
NAV_LOOKAHEAD       = 0.90   # how many meters do we “look” ahead?
NAV_STOP_DIST       = 0.30   # buffer to the obstacle at which we begin to brake/bypass
NAV_INFLATE_CELLS   = 2      # «robot thickness" in minimap cells
NAV_SEARCH_DEG      = 120    # search fan for an alternative course (±)
NAV_RAYS            = 29     # number of rays in the fan (odd)
NAV_MAX_YAW_RATE    = math.radians(160.0)  # physical rotation speed limit

# occupation cache for several ticks, so as not to build a map every time
_NAV_OCC_CACHE = {"n": 64, "occ": None, "tick": -1000}

# excluded geoms by name/type (floor, infinite planes, etc..)
MAP_EXCLUDE_SUBSTR = ("floor", "ground")
def _world_to_grid(x: float, y: float, n: int) -> tuple[int, int]:
    """
    World point translation (x,y) to minimap pixels:
    - the center of the map coincides with the robot’s position;
    - at MAP_FOLLOW_YAW=True the map is rotated so that the robot is ahead» = up.
    """
    # center of the window (meters from the robot)
    half = MAP_SIZE_M * 0.5

    # robot position and heading
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    qw, qx, qy, qz = d.qpos[3:7]
    yaw = _quat_yaw(qw, qx, qy, qz) if MAP_FOLLOW_YAW else 0.0

    # shift to robot system
    dx, dy = x - rx, y - ry

    # turn to -yaw (world -> «frame»)
    ca, sa = math.cos(-yaw), math.sin(-yaw)
    xr = ca * dx - sa * dy   # forward (robot axis)
    yr = sa * dx + ca * dy   # left (robot axis)

    # projection to pixels:
    #   gx — horizontal (right positive) ← take “robot left”»
    #   gy — vertical (down positive)    ← let's take the robot forward»
    # After flipud() «forward robot" will appear up on the screen.
    gx = int((yr + half) / MAP_CELL_M)
    gy = int((xr + half) / MAP_CELL_M)
    return gx, gy

def _draw_ring(img: np.ndarray, cx: int, cy: int, r: int, rgb=(0,0,0), thickness: int = 2):
    # thin radius ring r (thickness in pixels)
    for t in range(0, 360):
        ang = math.radians(t)
        for k in range(thickness):
            xx = cx + int((r - k) * math.cos(ang))
            yy = cy + int((r - k) * math.sin(ang))
            _safe_set_px(img, xx, yy, rgb)


def _safe_set_px(img: np.ndarray, x: int, y: int, rgb: tuple[int,int,int] | int):
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        if isinstance(rgb, tuple):
            img[y, x, :] = rgb
        else:
            img[y, x, :] = rgb

def _draw_disc(img: np.ndarray, cx: int, cy: int, r: int, rgb=(0,0,0)):
    r2 = r*r
    for dy in range(-r, r+1):
        yy = cy + dy
        dx_max = int((r2 - dy*dy) ** 0.5)
        for dx in range(-dx_max, dx_max+1):
            xx = cx + dx
            _safe_set_px(img, xx, yy, rgb)
def _clamp(x, lo, hi): 
    return lo if x < lo else hi if x > hi else x
def _ray_first_hit_distance_world(x0: float, y0: float, ang: float,
                                  max_dist: float, n: int = 48,
                                  skip_first: float = 0.05,
                                  mat: np.ndarray | None = None) -> float | None:
    """
    Beam from (x0,y0) at an angle ang on max_dist.
    Returns the distance to the FIRST obstacle (m) or None.
    skip_first — how many meters from the start we ignore (so as not to see ourselves).
    """
    if mat is None:
        mat = minimap_matrix(n)
    cell_m = MAP_SIZE_M / n
    step = cell_m * 0.5  # half a cell - reliable
    t = max(step, skip_first)
    while t <= max_dist:
        x = x0 + t * math.cos(ang)
        y = y0 + t * math.sin(ang)
        mx, my = _world_to_mat_idx(x, y, n)
        if 0 <= mx < n and 0 <= my < n:
            if int(mat[my, mx]) == 1:
                return t
        t += step
    return None


def _free_distance_in_direction(x0: float, y0: float, ang: float,
                                max_dist: float, n: int = 48,
                                mat: np.ndarray | None = None) -> float:
    """
    How many meters are free in the direction ang (0..max_dist), on the minimap grid.
    """
    hit = _ray_first_hit_distance_world(x0, y0, ang, max_dist, n=n, skip_first=0.0, mat=mat)
    return max_dist if hit is None else max(0.0, hit)
# ---------- GRID A* + FOLLOW ----------

def _mat_idx_to_world(mx: int, my: int, n: int) -> tuple[float, float]:
    """Inversion _world_to_mat_idx: cell center (mx,my) -> world (x,y)."""
    half = MAP_SIZE_M * 0.5
    cell = MAP_SIZE_M / float(n)

    # matrix minimap_matrix — upside down Y (flipud), roll back:
    gx = mx
    gy = (n - 1) - my

    xr = (gy + 0.5) * cell - half   # forward robot, m
    yr = (gx + 0.5) * cell - half   # left robot, m

    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    yaw = _quat_yaw(*d.qpos[3:7]) if MAP_FOLLOW_YAW else 0.0
    ca, sa = math.cos(yaw), math.sin(yaw)
    dx = ca * xr - sa * yr
    dy = sa * xr + ca * yr
    return rx + dx, ry + dy


def _inflate_occupancy(mat: np.ndarray, r: int) -> np.ndarray:
    """Inflate obstacles on r cells (Manhattan circle). 1/2 → let, 0 → free."""
    if r <= 0:
        return (mat == 1).astype(np.uint8)
    n = mat.shape[0]
    obs = (mat == 1)
    out = np.zeros_like(mat, dtype=np.uint8)
    idx = np.argwhere(obs)
    for (y, x) in idx:
        y0 = max(0, y - r); y1 = min(n - 1, y + r)
        x0 = max(0, x - r); x1 = min(n - 1, x + r)
        out[y0:y1+1, x0:x1+1] = 1
    # robot (2) is considered free
    out[(mat == 2)] = 0
    return out


def _neighbors8(x: int, y: int):
    for dx, dy in _NEIGH8:
        yield x + dx, y + dy


def _a_star(start: tuple[int,int], goal: tuple[int,int], occ: np.ndarray,
            avoid_corner_cut: bool = True) -> list[tuple[int,int]] | None:
    """A* on the grid occ (1 — wall). 8-connectivity, we prohibit “cutting corners” near walls."""
    n = occ.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < n and 0 <= sy < n and 0 <= gx < n and 0 <= gy < n):
        return None
    if occ[sy, sx] == 1:
        return None
    if occ[gy, gx] == 1:
        return None

    import heapq
    h = lambda x, y: math.hypot(x - gx, y - gy)

    openq = []
    heapq.heappush(openq, (h(sx, sy), 0.0, (sx, sy)))
    came = { (sx, sy): None }
    gsc  = { (sx, sy): 0.0 }

    while openq:
        _, gc, (x, y) = heapq.heappop(openq)
        if (x, y) == (gx, gy):
            # restore the path
            path = [(x, y)]
            while came[(x, y)] is not None:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        for nx, ny in _neighbors8(x, y):
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if occ[ny, nx] == 1:
                continue
            # ban corner-cut: if diagonal, two adjacent cardinals must be free
            if avoid_corner_cut and nx != x and ny != y:
                if occ[y, nx] == 1 or occ[ny, x] == 1:
                    continue

            step = math.hypot(nx - x, ny - y)
            ng = gc + step
            if ng + 1e-9 < gsc.get((nx, ny), 1e18):
                gsc[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                heapq.heappush(openq, (ng + h(nx, ny), ng, (nx, ny)))
    return None


def _segment_is_clear(x0: float, y0: float, x1: float, y1: float,
                      n: int, occ: np.ndarray,
                      inflate_cells: int = 0,
                      skip_first_m: float = 0.02) -> bool:
    """Checking visibility around the world based on a grid occ."""
    # we use an existing trace around the world → net
    cell = MAP_SIZE_M / float(n)
    step = max(cell * 0.5, 0.02)
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return True
    t = max(skip_first_m, step)
    while t <= dist:
        x = x0 + (x1 - x0) * (t / dist)
        y = y0 + (y1 - y0) * (t / dist)
        mx, my = _world_to_mat_idx(x, y, n)
        if 0 <= mx < n and 0 <= my < n:
            # local inflation on check, so as not to scratch the side
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    xx, yy = mx + dx, my + dy
                    if 0 <= xx < n and 0 <= yy < n and int(occ[yy, xx]) == 1:
                        return False
        t += step
    return True


def _simplify_world_path(points_world: list[tuple[float,float]],
                         n: int, occ: np.ndarray,
                         inflate_on_los: int = 1) -> list[tuple[float,float]]:
    """Simplify the polyline: discard extra points if the segment is visible."""
    if len(points_world) <= 2:
        return points_world[:]
    out = [points_world[0]]
    i = 0
    while i < len(points_world) - 1:
        j = len(points_world) - 1
        # from far to near we are looking for the farthest visible
        while j > i + 1:
            if _segment_is_clear(points_world[i][0], points_world[i][1],
                                 points_world[j][0], points_world[j][1],
                                 n, occ, inflate_cells=inflate_on_los):
                break
            j -= 1
        out.append(points_world[j])
        i = j
    return out


def go_to_xy_blocking(tx: float, ty: float, speed: float = 0.22,
                      stop: float = 0.30, slow_r: float = 0.70,
                      yaw_kp: float = 2.0, yaw_max_deg: float = 120.0,
                      n_map: int = 64,
                      inflate_cells: int = 2,          # «thickness" of the robot in cages
                      replan_every: int = 90,          # sometimes we reschedule (tika)
                      los_inflate: int = 1):           # how much to expand when checking visibility
    """
    Planning A* on the minimap and following simplified waypoints.
    Works in world coordinates, but the plan is built in a grid minimap_matrix(n_map).
    """
    # 1) snapshot of the map and the “inflated” occupation
    mat_raw = minimap_matrix(n_map)
    occ = _inflate_occupancy(mat_raw, inflate_cells)

    # 2) start/finish of the cage (we allow the robot to be free)
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    sx, sy = _world_to_mat_idx(rx, ry, n_map)
    gxw, gyw = snap_goal_to_free(tx, ty, n=n_map, inflate=1)
    gx, gy = _world_to_mat_idx(gxw, gyw, n_map)

    # 3) A*
    path = _a_star((sx, sy), (gx, gy), occ)
    if not path:
        print("[goto/a*] no path")
        return

    # 4) → world + simplification by direct lines of sight
    pts_world = [_mat_idx_to_world(x, y, n_map) for (x, y) in path]
    pts_world = _simplify_world_path(pts_world, n_map, occ, inflate_on_los=los_inflate)

    print(f"[goto/a*] grid len={len(path)}  waypoints={len(pts_world)}")
    # 5) waypoint tracking
    stop_stance_hold(); clear_abort()
    yaw_max = math.radians(yaw_max_deg)

    wp_i = 0
    last_plan_counter = counter
    dist0 = max(1e-6, math.hypot(tx - rx, ty - ry))

    while not glfw.window_should_close(window) and not _should_abort():
        # target - current waypoint or final point
        gx, gy = pts_world[wp_i] if wp_i < len(pts_world) else (gxw, gyw)

        rx, ry = float(d.qpos[0]), float(d.qpos[1])
        dx, dy = gx - rx, gy - ry
        dist = math.hypot(dx, dy)
        if dist <= stop:
            wp_i += 1
            if wp_i >= len(pts_world):
                break
            continue

        yaw   = _quat_yaw(*d.qpos[3:7])
        th    = math.atan2(dy, dx)
        yaw_e = _wrap_to_pi(th - yaw)

        v = speed * min(1.0, dist / max(slow_r, 1e-3))
        v *= max(0.0, math.cos(abs(yaw_e)))

        vxb = v * math.cos(yaw_e)
        vyb = v * math.sin(yaw_e)
        yaw_rate = _clamp(yaw_kp * yaw_e, -yaw_max, yaw_max)
        cmd[:] = [vxb, vyb, yaw_rate]
        step_once(render=True)

        # sometimes we reschedule on a fresh map (if the world is dynamic)
        if (counter - last_plan_counter) >= replan_every:
            last_plan_counter = counter
            mat_raw = minimap_matrix(n_map)
            occ = _inflate_occupancy(mat_raw, inflate_cells)
            sx, sy = _world_to_mat_idx(rx, ry, n_map)
            gx, gy = _world_to_mat_idx(gxw, gyw, n_map)
            path = _a_star((sx, sy), (gx, gy), occ)
            if path:
                pts_world = _simplify_world_path([_mat_idx_to_world(x, y, n_map) for (x, y) in path],
                                                 n_map, occ, inflate_on_los=los_inflate)
                wp_i = 0  # we start with the one closest to us
                # move the index to the nearest visible one
                for k in range(len(pts_world)-1):
                    if _segment_is_clear(rx, ry, pts_world[k][0], pts_world[k][1], n_map, occ, inflate_cells=los_inflate):
                        wp_i = k
                    else:
                        break

        if counter % 20 == 0:
            to_final = math.hypot(tx - rx, ty - ry)
            prog = 100 * (1 - min(1.0, to_final / dist0))
            print(f"\r[nav/a*] {prog:3.0f}% | pos=({rx:+.2f},{ry:+.2f}) | wp {wp_i+1}/{len(pts_world)}",
                  end="", flush=True)

    cmd[:] = 0.0
    d.ctrl[:] = 0.0
    print("\n[goto/a*] done")

def _minimap_collect_geom_ids() -> list[int]:
    global _ROBOT_BODY_TREE
    if _ROBOT_BODY_TREE is None:
        _ROBOT_BODY_TREE = _build_robot_body_tree()

    ids: list[int] = []
    for gid in range(m.ngeom):
        gtype = m.geom_type[gid]
        if gtype == mj.mjtGeom.mjGEOM_PLANE:
            continue

        # let's skip the robot geoms
        body_id = int(m.geom_bodyid[gid])
        if _ROBOT_BODY_TREE and body_id in _ROBOT_BODY_TREE:
            continue

        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gid) or ""
        if any(s in name.lower() for s in MAP_EXCLUDE_SUBSTR):
            continue

        ids.append(gid)
    return ids



def _draw_filled_polygon(img: np.ndarray, pts: list[tuple[int,int]], rgb=(80,80,80)):
    if len(pts) < 3:
        return
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmin, xmax = max(min(xs), 0), min(max(xs), img.shape[1]-1)
    ymin, ymax = max(min(ys), 0), min(max(ys), img.shape[0]-1)
    for yy in range(ymin, ymax + 1):
        inter = []
        for i in range(len(pts)):
            x1, y1 = pts[i]; x2, y2 = pts[(i+1) % len(pts)]
            if (y1 <= yy < y2) or (y2 <= yy < y1):
                t = (yy - y1) / ((y2 - y1) + 1e-9)
                inter.append(int(round(x1 + t * (x2 - x1))))
        inter.sort()
        for k in range(0, len(inter), 2):
            xL = max(xmin, inter[k])
            xR = min(xmax, inter[k+1] if k+1 < len(inter) else inter[k])
            if xL <= xR:
                img[yy, xL:xR+1, :] = rgb

def _box_corners_world_xy(gid: int) -> list[tuple[float,float]]:
    """4 track angle mjGEOM_BOX in the world plane XY (ordered without self-intersections)."""
    sz = np.array(m.geom_size[gid], dtype=float)  # (sx, sy, sz) — HALF sizes
    R  = np.array(d.geom_xmat[gid]).reshape(3,3)  # geom world orientation
    p  = np.array(d.geom_xpos[gid], dtype=float)  # center

    # local semi-axes X,Y, projected into XY peace
    ux = R[:2, 0] * sz[0]
    vy = R[:2, 1] * sz[1]
    c  = p[:2]

    # 4 peaks clockwise
    return [
        (c[0] + ux[0] + vy[0], c[1] + ux[1] + vy[1]),
        (c[0] - ux[0] + vy[0], c[1] - ux[1] + vy[1]),
        (c[0] - ux[0] - vy[0], c[1] - ux[1] - vy[1]),
        (c[0] + ux[0] - vy[0], c[1] + ux[1] - vy[1]),
    ]

def _world_poly_to_grid(poly_xy: list[tuple[float,float]], n: int) -> list[tuple[int,int]]:
    return [_world_to_grid(x, y, n) for (x, y) in poly_xy]
# --- snap goal to nearest free minimap cell ---------------------------------

def snap_goal_to_free(tx: float, ty: float, n: int = 48, inflate: int = 1) -> tuple[float, float]:
    """
    If the target hits an obstacle, we move it to the nearest free cell.
    Important: minimap_matrix() turns over inside Y, so we clearly take into account here flip.
    Works in BASIC grid resolution (MAP_CELL_M), to avoid scale desynchronization.
    """
    # --- base number of cells, as in _minimap_build_image() ---
    base_n = int(MAP_SIZE_M / MAP_CELL_M)
    base_n = max(32, min(1024, base_n))

    # occupation matrix in the base grid (0—free, 1—obstacle, 2—robot), already flipud
    mat = minimap_matrix(base_n)   # shape=(base_n, base_n), flipped By Y

    # world -> grid (non-inverted mesh "unflipped")
    gx_u, gy_u = _world_to_grid(tx, ty, base_n)
    gx_u = int(np.clip(gx_u, 0, base_n - 1))
    gy_u = int(np.clip(gy_u, 0, base_n - 1))

    # grid(unflipped) -> mat(flippedY)
    mx0, my0 = gx_u, (base_n - 1 - gy_u)

    def _is_free(mx: int, my: int) -> bool:
        """Is the cell free? c taking into account inflation inflate (0/1/2; 2=We consider the robot free)."""
        nloc = mat.shape[0]
        for yy in range(max(0, my - inflate), min(nloc - 1, my + inflate) + 1):
            for xx in range(max(0, mx - inflate), min(nloc - 1, mx + inflate) + 1):
                v = int(mat[yy, xx])
                if v == 1:  # let
                    return False
        return True  # 0 or 2 - ok

    occ0 = int(mat[my0, mx0])
   # print(f"[snap] check world=({tx:.3f},{ty:.3f}) -> grid=({gx_u},{gy_u}) -> mat=({mx0},{my0}) occ={occ0}")

    # already free - we do nothing
    if _is_free(mx0, my0):
    #    print("[snap] already FREE — keep original target")
        return tx, ty

    # BFS according to the matrix (in coordinates mat, those. with a flip over Y)

    seen = np.zeros_like(mat, dtype=bool)
    q = deque([(mx0, my0, 0)])  # (mx, my, rings)
    while q:
        mx, my, rings = q.popleft()
        if not (0 <= mx < base_n and 0 <= my < base_n) or seen[my, mx]:
            continue
        seen[my, mx] = True

        if _is_free(mx, my):
            # mat -> grid(unflipped)
            gx_f = mx
            gy_f = (base_n - 1 - my)

            # grid(unflipped) -> world (inversion _world_to_grid with base cell)
            half = MAP_SIZE_M * 0.5
            cell = MAP_CELL_M
            xr = (gy_f + 0.5) * cell - half   # forward (robot axis)
            yr = (gx_f + 0.5) * cell - half   # left (robot axis)

            rx, ry = float(d.qpos[0]), float(d.qpos[1])
            yaw = _quat_yaw(*d.qpos[3:7]) if MAP_FOLLOW_YAW else 0.0
            ca, sa = math.cos(yaw), math.sin(yaw)
            dx = ca * xr - sa * yr
            dy = sa * xr + ca * yr
            wx, wy = rx + dx, ry + dy

            dist = math.hypot(wx - tx, wy - ty)
          #  print(f"[snap] OCC -> FREE: mat=({mx},{my}) grid=({gx_f},{gy_f}) world=({wx:.3f},{wy:.3f}) "
          #        f"rings={rings}, shift={dist:.3f}m")
            return wx, wy

        # 8-neighbors
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q.append((mx + dx, my + dy, rings + 1))

    print("[snap] WARN: no FREE cell found — keep original target")
    return tx, ty

# ---------------------------------------------------------------------------

def _minimap_build_image() -> np.ndarray:
    """Assembling the minimap picture (RGB uint8), from top to bottom (then we’ll turn it over)."""
    n = int(MAP_SIZE_M / MAP_CELL_M)
    n = max(32, min(1024, n))  # reasonable limits
    img = np.full((n, n, 3), MAP_BG, dtype=np.uint8)

    # frame (1 pixel around the perimeter)
    img[0, :, :] = 0
    img[-1, :, :] = 0
    img[:, 0, :] = 0
    img[:, -1, :] = 0

    # objects (by geomes)
    geom_ids = _minimap_collect_geom_ids()
    for gid in geom_ids:
        x, y = d.geom_xpos[gid][0], d.geom_xpos[gid][1]
        gx, gy = _world_to_grid(x, y, n)

        gtype = m.geom_type[gid]
        sz    = np.array(m.geom_size[gid])

        if gtype == mj.mjtGeom.mjGEOM_BOX and sz.size >= 2:
            poly_xy = _box_corners_world_xy(gid)     # 4 corner in the world
            poly_px = _world_poly_to_grid(poly_xy, n)  # to map pixels
            _draw_filled_polygon(img, poly_px, (MAP_OBJ_COL, MAP_OBJ_COL, MAP_OBJ_COL))
        else:
            r_m = float(np.max(sz[:2])) if sz.size else 0.05
            r_px = max(1, int(r_m / MAP_CELL_M))
            _draw_disc(img, gx, gy, r_px, (MAP_OBJ_COL, MAP_OBJ_COL, MAP_OBJ_COL))

    # robot (position + course arrow)
       # robot is always in the center of the minimap
    cx = n // 2
    cy = n // 2
    _draw_disc(img, cx, cy, 2, MAP_ROBOT_COL)

    # circle "zone 3 m"»
    zone_px = max(1, int(MAP_RADIUS_M / MAP_CELL_M))
    _draw_ring(img, cx, cy, zone_px, MAP_ZONE_COL, thickness=2)

    # course arrow: at MAP_FOLLOW_YAW the map is already rotated according to the course,
    # so we draw an arrow “up” (stable and readable).
    L = 8  # arrow length in cells
    tipx = cx
    tipy = cy - L
    steps = max(abs(tipx - cx), abs(tipy - cy), 1)
    for i in range(steps + 1):
        xx = cx + int((tipx - cx) * i / steps)
        yy = cy + int((tipy - cy) * i / steps)
        _safe_set_px(img, xx, yy, MAP_ROBOT_COL)


    return img
# Let's calculate the set once body_id robot (its subtree)
_ROBOT_BODY_TREE: set[int] | None = None

def _build_robot_body_tree() -> set[int]:
    # looking for free-joint With qposadr==0 — robot base
    base_jid = -1
    for j in range(m.njnt):
        if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and m.jnt_qposadr[j] == 0:
            base_jid = j
            break
    if base_jid < 0:
        return set()

    root_bid = int(m.jnt_bodyid[base_jid])

    # children's table for crawling
    parent = np.array(m.body_parentid, dtype=int)
    nbody = int(m.nbody)
    children: list[list[int]] = [[] for _ in range(nbody)]
    for b in range(1, nbody):
        p = parent[b]
        if p >= 0:
            children[p].append(b)

    # DFS
    stack = [root_bid]
    out = {root_bid}
    while stack:
        b = stack.pop()
        for c in children[b]:
            if c not in out:
                out.add(c)
                stack.append(c)
    return out

def minimap_matrix(n: int = 48):
    img = _resize_nn(_minimap_build_image(), n, n)  # n×n RGB

    # obstacles - only pixels of object color (MAP_OBJ_COL)
    obj_mask = np.all(img == MAP_OBJ_COL, axis=2)
    rob_mask = np.all(img == MAP_ROBOT_COL, axis=2)

    mat = obj_mask.astype(np.uint8)
    mat[rob_mask] = 2  # the robot is marked with a separate class

    return np.flipud(mat)

def _resize_nn(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    yy = np.linspace(0, h - 1, out_h).astype(np.int32)
    xx = np.linspace(0, w - 1, out_w).astype(np.int32)
    yy_grid, xx_grid = np.meshgrid(yy, xx, indexing="ij")
    return img[yy_grid, xx_grid]

def draw_minimap_overlay(screen_w: int, screen_h: int):
    img = _minimap_build_image()
    img = _resize_nn(img, MINIMAP_PX, MINIMAP_PX)
    img = np.flipud(img).astype(np.uint8)
    rgb_buf = np.ascontiguousarray(img.reshape(-1))

    rect = mj.MjrRect(
        max(0, screen_w - MINIMAP_PX - MAP_BORDER),
        max(0, screen_h - MINIMAP_PX - MAP_BORDER),
        MINIMAP_PX, MINIMAP_PX
    )

    # draw out the window
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, context)

    # Background (without context in this version MuJoCo)
    try:
        mj.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.6)
    except TypeError:
        # just in case - other versions require context
        mj.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.6, context)

    # Minimap pixels (this function always accepts context)
    mj.mjr_drawPixels(rgb_buf, None, rect, context)

# =========================
# Saving helpers
# =========================
# === Wavefront (Lee) on the local minimap ===

_NEIGH8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
_NEIGH4 = [(0,-1),(1,0),(0,1),(-1,0)]

def _world_to_mat_idx(x: float, y: float, n: int) -> tuple[int, int]:
    """
    World (x,y) -> indexes in the matrix minimap_matrix(n).
    IMPORTANT: we use the cage size = MAP_SIZE_M / n, not fixed MAP_CELL_M.
    Taking into account the rotation of the map (MAP_FOLLOW_YAW) And flipud() V minimap_matrix().
    """
    # window center meters from the robot
    half = MAP_SIZE_M * 0.5

    # robot position and heading
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    qw, qx, qy, qz = d.qpos[3:7]
    yaw = _quat_yaw(qw, qx, qy, qz) if MAP_FOLLOW_YAW else 0.0

    # into the robot system
    dx, dy = x - rx, y - ry
    ca, sa = math.cos(-yaw), math.sin(-yaw)
    xr = ca * dx - sa * dy   # "forward" robot
    yr = sa * dx + ca * dy   # "left" robot

    # current matrix cell size n×n
    cell = MAP_SIZE_M / float(n)

    # pre-coup indices
    gx = int((yr + half) / cell)
    gy = int((xr + half) / cell)

    # matrix of minimap_matrix(n) upside down Y (flipud)
    mx = int(np.clip(gx,        0, n - 1))
    my = int(np.clip((n - 1) - gy, 0, n - 1))
    return mx, my



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
    # soft check of model readiness/data
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
    # which joints does politics read from qpos[7:7+na] (in the right order)
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
    Returns the index→name for 8 corner format:
      0..3: left shoulder (pitch, roll, yaw), left elbow (one axis - preferably pitch)
      4..7: right shoulder (pitch, roll, yaw), right elbow (one axis - preferably pitch)

    For each slot we try:
      (1) exact match,
      (2) unique suffix,
      (3) substring heuristic.
    Works with name prefixes (r1_, r2_, ...).
    """
    # Slot descriptions: (human readable_name, list of alternatives, heuristics)
    SLOTS = [
        ("left_shoulder_pitch_joint",
         ["left_shoulder_pitch_joint"],                            {"must": ["left", "shoulder", "pitch"]}),
        ("left_shoulder_roll_joint",
         ["left_shoulder_roll_joint"],                             {"must": ["left", "shoulder", "roll"]}),
        ("left_shoulder_yaw_joint",
         ["left_shoulder_yaw_joint"],                              {"must": ["left", "shoulder", "yaw"]}),
        ("left_elbow_joint",      # preferably pitch-option
         ["left_elbow_joint", "left_elbow_pitch_joint"],           {"must": ["left", "elbow"], "prefer": ["pitch"]}),

        ("right_shoulder_pitch_joint",
         ["right_shoulder_pitch_joint"],                           {"must": ["right", "shoulder", "pitch"]}),
        ("right_shoulder_roll_joint",
         ["right_shoulder_roll_joint"],                            {"must": ["right", "shoulder", "roll"]}),
        ("right_shoulder_yaw_joint",
         ["right_shoulder_yaw_joint"],                             {"must": ["right", "shoulder", "yaw"]}),
        ("right_elbow_joint",     # preferably pitch-option
         ["right_elbow_joint", "right_elbow_pitch_joint"],         {"must": ["right", "elbow"], "prefer": ["pitch"]}),
    ]

    names: list[str] = []
    missing: list[str] = []
    all_names = list(joint_map.keys())

    def _resolve_by_alts(alts: list[str]) -> str | None:
        # exact name
        for a in alts:
            if a in joint_map:
                return a
        # unique suffix
        for a in alts:
            hits = [n for n in all_names if n.endswith(a)]
            if len(hits) == 1:
                return hits[0]
        return None

    def _resolve_by_heuristic(hints: dict) -> str | None:
        must = [s.lower() for s in hints.get("must", [])]
        prefer = [s.lower() for s in hints.get("prefer", [])]
        # candidates by required substrings
        cands = []
        for n in all_names:
            low = n.lower()
            if all(tok in low for tok in must):
                cands.append(n)
        if not cands:
            return None
        # if there are “preferred” tokens, select them
        pref = [n for n in cands if any(t in n.lower() for t in prefer)] if prefer else []
        if len(pref) == 1:
            return pref[0]
        if pref:
            # if there are several, we take the “shortest” one (usually this is the required pitch)
            return sorted(pref, key=len)[0]
        # otherwise we take the shortest one must-candidates
        return sorted(cands, key=len)[0]

    for human_name, alts, hints in SLOTS:
        nm = _resolve_by_alts(alts)
        if nm is None:
            nm = _resolve_by_heuristic(hints)
        if nm is None:
            missing.append(human_name)
        else:
            names.append(nm)

    if missing:
        print("Warning: missing joints for frame control:", ", ".join(missing))
    else:
        print("[FRAME MAP] 8-joint order:", names)

    return {i: n for i, n in enumerate(names)}


# Simple aliases the LLM might produce
JOINT_ALIASES = {
    "right_arm": "right_shoulder_pitch_joint",
    "left_arm":  "left_shoulder_pitch_joint",
    "right_hand":"right_elbow_joint",
    "left_hand": "left_elbow_joint",
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
    return LLM.chat(messages)

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
                # 6) canonicalize action name for navigation
        if "name" in norm and isinstance(norm["name"], str):
            nm = norm["name"].lower().strip()
            if nm in ("go_to", "move_to", "navigate", "goto_xy", "go", "nav"):
                nm = "goto"
            norm["name"] = nm

        # 7) allow alt field names for x,y,speed
        #    (in case he comes tx/ty or vx/vy — gently ignore,
        #     But x/y/speed leave it as the main interface)
        if "tx" in norm and "x" not in norm:
            norm["x"] = norm.pop("tx")
        if "ty" in norm and "y" not in norm:
            norm["y"] = norm.pop("ty")
        if "v" in norm and "speed" not in norm:
            norm["speed"] = norm.pop("v")

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
        
        msearch = re.search(r"(\[.*\]|\{.*\})", raw, flags=re.S)
        json_text = msearch.group(1) if msearch else raw.strip()

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

# === STANCE: steps in place (RL included) ===
STANCE_HOLD = False
STANCE_V = 0.12                  # target principal axis amplitude
STANCE_AXIS = "x"                # "x" or "y"
STANCE_STEPS_EACH = 1
_stance_dir = +1
_stance_local = 0

# Main axis (rhythm ±V + PI for position error)
STANCE_KP = 1.0                  # m/(With·m)
STANCE_KD = 0.3                  # m/from to m/With
STANCE_KI = 0.2                  # m/(With·m·c) — integral removes residual drift
STANCE_I_CLAMP = 0.3             # integrator limitation (in "equivalent" m)

# Cross axis (no rhythm, drift suppression only)
STANCE_ORTH_KP = 1.2
STANCE_ORTH_KD = 0.25

# Staying on course (yaw → 0 relative to the start of the stance)
STANCE_YAW_KP = 2.0              # glad/from to rad
STANCE_YAW_KD = 0.2              # glad/from to rad/With

STANCE_HOME = None               # np.array([x0, y0])
STANCE_YAW_HOME = None           # float (glad)

# controller internal states
STANCE_I_MAIN = 0.0              # integral sum along the principal axis

def start_stance_hold(v: float = 0.12, axis: str = "x", steps_each: int = 1):
    """Enable Rack: Rhythm ±V + PI main axis, lateral axis drift suppression, hold yaw."""
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

    # Remember "home"»
    STANCE_HOME = np.array([float(d.qpos[0]), float(d.qpos[1])], dtype=np.float32)

    # Remembering the course at the start of the stance
    qw, qx, qy, qz = d.qpos[3:7]
    STANCE_YAW_HOME = _quat_yaw(qw, qx, qy, qz)

    # Resetting the integrator
    STANCE_I_MAIN = 0.0

    # The starting command is zero: the controller will set it in a cycle
    cmd[:] = 0.0

def stop_stance_hold():
    """Turn off the rack in place."""
    global STANCE_HOLD
    STANCE_HOLD = False
    cmd[:] = 0.0

# for step detection sin(phase)
PHASE_SIN = 0.0
PHASE_PREV_SIN = 0.0
def render_once(minimap: bool = True):
    # 1) process events and apply WASD to the camera
    glfw.poll_events()
    update_camera_from_input()

    # 2) render the scene
    w, h = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, w, h)
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)
    if minimap:
        draw_minimap_overlay(w, h)
    glfw.swap_buffers(window)


def apply_gait_pd():
    na = num_actions
    tau = pd_control(
        target_dof_pos,
        d.qpos[7:7+na],
        kps,
        np.zeros_like(kds),
        d.qvel[6:6+na],
        kds
    )
    for k, aid in enumerate(GAIT_ACT_IDS):
        d.ctrl[aid] = float(tau[k])

def apply_upper_pd():
    if not ARM_ACT_IDS:
        return
    q  = d.qpos[ARM_QPOS_ADDRS]
    dq = d.qvel[ARM_QVEL_ADDRS]
    err = ARM_TARGETS - q
    global ARM_IERR
    ARM_IERR += err * m.opt.timestep
    ARM_IERR = np.clip(ARM_IERR, -ARM_I_CLAMP, ARM_I_CLAMP)

    bias = d.qfrc_bias[ARM_QVEL_ADDRS]
    tau_arm = bias + ARM_KPS * err - ARM_KDS * dq + ARM_KIS * ARM_IERR

    gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else None
    for i, aid in enumerate(ARM_ACT_IDS):
        g = float(gears[aid, 0]) if gears is not None else 1.0
        lo, hi = m.actuator_ctrlrange[aid]
        u = tau_arm[i] / max(g, 1e-6)
        d.ctrl[aid] = float(np.clip(u, lo, hi))

def policy_tick():
    """Updates obs → action → target_dof_pos. Call sparsely by decimation."""
    global action, target_dof_pos, PHASE_SIN, PHASE_PREV_SIN
    if counter % control_decimation != 0:
        return
    na = num_actions
    qj = d.qpos[7:7+na]; dqj = d.qvel[6:6+na]
    quat = d.qpos[3:7];  omega = d.qvel[3:6]

    qj_scaled  = (qj - default_angles) * dof_pos_scale
    dqj_scaled = dqj * dof_vel_scale
    grav = get_gravity_orientation(quat)
    omega_scaled = omega * ang_vel_scale

    period = 0.8
    t_sim  = counter * simulation_dt
    phase  = (t_sim % period) / period
    sinp   = math.sin(2 * math.pi * phase)
    cosp   = math.cos(2 * math.pi * phase)

    obs[:3] = omega_scaled
    obs[3:6] = grav
    obs[6:9] = cmd * cmd_scale
    obs[9:9+na] = qj_scaled
    obs[9+na:9+2*na] = dqj_scaled
    obs[9+2*na:9+3*na] = action
    obs[9+3*na:9+3*na+2] = np.array([sinp, cosp])

    action[:] = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
    target_dof_pos[:] = action * action_scale + default_angles

    PHASE_PREV_SIN, PHASE_SIN = PHASE_SIN, sinp

def step_once(render: bool = True) -> bool:
    """One full tick: PD legs, PD hands, mj_step, policy_tick, (opt.) render."""
    global counter
    if _should_abort():
        d.ctrl[:] = 0.0
        return False
    apply_gait_pd()
    apply_upper_pd()
    mj.mj_step(m, d)
    counter += 1
    policy_tick()
    if render:
        render_once(minimap=True)
    return True


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

# === GLOBAL ABORT ===
RUN_ABORT = False
def clear_abort():
    """Reset the emergency flag before STARTING new movements."""
    global RUN_ABORT
    RUN_ABORT = False

def _should_abort() -> bool:
    # general quick check in all cycles
    return RUN_ABORT or (window is not None and glfw.window_should_close(window))

def abort_all_motion():
    """Request an emergency stop of everything that is currently spinning."""
    global RUN_ABORT, steps_needed, steps_done, cmd, target_dof_pos, action, short_extra_done, last_activity_time
    RUN_ABORT = True
    # reset all gait controls/hands
    steps_needed = 0
    steps_done = 0
    cmd[:] = 0.0
    action[:] = 0.0
    if 'default_angles' in globals() and default_angles is not None:
        target_dof_pos = default_angles.copy()
    if ARM_TARGETS is not None and len(ARM_TARGETS) > 0:
        ARM_TARGETS[:] = 0.0
    # immediately turn off the actuators
    d.ctrl[:] = 0.0
    d.qacc[:] = 0.0
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    # so that auto balance does not start
    short_extra_done = True
    last_activity_time = time.time()


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control law."""
    return (target_q - q) * kp + (target_dq - dq) * kd
def hard_reset(reload_model: bool = False, reload_policy: bool = True):
    """Full reset. Reboot optional XML And/or politics."""
    abort_all_motion()  # single emergency stop + resetting all controls

    global m, d, scene, context, opt
    global policy, INIT_QPOS
    global joint_map, joint_index_map, ALLOWED_JOINTS, GAIT_ACT_IDS
    global counter, old_sin_phase, short_extra_done, last_activity_time
    global steps_needed, steps_done, cmd, action, target_dof_pos

    # 1) reload the model if necessary/context
    if reload_model:
        m = mj.MjModel.from_xml_path(xml_path)
        d = mj.MjData(m)
        m.opt.timestep = simulation_dt
        scene = mj.MjvScene(m, maxgeom=10000)
        context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)
        opt = mj.MjvOption()

    # 2) reload the policy if necessary
    if reload_policy:
        policy = torch.jit.load(policy_path)

    # 3) rebuild model-dependent structures
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    joint_map = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}": m.jnt_qposadr[j]
                 for j in range(m.njnt)}
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    ]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)
    auto_tune_upper_kp()

    GAIT_ACT_IDS = build_gait_act_ids()
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS), "intersection of arm and gait actuators!"

    # 4) hard physical reset and buffer initialization
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    steps_needed = steps_done = 0
    cmd[:] = 0.0
    action[:] = 0.0
    target_dof_pos = default_angles.copy()

    counter = 0
    old_sin_phase = 0.0
    short_extra_done = True          # how was it with you hard_reset
    last_activity_time = time.time()

    print("[HARD RESET] model:", "reloaded" if reload_model else "kept",
          "| policy:", "reloaded" if reload_policy else "kept")

    clear_abort()  # remove the emergency flag, you can move again
    return


def do_reset():
    """Soft reset: base -> INIT_QPOS, all joints to zero, without rebooting the model/politicians."""
    abort_all_motion()  # stop everything, reset control signals

    global counter, old_sin_phase, short_extra_done, last_activity_time
    global target_dof_pos

    # Complete reset of the state and return of the base to its original position
    mj.mj_resetData(m, d)
    d.qpos[:] = INIT_QPOS

    # Let's not lose everything-free joints; BALL -> unit quaternion
    for j in range(m.njnt):
        jtype = m.jnt_type[j]; adr = m.jnt_qposadr[j]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            d.qpos[adr] = 0.0
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            d.qpos[adr:adr+4] = np.array([1.0, 0.0, 0.0, 0.0])

    mj.mj_forward(m, d)

    # Counters/flags
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False         # how was it with you do_reset
    target_dof_pos = default_angles.copy()
    last_activity_time = time.time()

    print("Reset done: base -> initial pose, joints -> zero.")
    clear_abort()
    return


def _control_step_with_arms():
    return step_once(render=False)


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
         - {"name": "goto", "x": <m>, "y": <m>, "speed": <m/s>, ["stop": <m>, "slow_r": <m>, "yaw_kp": <>, "yaw_max_deg": <>]}
        — go to the world point (X,Y); speed and additional fields are optional.

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

        

        def smoothstep(t: float) -> float:
            # t in [0,1] -> smooth S-curve
            return t * t * (3 - 2 * t)

        fps_dt = 1.0 / fps


        # ===== Variant B (A): list of objects (incl. simultaneous frames via "frame") =====
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
                        # --- NEW: goto (world XY) ---
            if isinstance(item, dict) and item.get("name") in ("goto", "go_to", "move_to", "navigate", "goto_xy"):
                try:
                    tx = float(item.get("x"))
                    ty = float(item.get("y"))
                except Exception:
                    print(f"[goto] need numeric x,y; got: {item}")
                    continue
                v  = float(item.get("speed", 0.20))
                stop   = float(item.get("stop", 0.10))
                slow_r = float(item.get("slow_r", 0.60))
                yaw_kp = float(item.get("yaw_kp", 2.0))
                yaw_max_deg = float(item.get("yaw_max_deg", 120.0))
                print(f"[goto] ({tx:.2f}, {ty:.2f}) speed={v:.2f} stop={stop:.2f} slow_r={slow_r:.2f}")
                go_to_xy_blocking(
                    tx, ty, speed=v,
                    stop=stop, slow_r=slow_r,
                    yaw_kp=yaw_kp, yaw_max_deg=yaw_max_deg
                )
                continue

            # Frame with several joints — SIMULTANEOUS
            if isinstance(item, dict) and "frame" in item:
                joints = item["frame"]
                if not isinstance(joints, list) or not joints:
                    print(f"Empty or invalid 'frame' in item: {item}")
                    continue

                # Dividing: Hand Targets (via PD) and for the rest (directly in qpos)
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

                # Starting values
                start_arm = {jn: float(ARM_TARGETS[ARM_NAME_TO_IDX[jn]]) for jn in arm_goals.keys()}
                start_angles = {jn: d.qpos[joint_map[jn]] for jn in direct_targets.keys()}

                for step in range(n_steps):
                    if _should_abort():
                        print("[ABORT] single-joint")
                        return
                    t = smoothstep((step + 1) / n_steps)

                    # HANDS: smoothly lead PD-goals
                    for jname, goal in arm_goals.items():
                        idx = ARM_NAME_TO_IDX[jname]
                        ARM_TARGETS[idx] = (1 - t) * start_arm[jname] + t * goal

                    # OTHER: as before - directly to qpos
                    for jname, target_rad in direct_targets.items():
                        start_rad = start_angles[jname]
                        d.qpos[joint_map[jname]] = (1 - t) * start_rad + t * target_rad

                    # --- let it work PD hands-on (mini-steps of physics) ---
                    for _ in range(substeps_per_render):
                        if _should_abort():
                            print("[ABORT] single-joint/substep")
                            return
                        _control_step_with_arms()
                    render_once(minimap=True)
                    time.sleep(fps_dt)
                continue


            # Regular single joint — backward compatible with previous format
            if isinstance(item, dict) and "name" in item and "angle" in item:
                
                jname = item["name"]
                if jname not in joint_map:
                    print(f"Joint '{jname}' not found.")
                    continue
                
                target_angle_rad = math.radians(float(item["angle"]))
                idx = ARM_NAME_TO_IDX[jname]
                duration = float(item.get("duration", duration_per_frame))
                n_steps = max(1, int(duration * fps))
                start = d.qpos[idx]
                
                print("# Regular single joint — backward compatible with previous format")
                print(jname, " ", target_angle_rad)
                
                print("IDX : ", idx)
                
                qpos_idx = joint_map[jname]
                
                print("WHY QPOS_IDX != IDX ", qpos_idx, " != ", idx)
                
                
                for step in range(n_steps):
                    t = smoothstep((step + 1) / n_steps)
                    ARM_TARGETS[idx] = (1 - t) * start + t * target_angle_rad

                    # we give PD-hand control work inside each interpolation step
                    for _ in range(substeps_per_render):
                        _control_step_with_arms()
                    render_once(minimap=True)
                    time.sleep(fps_dt)

                qpos_idx = joint_map[jname]
                start_rad = d.qpos[qpos_idx]
                for step in range(n_steps):
                    t = smoothstep((step + 1) / n_steps)
                    d.qpos[qpos_idx] = (1 - t) * start_rad + t * target_angle_rad
                    mj.mj_forward(m, d)
                    render_once(minimap=True)
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
                   # <- IMPORTANT: please stop first
        do_reset()
                   


    elif key == glfw.KEY_H and pressed:
                # <- first stop everything
        hard_reset(reload_model=True, reload_policy=True)
        auto_tune_upper_kp()
                


    # 'P' — paste motions via modal window (frames or action objects)
    elif key == glfw.KEY_P and pressed:
        cmd_list = open_paste_window()
        if cmd_list:
            clear_abort()               # ← add
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
      
    elif key == glfw.KEY_M and pressed:
        mat = minimap_matrix()
        print("\n".join(" ".join(map(str, r)) for r in mat))
        
    elif key == glfw.KEY_B and pressed:
        try:
            tx = float(input("Target X (m): "))
            ty = float(input("Target Y (m): "))
            v  = input("Speed (m/s) [0.2]: ").strip()
            v  = float(v) if v else 0.2
        except Exception:
            print("Invalid input.")
            return
        print(f"Go-to ({tx:.2f}, {ty:.2f}) at {v:.2f} m/s…")
        go_to_xy_blocking(tx, ty, speed=v)

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

# somewhere near the globals
CAMERA_SPEED = 0.2

def update_camera_from_input():
    """Apply WASD + arrows to cam.lookat."""
    global cam, movement, CAMERA_SPEED
    if cam is None or not movement:
        return
    azimuth_rad = math.radians(cam.azimuth)
    right_vec   = np.array([math.sin(azimuth_rad), -math.cos(azimuth_rad), 0.0], dtype=np.float32)
    forward_vec = np.array([ right_vec[1],         -right_vec[0],          0.0], dtype=np.float32)

    if movement.get("forward"):   cam.lookat -= CAMERA_SPEED * forward_vec
    if movement.get("backward"):  cam.lookat += CAMERA_SPEED * forward_vec
    if movement.get("left"):      cam.lookat -= CAMERA_SPEED * right_vec
    if movement.get("right"):     cam.lookat += CAMERA_SPEED * right_vec
    if movement.get("rise"):      cam.lookat[2] += CAMERA_SPEED
    if movement.get("fall"):      cam.lookat[2] -= CAMERA_SPEED

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

if __name__ == "__main__":
    # ---- 0) arguments ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--rag-csv", default="1000.csv")
    parser.add_argument("--rag-json", default="data_action")
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default="g1")
    parser.add_argument("--xml", help="xml scene", default="scene.xml")
    args = parser.parse_args()

    # RAG-settings for L/IN/G hot keys
    settings.RAG_K = args.rag_k
    settings.RAG_CSV = args.rag_csv
    settings.RAG_JSON = args.rag_json

    # ---- 1) files for the selected robot ----
    name_ = args.robot
    scene_file = args.xml
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path    = f"unitree_rl_gym/resources/robots/{name_}/{scene_file}"

    # ---- 2) config → globals ----
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    simulation_dt       = cfg["simulation_dt"]
    control_decimation  = cfg["control_decimation"]
    kps                 = np.array(cfg["kps"], dtype=np.float32)
    kds                 = np.array(cfg["kds"], dtype=np.float32)
    default_angles      = np.array(cfg["default_angles"], dtype=np.float32)
    ang_vel_scale       = cfg["ang_vel_scale"]
    dof_pos_scale       = cfg["dof_pos_scale"]
    dof_vel_scale       = cfg["dof_vel_scale"]
    action_scale        = cfg["action_scale"]
    cmd_scale           = np.array(cfg["cmd_scale"], dtype=np.float32)
    num_actions         = cfg["num_actions"]
    num_obs             = cfg["num_obs"]

    # ---- 3) MuJoCo model/data ----
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt
    mj.mj_forward(m, d)

    # maps of joint names and allowed joints
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    joint_map = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}": m.jnt_qposadr[j]
                 for j in range(m.njnt)}
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    ]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    # ---- 4) PD upper part (binding of actuators) ----
    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)   # creates ARM_* arrays
    auto_tune_upper_kp()                      # adjusts Kp/Kd By bias’at

    # ---- 5) what actuators are touched by gait? ----
    GAIT_ACT_IDS = build_gait_act_ids()       # requires m, num_actions
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS), "intersection of arm and gait actuators!"

    # ---- 6) policy ----
    policy = torch.jit.load(policy_path)

    # ---- 7) window and visualization ----
    glfw.init()
    window = glfw.create_window(1200, 900, "MuJoCo Manual Viewer", None, None)
    glfw.make_context_current(window)

    cam = mj.MjvCamera();  cam.azimuth = 180; cam.elevation = -15; cam.distance = 4.0
    cam.lookat = np.array([0.0, 0.0, 0.8])
    opt = mj.MjvOption()
    scene = mj.MjvScene(m, maxgeom=10000)
    context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    movement = {"forward": False, "backward": False, "left": False, "right": False, "rise": False, "fall": False}
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # ---- 8) working buffers/counters ----
    INIT_QPOS = d.qpos.copy()
    steps_needed = 0
    steps_done   = 0
    cmd          = np.zeros(3, dtype=np.float32)
    action       = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs          = np.zeros(num_obs, dtype=np.float32)
    counter      = 0
    old_sin_phase = 0.0
    last_activity_time = time.time()
    short_extra_done = False

    # ---- main loop ----
    while not glfw.window_should_close(window):
        step_once(render=True)
    glfw.terminate()


