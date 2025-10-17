
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


_LEG_TOKENS   = ("hip", "knee", "ankle", "toe", "foot", "leg")
_FINGER_TOKENS= ("thumb", "index", "middle", "ring", "pinky", "finger")
_UPPER_TOKENS = ("shoulder","elbow","wrist","hand","torso","waist","spine","chest","neck","head") + _FINGER_TOKENS

def _is_upper_joint(name: str) -> bool:
    """True для суставов верха, включая пальцы h1/h1_2 (L_/R_*)."""
    nm = name.lower()

    # убрать префикс робота r1_/r2_/r3_ и суффикс _joint
    nm = re.sub(r"^r\d+_", "", nm)
    if nm.endswith("_joint"):
        nm = nm[:-6]

    # быстрый фильтр ног
    if any(tok in nm for tok in _LEG_TOKENS):
        return False

    # явные признаки верха (включая thumb/index/…)
    if any(tok in nm for tok in _UPPER_TOKENS):
        return True

    # формат h1/h1_2: "l_*" / "r_*" + признаки кисти/пальцев/руки/плеча
    if nm.startswith(("l_", "r_")):
        rest = nm[2:]
        if any(tok in rest for tok in _FINGER_TOKENS + ("wrist","hand","elbow","shoulder")):
            return True

    return False


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
        # детекция шага по фронту sin(phase)
        if PHASE_SIN >= 0 and PHASE_PREV_SIN < 0:
            if STANCE_HOLD:
                # (как у вас было; можно оставить логику stance)
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
# ==== MINIMAP (реальное время, вид сверху) ====
MAP_CELL_M    = 0.05          # 5 см на пиксель (оставьте как есть)
MAP_RADIUS_M  = 3.0           # радиус видимой зоны вокруг робота, м
MAP_SIZE_M    = MAP_RADIUS_M * 2.0   # окно карты — диаметр 6 м
MINIMAP_PX    = 220
MAP_BG        = 230
MAP_OBJ_COL   = 80
MAP_ROBOT_COL = (220, 40, 40)
MAP_ZONE_COL  = (120, 180, 240)  # цвет кольца зоны (3 м)
MAP_BORDER    = 40

MAP_FOLLOW_YAW = True  # карта поворачивается с роботом (False — «север вверх»)
# === Reactive obstacle avoidance (applies to ANY walking) ===
NAV_AVOID_ENABLE    = True
NAV_LOOKAHEAD       = 0.90   # на сколько метров «заглядываем» вперёд
NAV_STOP_DIST       = 0.30   # буфер до препятствия, при котором начинаем тормозить/обходить
NAV_INFLATE_CELLS   = 2      # «толщина робота» в клетках миникарты
NAV_SEARCH_DEG      = 120    # веер поиска альтернативного курса (±)
NAV_RAYS            = 29     # число лучей в веере (нечётное)
NAV_MAX_YAW_RATE    = math.radians(160.0)  # физ. предел скорости вращения

# кэш оккупации на несколько тиков, чтобы не строить карту каждый раз
_NAV_OCC_CACHE = {"n": 64, "occ": None, "tick": -1000}

# исключаемые геомы по имени/типу (пол, бесконечные плоскости и т.п.)
MAP_EXCLUDE_SUBSTR = ("floor", "ground")
def _world_to_grid(x: float, y: float, n: int) -> tuple[int, int]:
    """
    Перевод мировой точки (x,y) в пиксели миникарты:
    - центр карты совпадает с позицией робота;
    - при MAP_FOLLOW_YAW=True карта повёрнута так, что «вперёд робота» = вверх.
    """
    # центр окна (в метрах от робота)
    half = MAP_SIZE_M * 0.5

    # позиция и курс робота
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    qw, qx, qy, qz = d.qpos[3:7]
    yaw = _quat_yaw(qw, qx, qy, qz) if MAP_FOLLOW_YAW else 0.0

    # сдвиг в систему робота
    dx, dy = x - rx, y - ry

    # поворот на -yaw (мир -> «корпус»)
    ca, sa = math.cos(-yaw), math.sin(-yaw)
    xr = ca * dx - sa * dy   # вперёд (ось робота)
    yr = sa * dx + ca * dy   # влево  (ось робота)

    # проекция в пиксели:
    #   gx — горизонталь (вправо положительно) ← берём «влево робота»
    #   gy — вертикаль (вниз положительно)    ← берём «вперёд робота»
    # После flipud() «вперёд робота» станет вверх на экране.
    gx = int((yr + half) / MAP_CELL_M)
    gy = int((xr + half) / MAP_CELL_M)
    return gx, gy

def _draw_ring(img: np.ndarray, cx: int, cy: int, r: int, rgb=(0,0,0), thickness: int = 2):
    # тонкое кольцо радиуса r (толщина в пикселях)
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
    Луч от (x0,y0) под углом ang на max_dist.
    Возвращает расстояние до ПЕРВОГО препятствия (м) или None.
    skip_first — сколько метров от старта игнорируем (чтобы не увидеть себя).
    """
    if mat is None:
        mat = minimap_matrix(n)
    cell_m = MAP_SIZE_M / n
    step = cell_m * 0.5  # полклетки — надёжно
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
    Сколько метров свободно по направлению ang (0..max_dist), по сетке миникарты.
    """
    hit = _ray_first_hit_distance_world(x0, y0, ang, max_dist, n=n, skip_first=0.0, mat=mat)
    return max_dist if hit is None else max(0.0, hit)
# ---------- GRID A* + FOLLOW ----------

def _mat_idx_to_world(mx: int, my: int, n: int) -> tuple[float, float]:
    """Инверсия _world_to_mat_idx: центр клетки (mx,my) -> мировые (x,y)."""
    half = MAP_SIZE_M * 0.5
    cell = MAP_SIZE_M / float(n)

    # матрица minimap_matrix — перевёрнута по Y (flipud), откатываем:
    gx = mx
    gy = (n - 1) - my

    xr = (gy + 0.5) * cell - half   # вперёд робота, м
    yr = (gx + 0.5) * cell - half   # влево робота, м

    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    yaw = _quat_yaw(*d.qpos[3:7]) if MAP_FOLLOW_YAW else 0.0
    ca, sa = math.cos(yaw), math.sin(yaw)
    dx = ca * xr - sa * yr
    dy = sa * xr + ca * yr
    return rx + dx, ry + dy


def _inflate_occupancy(mat: np.ndarray, r: int) -> np.ndarray:
    """Надуть препятствия на r клеток (манхэттен-круг). 1/2 → препятствие, 0 → свободно."""
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
    # робот (2) считаем свободным
    out[(mat == 2)] = 0
    return out


def _neighbors8(x: int, y: int):
    for dx, dy in _NEIGH8:
        yield x + dx, y + dy


def _a_star(start: tuple[int,int], goal: tuple[int,int], occ: np.ndarray,
            avoid_corner_cut: bool = True) -> list[tuple[int,int]] | None:
    """A* на сетке occ (1 — стена). 8-связность, запрещаем «резать углы» возле стен."""
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
            # восстановить путь
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
            # запрет corner-cut: если диагональ, два смежных кардинальных должны быть свободны
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
    """Проверка видимости по миру, опираясь на сетку occ."""
    # используем существующий трейс по миру → сетка
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
            # локальное раздувание на проверке, чтобы не чесать бортом
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
    """Упростить ломаную: выбрасываем лишние точки, если отрезок виден."""
    if len(points_world) <= 2:
        return points_world[:]
    out = [points_world[0]]
    i = 0
    while i < len(points_world) - 1:
        j = len(points_world) - 1
        # от дальнего к ближнему ищем самую дальнюю видимую
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
                      inflate_cells: int = 2,          # «толщина» робота в клетках
                      replan_every: int = 90,          # иногда перепланируем (тика)
                      los_inflate: int = 1):           # сколько расширять при проверке видимости
    """
    Планирование A* по миникарте и следование по упрощённым вейпоинтам.
    Работает в координатах мира, но план строится в сетке minimap_matrix(n_map).
    """
    # 1) снимок карты и «раздутая» оккупация
    mat_raw = minimap_matrix(n_map)
    occ = _inflate_occupancy(mat_raw, inflate_cells)

    # 2) старт/финиш клетки (робота разрешаем как свободную)
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    sx, sy = _world_to_mat_idx(rx, ry, n_map)
    gxw, gyw = snap_goal_to_free(tx, ty, n=n_map, inflate=1)
    gx, gy = _world_to_mat_idx(gxw, gyw, n_map)

    # 3) A*
    path = _a_star((sx, sy), (gx, gy), occ)
    if not path:
        print("[goto/a*] no path")
        return

    # 4) → мир + упрощение прямыми видимостями
    pts_world = [_mat_idx_to_world(x, y, n_map) for (x, y) in path]
    pts_world = _simplify_world_path(pts_world, n_map, occ, inflate_on_los=los_inflate)

    print(f"[goto/a*] grid len={len(path)}  waypoints={len(pts_world)}")
    # 5) трекинг вейпоинтов
    stop_stance_hold(); clear_abort()
    yaw_max = math.radians(yaw_max_deg)

    wp_i = 0
    last_plan_counter = counter
    dist0 = max(1e-6, math.hypot(tx - rx, ty - ry))

    while not glfw.window_should_close(window) and not _should_abort():
        # цель — текущий вейпоинт или финальная точка
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

        # иногда перепланируем на свежей карте (если мир динамический)
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
                wp_i = 0  # начинаем с ближайшего к себе
                # продвинем индекс к ближайшему видимому
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

        # пропустим геомы робота
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
    """4 угла следа mjGEOM_BOX в мировой плоскости XY (упорядочены без самопересечений)."""
    sz = np.array(m.geom_size[gid], dtype=float)  # (sx, sy, sz) — ПОЛУразмеры
    R  = np.array(d.geom_xmat[gid]).reshape(3,3)  # мировая ориентация геома
    p  = np.array(d.geom_xpos[gid], dtype=float)  # центр

    # локальные полуоси X,Y, спроецированные в XY мира
    ux = R[:2, 0] * sz[0]
    vy = R[:2, 1] * sz[1]
    c  = p[:2]

    # 4 вершины по часовой
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
    Если цель попала в препятствие — переносим на ближайшую свободную клетку.
    Важно: minimap_matrix() внутри переворачивает по Y, поэтому тут явно учитываем flip.
    Работает в БАЗОВОМ разрешении сетки (MAP_CELL_M), чтобы избежать рассинхрона масштабов.
    """
    # --- базовое количество клеток, как в _minimap_build_image() ---
    base_n = int(MAP_SIZE_M / MAP_CELL_M)
    base_n = max(32, min(1024, base_n))

    # матрица оккупации в базной сетке (0—своб., 1—препятс., 2—робот), уже flipud
    mat = minimap_matrix(base_n)   # shape=(base_n, base_n), flipped по Y

    # world -> grid (неперевёрнутая сетка "unflipped")
    gx_u, gy_u = _world_to_grid(tx, ty, base_n)
    gx_u = int(np.clip(gx_u, 0, base_n - 1))
    gy_u = int(np.clip(gy_u, 0, base_n - 1))

    # grid(unflipped) -> mat(flippedY)
    mx0, my0 = gx_u, (base_n - 1 - gy_u)

    def _is_free(mx: int, my: int) -> bool:
        """Свободна ли клетка c учётом надувания inflate (0/1/2; 2=робот считаем свободным)."""
        nloc = mat.shape[0]
        for yy in range(max(0, my - inflate), min(nloc - 1, my + inflate) + 1):
            for xx in range(max(0, mx - inflate), min(nloc - 1, mx + inflate) + 1):
                v = int(mat[yy, xx])
                if v == 1:  # препятствие
                    return False
        return True  # 0 или 2 — ок

    occ0 = int(mat[my0, mx0])
   # print(f"[snap] check world=({tx:.3f},{ty:.3f}) -> grid=({gx_u},{gy_u}) -> mat=({mx0},{my0}) occ={occ0}")

    # уже свободно — ничего не делаем
    if _is_free(mx0, my0):
    #    print("[snap] already FREE — keep original target")
        return tx, ty

    # BFS по матрице (в координатах mat, т.е. с переворотом по Y)

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

            # grid(unflipped) -> world (инверсия _world_to_grid с базной ячейкой)
            half = MAP_SIZE_M * 0.5
            cell = MAP_CELL_M
            xr = (gy_f + 0.5) * cell - half   # вперёд (ось робота)
            yr = (gx_f + 0.5) * cell - half   # влево  (ось робота)

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

        # 8-соседей
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q.append((mx + dx, my + dy, rings + 1))

    print("[snap] WARN: no FREE cell found — keep original target")
    return tx, ty

# ---------------------------------------------------------------------------

def _minimap_build_image() -> np.ndarray:
    """Собираем картинку миникарты (RGB uint8), сверху—вниз (потом перевернём)."""
    n = int(MAP_SIZE_M / MAP_CELL_M)
    n = max(32, min(1024, n))  # разумные пределы
    img = np.full((n, n, 3), MAP_BG, dtype=np.uint8)

    # рамка (1 пиксель по периметру)
    img[0, :, :] = 0
    img[-1, :, :] = 0
    img[:, 0, :] = 0
    img[:, -1, :] = 0

    # объекты (по геомам)
    geom_ids = _minimap_collect_geom_ids()
    for gid in geom_ids:
        x, y = d.geom_xpos[gid][0], d.geom_xpos[gid][1]
        gx, gy = _world_to_grid(x, y, n)

        gtype = m.geom_type[gid]
        sz    = np.array(m.geom_size[gid])

        if gtype == mj.mjtGeom.mjGEOM_BOX and sz.size >= 2:
            poly_xy = _box_corners_world_xy(gid)     # 4 угла в мире
            poly_px = _world_poly_to_grid(poly_xy, n)  # в пиксели карты
            _draw_filled_polygon(img, poly_px, (MAP_OBJ_COL, MAP_OBJ_COL, MAP_OBJ_COL))
        else:
            r_m = float(np.max(sz[:2])) if sz.size else 0.05
            r_px = max(1, int(r_m / MAP_CELL_M))
            _draw_disc(img, gx, gy, r_px, (MAP_OBJ_COL, MAP_OBJ_COL, MAP_OBJ_COL))

    # робот (позиция + стрелка курса)
       # робот — всегда в центре миникарты
    cx = n // 2
    cy = n // 2
    _draw_disc(img, cx, cy, 2, MAP_ROBOT_COL)

    # круг «зона 3 м»
    zone_px = max(1, int(MAP_RADIUS_M / MAP_CELL_M))
    _draw_ring(img, cx, cy, zone_px, MAP_ZONE_COL, thickness=2)

    # стрелка курса: при MAP_FOLLOW_YAW карта уже повернута по курсу,
    # поэтому стрелку рисуем «вверх» (стабильно и читабельно).
    L = 8  # длина стрелки в клетках
    tipx = cx
    tipy = cy - L
    steps = max(abs(tipx - cx), abs(tipy - cy), 1)
    for i in range(steps + 1):
        xx = cx + int((tipx - cx) * i / steps)
        yy = cy + int((tipy - cy) * i / steps)
        _safe_set_px(img, xx, yy, MAP_ROBOT_COL)


    return img
# Один раз вычислим множество body_id робота (его поддерево)
_ROBOT_BODY_TREE: set[int] | None = None

def _build_robot_body_tree() -> set[int]:
    # ищем free-joint с qposadr==0 — базу робота
    base_jid = -1
    for j in range(m.njnt):
        if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and m.jnt_qposadr[j] == 0:
            base_jid = j
            break
    if base_jid < 0:
        return set()

    root_bid = int(m.jnt_bodyid[base_jid])

    # таблица детей для обхода
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

    # препятствия — только пиксели цвета объектов (MAP_OBJ_COL)
    obj_mask = np.all(img == MAP_OBJ_COL, axis=2)
    rob_mask = np.all(img == MAP_ROBOT_COL, axis=2)

    mat = obj_mask.astype(np.uint8)
    mat[rob_mask] = 2  # робот помечается отдельным классом

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

    # рисуем в окно
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, context)

    # Фон (без context в этой версии MuJoCo)
    try:
        mj.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.6)
    except TypeError:
        # на всякий случай — другие версии требуют context
        mj.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.6, context)

    # Пиксели миникарты (эта функция контекст принимает всегда)
    mj.mjr_drawPixels(rgb_buf, None, rect, context)

# =========================
# Saving helpers
# =========================
# === Wavefront (Lee) на локальной миникарте ===

_NEIGH8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
_NEIGH4 = [(0,-1),(1,0),(0,1),(-1,0)]

def _world_to_mat_idx(x: float, y: float, n: int) -> tuple[int, int]:
    """
    Мировые (x,y) -> индексы в матрице minimap_matrix(n).
    ВАЖНО: используем размер клетки = MAP_SIZE_M / n, а не фиксированный MAP_CELL_M.
    Учитываем поворот карты (MAP_FOLLOW_YAW) и flipud() в minimap_matrix().
    """
    # центр окна в метрах от робота
    half = MAP_SIZE_M * 0.5

    # позиция и курс робота
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    qw, qx, qy, qz = d.qpos[3:7]
    yaw = _quat_yaw(qw, qx, qy, qz) if MAP_FOLLOW_YAW else 0.0

    # в систему робота
    dx, dy = x - rx, y - ry
    ca, sa = math.cos(-yaw), math.sin(-yaw)
    xr = ca * dx - sa * dy   # "вперёд" робота
    yr = sa * dx + ca * dy   # "влево"  робота

    # размер клетки текущей матрицы n×n
    cell = MAP_SIZE_M / float(n)

    # индексы до переворота
    gx = int((yr + half) / cell)
    gy = int((xr + half) / cell)

    # матрица из minimap_matrix(n) перевёрнута по Y (flipud)
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
    Возвращает индекс→имя для формата из 8 углов:
      0..3: левое плечо (pitch, roll, yaw), левый локоть (одна ось — предпочтительно pitch)
      4..7: правое плечо (pitch, roll, yaw), правый локоть (одна ось — предпочтительно pitch)

    Для каждого слота пробуем:
      (1) точное совпадение,
      (2) уникальный суффикс,
      (3) эвристику по подстрокам.
    Работает с префиксами имён (r1_, r2_, ...).
    """
    # Описания слотов: (человекочитаемое_имя, список альтернатив, эвристика)
    SLOTS = [
        ("left_shoulder_pitch_joint",
         ["left_shoulder_pitch_joint"],                            {"must": ["left", "shoulder", "pitch"]}),
        ("left_shoulder_roll_joint",
         ["left_shoulder_roll_joint"],                             {"must": ["left", "shoulder", "roll"]}),
        ("left_shoulder_yaw_joint",
         ["left_shoulder_yaw_joint"],                              {"must": ["left", "shoulder", "yaw"]}),
        ("left_elbow_joint",      # предпочтительно pitch-вариант
         ["left_elbow_joint", "left_elbow_pitch_joint"],           {"must": ["left", "elbow"], "prefer": ["pitch"]}),

        ("right_shoulder_pitch_joint",
         ["right_shoulder_pitch_joint"],                           {"must": ["right", "shoulder", "pitch"]}),
        ("right_shoulder_roll_joint",
         ["right_shoulder_roll_joint"],                            {"must": ["right", "shoulder", "roll"]}),
        ("right_shoulder_yaw_joint",
         ["right_shoulder_yaw_joint"],                             {"must": ["right", "shoulder", "yaw"]}),
        ("right_elbow_joint",     # предпочтительно pitch-вариант
         ["right_elbow_joint", "right_elbow_pitch_joint"],         {"must": ["right", "elbow"], "prefer": ["pitch"]}),
    ]

    names: list[str] = []
    missing: list[str] = []
    all_names = list(joint_map.keys())

    def _resolve_by_alts(alts: list[str]) -> str | None:
        # точное имя
        for a in alts:
            if a in joint_map:
                return a
        # уникальный суффикс
        for a in alts:
            hits = [n for n in all_names if n.endswith(a)]
            if len(hits) == 1:
                return hits[0]
        return None

    def _resolve_by_heuristic(hints: dict) -> str | None:
        must = [s.lower() for s in hints.get("must", [])]
        prefer = [s.lower() for s in hints.get("prefer", [])]
        # кандидаты по обязательным подстрокам
        cands = []
        for n in all_names:
            low = n.lower()
            if all(tok in low for tok in must):
                cands.append(n)
        if not cands:
            return None
        # если есть «предпочтительные» токены — выбираем по ним
        pref = [n for n in cands if any(t in n.lower() for t in prefer)] if prefer else []
        if len(pref) == 1:
            return pref[0]
        if pref:
            # если несколько — берём самый «короткий» (обычно это нужный pitch)
            return sorted(pref, key=len)[0]
        # иначе берём самый короткий из must-кандидатов
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
        #    (на случай если придёт tx/ty или vx/vy — мягко игнорируем,
        #     но x/y/speed оставляем как основной интерфейс)
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

# для детекции шага по sin(phase)
PHASE_SIN = 0.0
PHASE_PREV_SIN = 0.0
def render_once(minimap: bool = True):
    # 1) обработать события и применить WASD к камере
    glfw.poll_events()
    update_camera_from_input()

    # 2) отрисовать сцену
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
    """Обновляет obs → action → target_dof_pos. Вызывайте разреженно по decimation."""
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
    """Один полный тик: PD ног, PD рук, mj_step, policy_tick, (опц.) render."""
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
    """Полный ресет. По желанию — перезагрузка XML и/или политики."""
    abort_all_motion()  # единый экстренный стоп + обнуление всего управления

    global m, d, scene, context, opt
    global policy, INIT_QPOS
    global joint_map, joint_index_map, ALLOWED_JOINTS, GAIT_ACT_IDS
    global counter, old_sin_phase, short_extra_done, last_activity_time
    global steps_needed, steps_done, cmd, action, target_dof_pos

    # 1) при необходимости перезагрузить модель/контекст
    if reload_model:
        m = mj.MjModel.from_xml_path(xml_path)
        d = mj.MjData(m)
        m.opt.timestep = simulation_dt
        scene = mj.MjvScene(m, maxgeom=10000)
        context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)
        opt = mj.MjvOption()

    # 2) при необходимости перезагрузить политику
    if reload_policy:
        policy = torch.jit.load(policy_path)

    # 3) пересобрать зависящие от модели структуры
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
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS), "пересечение актуаторов рук и походки!"

    # 4) жёсткий физический ресет и инициализация буферов
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    steps_needed = steps_done = 0
    cmd[:] = 0.0
    action[:] = 0.0
    target_dof_pos = default_angles.copy()

    counter = 0
    old_sin_phase = 0.0
    short_extra_done = True          # как у тебя было в hard_reset
    last_activity_time = time.time()

    print("[HARD RESET] model:", "reloaded" if reload_model else "kept",
          "| policy:", "reloaded" if reload_policy else "kept")

    clear_abort()  # снимаем аварийный флаг, можно снова двигаться
    return


def do_reset():
    """Мягкий ресет: база -> INIT_QPOS, все суставы в ноль, без перезагрузки модели/политики."""
    abort_all_motion()  # стоп всего, обнуление управляющих сигналов

    global counter, old_sin_phase, short_extra_done, last_activity_time
    global target_dof_pos

    # Полный сброс состояния и возврат базы в исходную позу
    mj.mj_resetData(m, d)
    d.qpos[:] = INIT_QPOS

    # Нулим все не-free суставы; BALL -> единичный кватернион
    for j in range(m.njnt):
        jtype = m.jnt_type[j]; adr = m.jnt_qposadr[j]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            d.qpos[adr] = 0.0
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            d.qpos[adr:adr+4] = np.array([1.0, 0.0, 0.0, 0.0])

    mj.mj_forward(m, d)

    # Счётчики/флаги
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False         # как у тебя было в do_reset
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
        — идти к мировой точке (X,Y); speed и доп.поля опциональны.

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

                    # даём PD-контролю рук поработать внутри каждого шага интерполяции
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
                   # <- ВАЖНО: сначала просим остановку
        do_reset()
                   


    elif key == glfw.KEY_H and pressed:
                # <- сначала стоп всего
        hard_reset(reload_model=True, reload_policy=True)
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

# где-то рядом с глобалами
CAMERA_SPEED = 0.2

def update_camera_from_input():
    """Применить WASD + стрелки к cam.lookat."""
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
    # ---- 0) аргументы ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--rag-csv", default="1000.csv")
    parser.add_argument("--rag-json", default="data_action")
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default="g1")
    parser.add_argument("--xml", help="xml scene")
    args = parser.parse_args()

    # RAG-настройки для L/В/Г горячих клавиш
    settings.RAG_K = args.rag_k
    settings.RAG_CSV = args.rag_csv
    settings.RAG_JSON = args.rag_json

    # ---- 1) файлы под выбранного робота ----
    name_ = args.robot
    scene_file = args.xml
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path    = f"unitree_rl_gym/resources/robots/{name_}/{scene_file}"

    # ---- 2) конфиг → глобали ----
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

    # ---- 3) MuJoCo модель/данные ----
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt
    mj.mj_forward(m, d)

    # карты имён суставов и разрешённых суставов
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    joint_map = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}": m.jnt_qposadr[j]
                 for j in range(m.njnt)}
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    ]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    # ---- 4) PD верхней части (привязка актуаторов) ----
    setup_arm_pd(zero_pose=UPPER_HOLD_ZERO)   # создаёт ARM_* массивы
    auto_tune_upper_kp()                      # подстраивает Kp/Kd по bias’у

    # ---- 5) какие актуаторы трогает походка ----
    GAIT_ACT_IDS = build_gait_act_ids()       # требует m, num_actions
    assert set(GAIT_ACT_IDS).isdisjoint(ARM_ACT_IDS), "пересечение актуаторов рук и походки!"

    # ---- 6) политика ----
    policy = torch.jit.load(policy_path)

    # ---- 7) окно и визуализация ----
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

    # ---- 8) рабочие буферы/счётчики ----
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

    # ---- главный цикл ----
    while not glfw.window_should_close(window):
        step_once(render=True)
    glfw.terminate()


