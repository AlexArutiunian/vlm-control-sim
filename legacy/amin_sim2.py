# minimal_sim_multi.py
# Многороботная версия: любое число копий робота в сцене.
# Переключение активного робота по клавишам 1..9.
# Команды (J, C, T, R) применяются только к активному роботу.
# Невидимые роботы продолжают жить по политике/PD и не падают.

import time, math, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import yaml
import torch

import mujoco as mj
from mujoco.glfw import glfw

# =========================
# Утилиты
# =========================
def rad(deg): return math.radians(deg)
def deg(rad): return math.degrees(rad)

def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def _is_upper_joint(core_name: str) -> bool:
    return (
        core_name == "torso_joint" or
        core_name.startswith((
            "waist_",
            "left_shoulder_", "right_shoulder_",
            "left_elbow", "right_elbow",
            "left_wrist_", "right_wrist_",
            "left_hand_", "right_hand_",
            "L_", "R_"
        ))
    )

def _suggest_kp(name: str) -> float:
    if name == "torso_joint": return 40.0
    if "shoulder" in name:   return 40.0
    if "elbow" in name:      return 30.0
    if "wrist" in name:      return 12.0
    return 4.0  # пальцы

# =========================
# Глобальные из конфига (общие для всех роботов одного типа)
# =========================
simulation_dt = None
control_decimation = None
kps = kds = default_angles = None
ang_vel_scale = dof_pos_scale = dof_vel_scale = action_scale = None
cmd_scale = None
num_actions = num_obs = None

# =========================
# Окно/рендер
# =========================
m = d = None
window = cam = opt = scene = context = None
policy = None

# =========================
# Поиск всех роботов в сцене (по префиксам)
# =========================
def detect_robot_prefixes() -> List[str]:
    """Ищем все префиксы по site 'imu' или body 'torso_link'. Пустой префикс '' допустим."""
    prefixes = set()
    # по сайтам
    for i in range(m.nsite):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) or ""
        if nm.endswith("imu"):
            prefixes.add(nm[:-3])
    # fallback — по телам
    if not prefixes:
        for i in range(m.nbody):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or ""
            if nm.endswith("torso_link"):
                prefixes.add(nm[:-10])
    # фильтруем на наличие free-joint
    valid = []
    for p in prefixes:
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{p}floating_base_joint")
        if jid >= 0 and m.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
            valid.append(p)
    # если ничего не нашли, пробуем без префикса (одиночный робот)
    if not valid:
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if jid >= 0 and m.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
            valid.append("")
    valid.sort()
    return valid

# =========================
# Класс робота
# =========================
@dataclass
class Robot:
    prefix: str
    # базовые индексы
    free_jid: int = -1
    free_qpos_adr: int = -1
    free_dof_adr: int = -1

    # нижняя часть (походка)
    gait_joint_ids: List[int] = field(default_factory=list)     # joint ids ног (hinge/slide)
    gait_qpos_addrs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gait_qvel_addrs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gait_act_ids: List[int] = field(default_factory=list)       # actuator ids ног (в том же порядке)
    # верхняя часть (PD)
    arm_name_to_idx: Dict[str, int] = field(default_factory=dict)
    arm_jids: List[int] = field(default_factory=list)
    arm_act_ids: List[int] = field(default_factory=list)
    arm_qpos_addrs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    arm_qvel_addrs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    arm_targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    arm_kp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    arm_kd: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    arm_ki: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    arm_ierr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    arm_i_clamp: float = 0.7

    # буферы политики
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    action: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    target_dof_pos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    obs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # фаза шага и счётчик для decimation
    counter: int = 0
    phase_sin: float = 0.0
    phase_prev_sin: float = 0.0

    # шаги походки (блокирующие по C)
    steps_needed: int = 0
    steps_done: int = 0

    def init_indices(self):
        # free joint
        self.free_jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{self.prefix}floating_base_joint")
        if self.free_jid < 0 or m.jnt_type[self.free_jid] != mj.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"[{self.prefix}] Не найден free joint")
        self.free_qpos_adr = int(m.jnt_qposadr[self.free_jid])
        self.free_dof_adr  = int(m.jnt_dofadr[self.free_jid])

        # нижняя часть: joints, чьи qpos-адреса в окне [base+7 .. base+7+num_actions)
        q0 = self.free_qpos_adr + 7
        jids = []
        for j in range(m.njnt):
            jt = m.jnt_type[j]
            if jt not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            adr = int(m.jnt_qposadr[j])
            if q0 <= adr < q0 + num_actions:
                jids.append(j)
        # сопоставление JOINT -> ACTUATOR (tar=JOINT)
        trn = np.array(m.actuator_trnid).reshape(m.nu, 2) if m.nu > 0 else np.zeros((0, 2), dtype=int)
        trnt = np.array(m.actuator_trntype) if m.nu > 0 else np.zeros((0,), dtype=int)
        jointid_to_act = {}
        for i in range(m.nu):
            if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
                jid_t = int(trn[i, 0])
                jointid_to_act.setdefault(jid_t, i)

        gait_act = []
        gait_qpos = []
        gait_qvel = []
        ordered_jids = []
        # порядок как в возрастании qpos-адресов
        jids_sorted = sorted(jids, key=lambda j: int(m.jnt_qposadr[j]))
        for j in jids_sorted:
            aid = jointid_to_act.get(j, -1)
            if aid >= 0:
                gait_act.append(aid)
                gait_qpos.append(int(m.jnt_qposadr[j]))
                gait_qvel.append(int(m.jnt_dofadr[j]))
                ordered_jids.append(j)

        self.gait_joint_ids = ordered_jids
        self.gait_act_ids = gait_act
        self.gait_qpos_addrs = np.array(gait_qpos, dtype=int)
        self.gait_qvel_addrs = np.array(gait_qvel, dtype=int)

        # буферы политики
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = default_angles.copy()
        self.obs = np.zeros(num_obs, dtype=np.float32)

    def init_upper_pd(self, hold_zero=True):
        """Привязать ВСЕ суставы верхней части (по именам) и инициализировать PD-цели."""
        name_map = {}
        jids = []
        act_ids = []
        qpos_addrs = []
        qvel_addrs = []
        # fallback joint->act
        trn = np.array(m.actuator_trnid).reshape(m.nu, 2) if m.nu > 0 else np.zeros((0, 2), dtype=int)
        trnt = np.array(m.actuator_trntype) if m.nu > 0 else np.zeros((0,), dtype=int)
        jointid_to_act = {}
        for i in range(m.nu):
            if trnt[i] == mj.mjtTrn.mjTRN_JOINT:
                jointid_to_act[int(trn[i, 0])] = i

        # сканируем револьвенты
        pairs = []
        for j in range(m.njnt):
            if m.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                continue
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if not nm.startswith(self.prefix):
                continue
            core = nm[len(self.prefix):]
            if _is_upper_joint(core):
                aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, nm)
                if aid < 0:
                    aid = jointid_to_act.get(j, -1)
                if aid < 0:  # нет привода — пропускаем
                    continue
                pairs.append((nm, j, aid))

        # упорядочим по имени для стабильности
        pairs.sort(key=lambda x: x[0])
        for nm, jid, aid in pairs:
            idx = len(jids)
            name_map[nm] = idx
            jids.append(jid)
            act_ids.append(aid)
            qpos_addrs.append(int(m.jnt_qposadr[jid]))
            qvel_addrs.append(int(m.jnt_dofadr[jid]))

        self.arm_name_to_idx = name_map
        self.arm_jids = jids
        self.arm_act_ids = act_ids
        self.arm_qpos_addrs = np.array(qpos_addrs, dtype=int)
        self.arm_qvel_addrs = np.array(qvel_addrs, dtype=int)

        n = len(jids)
        if n == 0:
            self.arm_targets = np.zeros(0, dtype=np.float32)
            self.arm_kp = np.zeros(0, dtype=np.float32)
            self.arm_kd = np.zeros(0, dtype=np.float32)
            self.arm_ki = np.zeros(0, dtype=np.float32)
            self.arm_ierr = np.zeros(0, dtype=np.float32)
            print(f"[{self.prefix}] UPPER: 0 joints")
            return

        # начальные цели
        if hold_zero:
            self.arm_targets = np.zeros(n, dtype=np.float32)
            print(f"[{self.prefix}] UPPER: держим нули на {n} шарнирах")
        else:
            self.arm_targets = np.array([d.qpos[i] for i in self.arm_qpos_addrs], dtype=np.float32)

        # Kp/Kd/Ki по имени без префикса
        kps_list = []
        for nm in name_map.keys():
            core = nm[len(self.prefix):]
            kps_list.append(_suggest_kp(core))
        self.arm_kp = np.array(kps_list, dtype=np.float32)
        self.arm_kd = np.clip(self.arm_kp * 0.05, 0.2, None).astype(np.float32)
        self.arm_ki = np.clip(self.arm_kp * 0.02, 0.0, 1.0).astype(np.float32)
        self.arm_ierr = np.zeros_like(self.arm_kp, dtype=np.float32)

        print(f"[{self.prefix}] UPPER PD готов ({n})")

    def retune_upper_kp_by_gravity(self, eps_deg=1.0, safety=1.5):
        if self.arm_qvel_addrs.size == 0:
            return
        mj.mj_forward(m, d)
        eps = rad(eps_deg)
        bias = np.abs(d.qfrc_bias[self.arm_qvel_addrs])
        req_kp = safety * bias / max(eps, 1e-5)
        self.arm_kp = np.maximum(self.arm_kp, req_kp).astype(np.float32)
        self.arm_kd = np.clip(self.arm_kp * 0.05, 0.2, None).astype(np.float32)
        print(f"[{self.prefix}] UPPER Kp подстроен по гравитации")

    def apply_upper_pd(self):
        if not self.arm_act_ids:
            return
        q  = d.qpos[self.arm_qpos_addrs]
        dq = d.qvel[self.arm_qvel_addrs]
        err = self.arm_targets - q
        self.arm_ierr += err * m.opt.timestep
        self.arm_ierr = np.clip(self.arm_ierr, -self.arm_i_clamp, self.arm_i_clamp)
        bias = d.qfrc_bias[self.arm_qvel_addrs]
        tau_arm = bias + self.arm_kp * err - self.arm_kd * dq + self.arm_ki * self.arm_ierr

        gears = np.array(m.actuator_gear).reshape(m.nu, 6) if m.nu > 0 else None
        for i, aid in enumerate(self.arm_act_ids):
            g = float(gears[aid, 0]) if gears is not None else 1.0
            lo, hi = m.actuator_ctrlrange[aid]
            u = tau_arm[i] / max(g, 1e-6)
            d.ctrl[aid] = float(np.clip(u, lo, hi))

    def apply_gait_pd(self):
        if not self.gait_act_ids:
            return
        # q/dq ног в порядке gait_qpos_addrs
        qj = d.qpos[self.gait_qpos_addrs]
        dqj = d.qvel[self.gait_qvel_addrs]
        tau = pd_control(self.target_dof_pos, qj, kps, np.zeros_like(kds), dqj, kds)
        for k, aid in enumerate(self.gait_act_ids):
            d.ctrl[aid] = float(tau[k])

    def policy_tick(self):
        # decimation общий, но счётчик локальный
        if self.counter % control_decimation != 0:
            return
        na = num_actions
        # суставы ног
        qj  = d.qpos[self.gait_qpos_addrs];   dqj = d.qvel[self.gait_qvel_addrs]
        # базовая ориентация и омега
        quat  = d.qpos[self.free_qpos_adr + 3: self.free_qpos_adr + 7]
        omega = d.qvel[self.free_dof_adr + 3: self.free_dof_adr + 6]

        qj_scaled  = (qj - default_angles) * dof_pos_scale
        dqj_scaled = dqj * dof_vel_scale
        grav = get_gravity_orientation(quat)
        omega_scaled = omega * ang_vel_scale

        period = 0.8
        t_sim  = self.counter * simulation_dt
        phase  = (t_sim % period) / period
        sinp   = math.sin(2 * math.pi * phase)
        cosp   = math.cos(2 * math.pi * phase)

        self.obs[:3] = omega_scaled
        self.obs[3:6] = grav
        self.obs[6:9] = self.cmd * cmd_scale
        self.obs[9:9+na] = qj_scaled
        self.obs[9+na:9+2*na] = dqj_scaled
        self.obs[9+2*na:9+3*na] = self.action
        self.obs[9+3*na:9+3*na+2] = np.array([sinp, cosp], dtype=np.float32)

        with torch.no_grad():
            self.action[:] = policy(torch.from_numpy(self.obs).unsqueeze(0)).numpy().squeeze()
        self.target_dof_pos[:] = self.action * action_scale + default_angles

        self.phase_prev_sin, self.phase_sin = self.phase_sin, sinp

    # блокирующие команды на одного робота
    def start_new_steps(self, num: int, dir_deg: float, speed: float):
        self.steps_needed = int(num)
        self.steps_done = 0
        dir_rad = rad(dir_deg)
        vx = speed * math.cos(dir_rad)
        vy = speed * math.sin(dir_rad)
        self.cmd[:] = [vx, vy, 0.0]
        print(f"[{self.prefix}] Походка: {self.steps_needed} шаг(ов), dir={dir_deg}°, v={speed} м/с")

    def run_walk_until_done(self, step_all_fn):
        while self.steps_done < self.steps_needed and not glfw.window_should_close(window):
            step_all_fn()
            # фронт синуса фазы — шаг
            if self.phase_sin >= 0 and self.phase_prev_sin < 0:
                self.steps_done += 1
                print(f"[{self.prefix}] step {self.steps_done}/{self.steps_needed}")
                if self.steps_done >= self.steps_needed:
                    self.cmd[:] = 0.0
                    self.target_dof_pos[:] = default_angles

        self.cmd[:] = 0.0

    def run_turn_blocking(self, angle_deg: float, spd_deg_s: float, step_all_fn):
        direction  = 1.0 if angle_deg >= 0 else -1.0
        yaw_rate   = direction * rad(abs(spd_deg_s))
        target_abs = rad(abs(angle_deg))

        self.cmd[:] = [0.0, 0.0, yaw_rate]
        qw, qx, qy, qz = d.qpos[self.free_qpos_adr+3:self.free_qpos_adr+7]
        last_yaw = _quat_yaw(qw, qx, qy, qz)
        acc = 0.0

        while acc < target_abs and not glfw.window_should_close(window):
            step_all_fn()
            qw, qx, qy, qz = d.qpos[self.free_qpos_adr+3:self.free_qpos_adr+7]
            yaw_now = _quat_yaw(qw, qx, qy, qz)
            acc += abs(_wrap_to_pi(yaw_now - last_yaw))
            last_yaw = yaw_now

        self.cmd[:] = 0.0

# =========================
# Камера/рендер
# =========================
def render_once():
    glfw.poll_events()
    w, h = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, w, h)
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)

# =========================
# Главная логика шагов
# =========================
ROBOTS: List[Robot] = []
ACTIVE = 0
INIT_QPOS_FULL = None  # начальная поза всей сцены (qpos), чтобы частично ресетить робота

def step_all_once():
    # 1) выставляем PD для всех роботов
    for r in ROBOTS:
        r.apply_gait_pd()
        r.apply_upper_pd()
    # 2) шаг физики
    mj.mj_step(m, d)
    # 3) локальные счётчики и политика
    for r in ROBOTS:
        r.counter += 1
        r.policy_tick()
    # 4) рендер
    render_once()
    return True

# =========================
# Сброс позы одного робота
# =========================
def reset_robot_posture(r: Robot):
    """Локальный мягкий ресет: вернуть qpos/quat базы и суставов к старту (из INIT_QPOS_FULL)."""
    # восстанавливаем free-позу (7 значений) и все hinge/slide шарниры ног
    if INIT_QPOS_FULL is None:
        return
    # base (xyz + quat)
    d.qpos[r.free_qpos_adr:r.free_qpos_adr+7] = INIT_QPOS_FULL[r.free_qpos_adr:r.free_qpos_adr+7]
    # все револьвенты этого робота: пройдём по всем его jids (верх+низ)
    # проще: сбросить все hinge/slide, чьи имена начинаются с префикса
    for j in range(m.njnt):
        if m.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith(r.prefix):
                adr = int(m.jnt_qposadr[j])
                d.qpos[adr] = 0.0
        elif m.jnt_type[j] == mj.mjtJoint.mjJNT_BALL:
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith(r.prefix):
                adr = int(m.jnt_qposadr[j])
                d.qpos[adr:adr+4] = np.array([1.0, 0.0, 0.0, 0.0])
    mj.mj_forward(m, d)
    # локальные буферы
    r.counter = 0
    r.target_dof_pos = default_angles.copy()
    r.cmd[:] = 0.0
    print(f"[{r.prefix}] Reset done.")

# =========================
# Мышь/клавиатура
# =========================
is_dragging = False
last_cursor_pos = (0.0, 0.0)

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

def _active_robot() -> Robot:
    return ROBOTS[min(max(ACTIVE, 0), len(ROBOTS)-1)]

def key_callback(window_, key, scancode, action, mods):
    global ACTIVE
    pressed = (action == glfw.PRESS or action == glfw.REPEAT)
    if action == glfw.RELEASE:
        pressed = False

    if key == glfw.KEY_ESCAPE and pressed:
        glfw.set_window_should_close(window_, True)
        return

    # Переключение активного робота 1..9
    if pressed and glfw.KEY_1 <= key <= glfw.KEY_9:
        idx = key - glfw.KEY_1  # 0..8
        if idx < len(ROBOTS):
            ACTIVE = idx
            pf = _active_robot().prefix or "<no-prefix>"
            glfw.set_window_title(window_, f"MuJoCo Multi ({idx+1}/{len(ROBOTS)}: {pf})")
            print(f"Активный робот: {idx+1}/{len(ROBOTS)} [{pf}]")
        return

    r = _active_robot()

    # Поворот (T) — только активный
    if key == glfw.KEY_T and pressed:
        try:
            ang = float(input("Turn angle (deg, +left / -right): "))
            spd = input("Speed (deg/s) [45]: ").strip()
            spd = float(spd) if spd else 45.0
        except Exception:
            print("Invalid input.")
            return
        print(f"[{r.prefix}] Поворот на {ang}° при {spd}°/с…")
        r.run_turn_blocking(ang, spd, step_all_once)
        return

    # Походка (C) — только активный
    if key == glfw.KEY_C and pressed:
        try:
            num = int(input("Number of steps (>0): "))
            dir_deg = float(input("Direction (deg, 0°=+X, 90°=+Y): "))
            spd = float(input("Speed (m/s): "))
        except Exception:
            print("Invalid input.")
            return
        if num <= 0:
            print("0 steps, ignore.")
            return
        r.start_new_steps(num, dir_deg, spd)
        r.run_walk_until_done(step_all_once)
        return

    # Локальный сброс активного робота (R)
    if key == glfw.KEY_R and pressed:
        reset_robot_posture(r)
        return

    # Установка PD-угла верхнего сустава (J) — только активный
    if key == glfw.KEY_J and pressed:
        if not r.arm_name_to_idx:
            print("No upper-body joints bound.")
            return
        print(f"Available upper joints for [{r.prefix or '<no-prefix>'}]:")
        # печатаем без префикса
        for nm in sorted(r.arm_name_to_idx.keys()):
            core = nm[len(r.prefix):]
            print("  -", core)
        try:
            nm_in = input("Joint name: ").strip()
            full = nm_in if nm_in.startswith(r.prefix) else f"{r.prefix}{nm_in}"
            if full not in r.arm_name_to_idx:
                print("Unknown joint.")
                return
            ang = float(input("Target angle (deg): "))
            idx = r.arm_name_to_idx[full]
            r.arm_targets[idx] = rad(ang)
            print(f"[{r.prefix}] {nm_in} -> {ang:.1f} deg (PD target).")
        except Exception:
            print("Invalid input.")
        return

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default="h1_2")
    parser.add_argument("--xml", required=True, help="robot or scene XML filename in its folder")
    parser.add_argument("--upper_zero", action="store_true", help="hold zero on upper joints")
    args = parser.parse_args()

    name_ = args.robot
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path    = f"unitree_rl_gym/resources/robots/{name_}/{args.xml}"

    # 1) конфиг (общий для всех копий)
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

    # 2) модель/данные
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt
    mj.mj_forward(m, d)

    # 3) политика (общая)
    policy = torch.jit.load(policy_path)

    # 4) окно
    glfw.init()
    window = glfw.create_window(1280, 900, "MuJoCo Multi", None, None)
    glfw.make_context_current(window)

    cam = mj.MjvCamera();  cam.azimuth = 180; cam.elevation = -15; cam.distance = 5.0
    cam.lookat = np.array([0.0, 0.0, 0.8])
    opt = mj.MjvOption()
    scene = mj.MjvScene(m, maxgeom=10000)
    context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # 5) детект всех роботов
    prefixes = detect_robot_prefixes()
    if not prefixes:
        raise RuntimeError("Роботы в сцене не найдены.")
    print("Найдены роботы:", [p or "<no-prefix>" for p in prefixes])

    # 6) инициализация роботов
    ROBOTS = []
    for p in prefixes:
        r = Robot(prefix=p)
        r.init_indices()
        r.init_upper_pd(hold_zero=args.upper_zero or True)
        r.retune_upper_kp_by_gravity()
        ROBOTS.append(r)

    # 7) буфер начальной позы всей сцены
    INIT_QPOS_FULL = d.qpos.copy()

    # 8) главный цикл
    glfw.set_window_title(window, f"MuJoCo Multi (1/{len(ROBOTS)}: {ROBOTS[0].prefix or '<no-prefix>'})")
    while not glfw.window_should_close(window):
        step_all_once()

    glfw.terminate()
