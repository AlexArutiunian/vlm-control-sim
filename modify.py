# add_arm_joints.py
# Использование: python add_arm_joints.py input.xml output.xml
import sys
import xml.etree.ElementTree as ET

def find_joint(root, name):
    for j in root.iter('joint'):
        if j.get('name') == name:
            return j
    return None

def find_body(root, name):
    for b in root.iter('body'):
        if b.get('name') == name:
            return b
    return None

def insert_joint_into_body(body_elem, joint_elem):
    # вставить до первого <geom>, сразу после <inertial> (если он есть)
    idx = 0
    for i, ch in enumerate(list(body_elem)):
        if ch.tag == 'inertial':
            idx = i + 1
            break
    for i, ch in enumerate(list(body_elem)[idx:], start=idx):
        if ch.tag == 'geom':
            body_elem.insert(i, joint_elem)
            return
    body_elem.insert(idx, joint_elem)

def ensure_motor(actuator_elem, joint_name):
    for m in actuator_elem.findall('motor'):
        if m.get('joint') == joint_name:
            return
    ET.SubElement(actuator_elem, 'motor', {'name': joint_name, 'joint': joint_name})

def set_attrs_from_spec(joint_elem, spec):
    # pos пусть будет 0 0 0 (как в твоём XML)
    joint_elem.set('pos', '0 0 0')
    for k in ('axis', 'range', 'actuatorfrcrange'):
        v = spec.get(k)
        if v is not None:
            joint_elem.set(k, v)

# ----- ПАРАМЕТРЫ ИЗ "ТВОЕГО XML НИЖЕ" -----

JOINT_SPECS = {
    # Туловище
    'torso_joint': {'axis': '0 0 1', 'range': '-2.35 2.35', 'actuatorfrcrange': '-200 200'},

    # ЛЕВАЯ РУКА
    'left_shoulder_pitch_joint': {'axis': '0 1 0', 'range': '-3.14 1.57', 'actuatorfrcrange': '-40 40'},
    'left_shoulder_roll_joint':  {'axis': '1 0 0', 'range': '-0.38 3.4', 'actuatorfrcrange': '-40 40'},
    'left_shoulder_yaw_joint':   {'axis': '0 0 1', 'range': '-2.66 3.01', 'actuatorfrcrange': '-18 18'},
    'left_elbow_pitch_joint':    {'axis': '0 1 0', 'range': '-0.95 3.18', 'actuatorfrcrange': '-18 18'},
    # В референсном XML нет elbow_roll; используем параметры wrist_roll
    'left_elbow_roll_joint':     {'axis': '1 0 0', 'range': '-3.01 2.75', 'actuatorfrcrange': '-19 19'},
    'left_wrist_pitch_joint':    {'axis': '0 1 0', 'range': '-0.4625 0.4625', 'actuatorfrcrange': '-19 19'},
    'left_wrist_yaw_joint':      {'axis': '0 0 1', 'range': '-1.27 1.27', 'actuatorfrcrange': '-19 19'},

    # ПАЛЬЦЫ ЛЕВОЙ КИСТИ
    'L_thumb_proximal_yaw_joint':   {'axis': '0 0 1',  'range': '-0.1 1.3', 'actuatorfrcrange': '-1 1'},
    'L_thumb_proximal_pitch_joint': {'axis': '0 0 -1', 'range': '-0.1 0.6', 'actuatorfrcrange': '-1 1'},
    'L_thumb_intermediate_joint':   {'axis': '0 0 -1', 'range': '0 0.8',   'actuatorfrcrange': '-1 1'},
    'L_thumb_distal_joint':         {'axis': '0 0 -1', 'range': '0 1.2',   'actuatorfrcrange': '-1 1'},

    'L_index_proximal_joint':       {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'L_index_intermediate_joint':   {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'L_middle_proximal_joint':      {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'L_middle_intermediate_joint':  {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'L_ring_proximal_joint':        {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'L_ring_intermediate_joint':    {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'L_pinky_proximal_joint':       {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'L_pinky_intermediate_joint':   {'axis': '0 0 -1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    # ПРАВАЯ РУКА
    'right_shoulder_pitch_joint': {'axis': '0 1 0', 'range': '-3.14 1.57', 'actuatorfrcrange': '-40 40'},
    'right_shoulder_roll_joint':  {'axis': '1 0 0', 'range': '-3.4 0.38',  'actuatorfrcrange': '-40 40'},
    'right_shoulder_yaw_joint':   {'axis': '0 0 1', 'range': '-3.01 2.66', 'actuatorfrcrange': '-18 18'},
    'right_elbow_pitch_joint':    {'axis': '0 1 0', 'range': '-0.95 3.18', 'actuatorfrcrange': '-18 18'},
    # аналоги elbow_roll — параметры wrist_roll
    'right_elbow_roll_joint':     {'axis': '1 0 0', 'range': '-2.75 3.01', 'actuatorfrcrange': '-19 19'},
    'right_wrist_pitch_joint':    {'axis': '0 1 0', 'range': '-0.4625 0.4625', 'actuatorfrcrange': '-19 19'},
    'right_wrist_yaw_joint':      {'axis': '0 0 1', 'range': '-1.27 1.27',    'actuatorfrcrange': '-19 19'},

    # ПАЛЬЦЫ ПРАВОЙ КИСТИ
    'R_thumb_proximal_yaw_joint':   {'axis': '0 0 -1', 'range': '-0.1 1.3', 'actuatorfrcrange': '-1 1'},
    'R_thumb_proximal_pitch_joint': {'axis': '0 0 1',  'range': '-0.1 0.6', 'actuatorfrcrange': '-1 1'},
    'R_thumb_intermediate_joint':   {'axis': '0 0 1',  'range': '0 0.8',    'actuatorfrcrange': '-1 1'},
    'R_thumb_distal_joint':         {'axis': '0 0 1',  'range': '0 1.2',    'actuatorfrcrange': '-1 1'},

    'R_index_proximal_joint':       {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'R_index_intermediate_joint':   {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'R_middle_proximal_joint':      {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'R_middle_intermediate_joint':  {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'R_ring_proximal_joint':        {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'R_ring_intermediate_joint':    {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},

    'R_pinky_proximal_joint':       {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
    'R_pinky_intermediate_joint':   {'axis': '0 0 1', 'range': '0 1.7', 'actuatorfrcrange': '-1 1'},
}

# Куда вставлять каждый сустав (body -> joint)
TARGETS = [
    # Торс
    ('torso_link', 'torso_joint'),

    # Левая рука
    ('left_shoulder_pitch_link', 'left_shoulder_pitch_joint'),
    ('left_shoulder_roll_link',  'left_shoulder_roll_joint'),
    ('left_shoulder_yaw_link',   'left_shoulder_yaw_joint'),
    ('left_elbow_pitch_link',    'left_elbow_pitch_joint'),
    ('left_elbow_roll_link',     'left_elbow_roll_joint'),
    ('left_wrist_pitch_link',    'left_wrist_pitch_joint'),
    ('left_wrist_yaw_link',      'left_wrist_yaw_joint'),

    # Пальцы левой кисти
    ('L_thumb_proximal_base', 'L_thumb_proximal_yaw_joint'),
    ('L_thumb_proximal',      'L_thumb_proximal_pitch_joint'),
    ('L_thumb_intermediate',  'L_thumb_intermediate_joint'),
    ('L_thumb_distal',        'L_thumb_distal_joint'),

    ('L_index_proximal',      'L_index_proximal_joint'),
    ('L_index_intermediate',  'L_index_intermediate_joint'),

    ('L_middle_proximal',     'L_middle_proximal_joint'),
    ('L_middle_intermediate', 'L_middle_intermediate_joint'),

    ('L_ring_proximal',       'L_ring_proximal_joint'),
    ('L_ring_intermediate',   'L_ring_intermediate_joint'),

    ('L_pinky_proximal',      'L_pinky_proximal_joint'),
    ('L_pinky_intermediate',  'L_pinky_intermediate_joint'),

    # Правая рука
    ('right_shoulder_pitch_link', 'right_shoulder_pitch_joint'),
    ('right_shoulder_roll_link',  'right_shoulder_roll_joint'),
    ('right_shoulder_yaw_link',   'right_shoulder_yaw_joint'),
    ('right_elbow_pitch_link',    'right_elbow_pitch_joint'),
    ('right_elbow_roll_link',     'right_elbow_roll_joint'),
    ('right_wrist_pitch_link',    'right_wrist_pitch_joint'),
    ('right_wrist_yaw_link',      'right_wrist_yaw_joint'),

    # Пальцы правой кисти
    ('R_thumb_proximal_base', 'R_thumb_proximal_yaw_joint'),
    ('R_thumb_proximal',      'R_thumb_proximal_pitch_joint'),
    ('R_thumb_intermediate',  'R_thumb_intermediate_joint'),
    ('R_thumb_distal',        'R_thumb_distal_joint'),

    ('R_index_proximal',      'R_index_proximal_joint'),
    ('R_index_intermediate',  'R_index_intermediate_joint'),

    ('R_middle_proximal',     'R_middle_proximal_joint'),
    ('R_middle_intermediate', 'R_middle_intermediate_joint'),

    ('R_ring_proximal',       'R_ring_proximal_joint'),
    ('R_ring_intermediate',   'R_ring_intermediate_joint'),

    ('R_pinky_proximal',      'R_pinky_proximal_joint'),
    ('R_pinky_intermediate',  'R_pinky_intermediate_joint'),
]

def upsert_joint(root, actuator, body_name, joint_name, spec):
    body = find_body(root, body_name)
    if body is None:
        return False  # такое звено отсутствует — пропускаем

    existing = find_joint(root, joint_name)
    if existing is not None:
        # Обновим атрибуты по "референсу"
        set_attrs_from_spec(existing, spec)
        ensure_motor(actuator, joint_name)
        return True

    # Создаём новый сустав
    j = ET.Element('joint', {'name': joint_name})
    set_attrs_from_spec(j, spec)
    insert_joint_into_body(body, j)
    ensure_motor(actuator, joint_name)
    return True

def main(inp, outp):
    tree = ET.parse(inp)
    root = tree.getroot()

    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')

    created_or_updated = []
    for body_name, joint_name in TARGETS:
        spec = JOINT_SPECS.get(joint_name)
        if not spec:
            continue
        ok = upsert_joint(root, actuator, body_name, joint_name, spec)
        if ok:
            created_or_updated.append(joint_name)

    # Сохраняем
    tree.write(outp, encoding='utf-8', xml_declaration=True)
    print(f'Updated/created {len(created_or_updated)} joints:')
    for n in created_or_updated:
        print(' -', n)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        inp, outp = sys.argv[1], sys.argv[2]
    else:
        # значения по умолчанию под твой текущий аплоад
        inp = "unitree_rl_gym/resources/robots/h1_2/h1_2_12dof.xml"
        outp = "unitree_rl_gym/resources/robots/h1_2/h1_2_full.xml"
    main(inp, outp)

