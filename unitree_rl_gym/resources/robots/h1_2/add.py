import xml.etree.ElementTree as ET

# === Путь к исходному XML ===
src = "h1_2_12dof.xml"
dst = "h1_2_12dof_with_sites.xml"

tree = ET.parse(src)
root = tree.getroot()

# === список имен терминальных сегментов пальцев ===
finger_tips = [
    "L_thumb_distal",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_ring_intermediate",
    "L_pinky_intermediate",
    "R_thumb_distal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_ring_intermediate",
    "R_pinky_intermediate",
]

# === добавляем site в каждый нужный body ===
for body in root.iter("body"):
    name = body.get("name", "")
    if name in finger_tips:
        site = ET.SubElement(body, "site")
        site.set("name", f"{name}_tip_site")
        site.set("pos", "0 0 0")          # в локальной системе пальца
        site.set("size", "0.005")         # маленький шар
        site.set("rgba", "1 0 0 1")       # красный цвет (можно менять)
        site.set("type", "sphere")

# === сохраняем в новый файл ===
tree.write(dst, encoding="utf-8", xml_declaration=True)
print(f"Sites добавлены, новый файл сохранён как {dst}")
