from __future__ import annotations
import json
from pathlib import Path

def convert_label_to_yolo(json_data: dict, class_map: dict[str, int]) -> str:
    img_w = json_data["size"]["width"]
    img_h = json_data["size"]["height"]
    lines: list[str] = []

    for obj in json_data.get("objects", []):
        cls = obj.get("classTitle")
        if cls not in class_map:
            continue
        class_id = class_map[cls]
        (x1, y1), (x2, y2) = obj["points"]["exterior"]
        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)
