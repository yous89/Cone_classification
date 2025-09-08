from pathlib import Path
from PIL import Image
import json
from .paths import ensure_dirs
from .convert import convert_label_to_yolo

def prepare_dataset(base_path: Path, img_out: Path, lbl_out: Path, class_map: dict[str,int]) -> int:
    ensure_dirs(img_out, lbl_out)
    count = 0
    for sub in base_path.iterdir():
        if not sub.is_dir(): 
            continue
        img_dir = sub / "img"
        ann_dir = sub / "ann"
        if not img_dir.exists() or not ann_dir.exists():
            print(f"⚠️ Überspringe {sub.name} (img/ann fehlt)")
            continue
        for png in img_dir.glob("*.png"):
            json_path = ann_dir / f"{png.name}.json"
            if not json_path.exists():
                print(f"❌ Kein JSON für: {png}")
                continue
            try:
                img = Image.open(png).convert("RGB")
                (img_out / png.name).parent.mkdir(parents=True, exist_ok=True)
                img.save(img_out / png.name)

                label_data = json.loads(json_path.read_text(encoding="utf-8"))
                yolo_txt = convert_label_to_yolo(label_data, class_map)
                (lbl_out / (png.stem + ".txt")).write_text(yolo_txt, encoding="utf-8")
                count += 1
            except Exception as e:
                print(f"⚠️ Fehler bei {png}: {e}")
    print(f"\n✅ Fertig: {count} Bilder + Labels gespeichert.")
    return count
