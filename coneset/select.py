from pathlib import Path
import shutil

def select_images_with_classes(labels_dir: Path, images_dir: Path, out_img: Path, out_lbl: Path, target_ids: set[int]) -> int:
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    selected = 0
    for lbl in labels_dir.glob("*.txt"):
        lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if any(int(ln.split()[0]) in target_ids for ln in lines):
            img = images_dir / (lbl.stem + ".png")
            if img.exists():
                shutil.copy2(img, out_img / img.name)
                shutil.copy2(lbl, out_lbl / (lbl.name))
                selected += 1
            else:
                print(f"⚠️ Bild nicht gefunden: {img}")
    print(f"\n✅ {selected} Bilder mit Zielklassen kopiert → {out_img}")
    return selected
