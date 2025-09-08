from pathlib import Path
import shutil

def copy_pair(img: Path, label_dir: Path, dst_img: Path, dst_lbl: Path) -> int:
    lab = label_dir / (img.stem + ".txt")
    if not lab.exists():
        return 0
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    i = 1
    out_img = dst_img / img.name
    out_lbl = dst_lbl / (img.stem + ".txt")
    while out_img.exists() or out_lbl.exists():
        out_img = dst_img / f"{img.stem}_{i}{img.suffix}"
        out_lbl = dst_lbl / f"{img.stem}_{i}.txt"
        i += 1
    shutil.copy2(img, out_img)
    shutil.copy2(lab, out_lbl)
    return 1

def merge_datasets(orig_img: Path, orig_lbl: Path, aug_img: Path, aug_lbl: Path, dst_img: Path, dst_lbl: Path) -> int:
    copied = 0
    for p in orig_img.rglob("*.png"):
        copied += copy_pair(p, orig_lbl, dst_img, dst_lbl)
    for p in aug_img.rglob("*.png"):
        copied += copy_pair(p, aug_lbl, dst_img, dst_lbl)
    print(f"Fertig. Kopierte Paare: {copied}")
    print("Bilder →", dst_img.resolve())
    print("Labels →", dst_lbl.resolve())
    return copied
