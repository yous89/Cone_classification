from ultralytics import YOLO
import cv2
from pathlib import Path

def predict(weights: str | Path, img_path: str | Path, out_dir: str | Path, conf: float = 0.25, device: str | int | None = None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    results = model.predict(source=str(img_path), conf=conf, device=device)
    for i, r in enumerate(results):
        im_annotated = r.plot()
        out_path = out_dir / f"{Path(img_path).stem}_annotated_{i}.jpg"
        cv2.imwrite(str(out_path), im_annotated)
        print("gespeichert:", out_path)
