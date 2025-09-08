from pathlib import Path
from ultralytics import YOLO

def write_dataset_yaml(path: Path, split_root: Path, nc: int, names: list[str]) -> None:
    content = f"""path: .
train: {split_root.as_posix()}/train/images
val:   {split_root.as_posix()}/val/images
test:  {split_root.as_posix()}/test/images
nc: {nc}
names: {names}
"""
    Path("dataset.yaml").write_text(content, encoding="utf-8")

def train(model_name: str, imgsz: int, batch: int, epochs: int, data_yaml: str = "dataset.yaml", device: str | int | None = None):
    model = YOLO(model_name)
    model.train(data=data_yaml, imgsz=imgsz, batch=batch, epochs=epochs, device=device)
