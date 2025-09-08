from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ProjectConfig:
    dataset_path: Path
    output_images_train: Path
    output_labels_train: Path
    output_images_sel: Path
    output_labels_sel: Path
    merged_images: Path
    merged_labels: Path
    class_map: dict[str, int]
    targets_for_selection: set[int]
    yolo_model: str
    yolo_imgsz: int
    yolo_batch: int
    yolo_epochs: int
    yolo_best: Path
    split_root: Path
    test_image: Path

def load_config(path: str | Path = "config/config.yaml") -> ProjectConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return ProjectConfig(
        dataset_path=Path(data["dataset_path"]),
        output_images_train=Path(data["output"]["images_train"]),
        output_labels_train=Path(data["output"]["labels_train"]),
        output_images_sel=Path(data["output"]["images_sel"]),
        output_labels_sel=Path(data["output"]["labels_sel"]),
        merged_images=Path(data["output"]["merged_images"]),
        merged_labels=Path(data["output"]["merged_labels"]),
        class_map=data["classes"],
        targets_for_selection=set(data["targets_for_selection"]),
        yolo_model=data["yolo"]["model"],
        yolo_imgsz=int(data["yolo"]["imgsz"]),
        yolo_batch=int(data["yolo"]["batch"]),
        yolo_epochs=int(data["yolo"]["epochs"]),
        yolo_best=Path(data["yolo"]["best_weights"]),
        split_root=Path(data["yolo"]["split_root"]),
        test_image=Path(data["yolo"]["test_image"]),
    )
