from ultralytics import YOLO

def evaluate(weights: str, data_yaml: str = "dataset.yaml", plots: bool = True):
    m = YOLO(weights)
    r = m.val(data=data_yaml, plots=plots)
    print("mAP50:", r.box.map50, "mAP50-95:", r.box.map)
    print("Per-class mAP50:", r.box.maps)
