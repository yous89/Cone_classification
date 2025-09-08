# coneset/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys

def _load_config():
    # Lazy import, damit coneset wenigstens startet, auch wenn Module fehlen
    try:
        from .config import load_config
        return load_config()
    except Exception as e:
        print(f"⚠️ Konnte config laden nicht. Erwarte 'config/config.yaml'. Details: {e}")
        return None

def _cmd_prepare():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .prepare import prepare_dataset
        prepare_dataset(cfg.dataset_path, cfg.output_images_train, cfg.output_labels_train, cfg.class_map)
    except ImportError as e:
        print("❌ Modul fehlt: prepare.py. Bitte Datei erstellen (siehe vorherige Anleitung).")
        print(e); sys.exit(1)

def _cmd_stats():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .stats import count_instances, plot_counts
        id_to_class = {v: k for k, v in cfg.class_map.items()}
        counts = count_instances(cfg.output_labels_train, id_to_class)
        print("Counts:", counts)
        plot_counts(counts)
    except ImportError as e:
        print("❌ Modul fehlt: stats.py."); print(e); sys.exit(1)

def _cmd_select():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .select import select_images_with_classes
        select_images_with_classes(
            cfg.output_labels_train,
            cfg.output_images_train,
            cfg.output_images_sel,
            cfg.output_labels_sel,
            cfg.targets_for_selection,
        )
    except ImportError as e:
        print("❌ Modul fehlt: select.py."); print(e); sys.exit(1)

def _cmd_augment():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .augment import run_augmentation
        out_img = Path("aug_orange/orange_bilder")
        out_lbl = Path("aug_orange/orange_labels")
        run_augmentation(cfg.output_images_sel, cfg.output_labels_sel, out_img, out_lbl)
    except ImportError as e:
        print("❌ Modul fehlt: augment.py."); print(e); sys.exit(1)

def _cmd_merge():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .merge import merge_datasets
        merge_datasets(
            cfg.output_images_train, cfg.output_labels_train,
            Path("images/orange_augment"), Path("labels/orange_augment"),
            cfg.merged_images, cfg.merged_labels
        )
    except ImportError as e:
        print("❌ Modul fehlt: merge.py."); print(e); sys.exit(1)

def _cmd_train(epochs: int | None):
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .yolo_train import write_dataset_yaml, train
        names = list(cfg.class_map.keys())
        write_dataset_yaml(Path("dataset.yaml"), cfg.split_root, nc=len(names), names=names)
        ep = epochs or cfg.yolo_epochs
        train(cfg.yolo_model, cfg.yolo_imgsz, cfg.yolo_batch, ep)
    except ImportError as e:
        print("❌ Modul fehlt: yolo_train.py."); print(e); sys.exit(1)

def _cmd_predict(img: str | None, out: str):
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .yolo_predict import predict
        target = Path(img) if img else cfg.test_image
        predict(cfg.yolo_best, target, out)
    except ImportError as e:
        print("❌ Modul fehlt: yolo_predict.py."); print(e); sys.exit(1)

def _cmd_eval():
    cfg = _load_config()
    if cfg is None:
        sys.exit(1)
    try:
        from .yolo_eval import evaluate
        evaluate(str(cfg.yolo_best))
    except ImportError as e:
        print("❌ Modul fehlt: yolo_eval.py."); print(e); sys.exit(1)

def main():
    ap = argparse.ArgumentParser(prog="coneset", description="Cone dataset toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="JSON->YOLO konvertieren & Bilder/Labels schreiben")
    sub.add_parser("stats", help="Klassen zählen & Balkendiagramm anzeigen")
    sub.add_parser("select", help="Bilder mit Zielklassen auswählen")
    sub.add_parser("augment", help="Auswahl augmentieren")
    sub.add_parser("merge", help="Original + Augment zusammenführen")

    p_train = sub.add_parser("train", help="YOLO Training starten")
    p_train.add_argument("--epochs", type=int, default=None)

    p_pred = sub.add_parser("predict", help="Einzelbild vorhersagen")
    p_pred.add_argument("--img", type=str, default=None)
    p_pred.add_argument("--out", type=str, default="runs/predict_cli")

    sub.add_parser("eval", help="Modell evaluieren")

    args = ap.parse_args()
    if args.cmd == "prepare": _cmd_prepare()
    elif args.cmd == "stats": _cmd_stats()
    elif args.cmd == "select": _cmd_select()
    elif args.cmd == "augment": _cmd_augment()
    elif args.cmd == "merge": _cmd_merge()
    elif args.cmd == "train": _cmd_train(args.epochs)
    elif args.cmd == "predict": _cmd_predict(args.img, args.out)
    elif args.cmd == "eval": _cmd_eval()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
