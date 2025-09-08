# Cone detection and classification

> Saubere Modul- & Paketstruktur zum Vorbereiten, Analysieren, Augmentieren, Mergen und Trainieren von YOLO-Datensätzen (Ultralytics) – speziell für Verkehrs-/Rennkegel.

---

## Inhalt

* [Features](#features)
* [Projektstruktur](#projektstruktur)
* [Voraussetzungen](#voraussetzungen)
* [Installation](#installation)
* [Konfiguration](#konfiguration)
* [Schnellstart](#schnellstart)
* [CLI-Befehle](#cli-befehle)
* [Datenformate](#datenformate)
---

## Features

* **Konvertierung**: JSON-Annotationen → YOLO-Format.
* **Vorbereitung**: Bilder/Labels in saubere Ordnerstruktur schreiben.
* **Statistiken**: Klassenverteilung zählen & bar chart plotten.
* **Selektion**: Teilmengen nach Klassen-IDs herausziehen (z. B. orange/large\_orange).
* **Augmentierung**: Helligkeit/Kontrast/Sättigung/Farbton (TensorFlow) – Labels werden mitkopiert.
* **Mergen**: Original- und augmentierte Daten kollisionsfrei zusammenführen.
* **Training**: Ultralytics YOLO trainieren (v8).
* **Evaluation & Inferenz**: mAP, per-class mAP, annotierte Ergebnisbilder speichern.
* **Konfigurierbar** via `config/config.yaml` (keine Hardcodierten Pfade!).

---

## Projektstruktur

```
hütchen/
├─ pyproject.toml                 # Projekt-/Abhängigkeiten (oder requirements.txt)
├─ README.md
├─ config/
│  └─ config.yaml                 # zentrale Pfade & Settings
├─ dataset.yaml                   # YOLO data file (wird generiert)
├─ coneset/                       # Python-Paket
│  ├─ __init__.py
│  ├─ config.py                   # Config laden/validieren
│  ├─ paths.py                    # Pfad-Utilities
│  ├─ convert.py                  # JSON -> YOLO
│  ├─ prepare.py                  # Dataset vorbereiten
│  ├─ stats.py                    # Zählen + Plotten
│  ├─ select.py                   # Teilmengen selektieren
│  ├─ augment.py                  # Augmentierung
│  ├─ merge.py                    # Datasets mergen
│  ├─ yolo_train.py               # Training (Ultralytics)
│  ├─ yolo_predict.py             # Inferenz
│  ├─ yolo_eval.py                # Validierung/Eval
│  └─ cli.py                      # Subcommand-CLI
└─ scripts/
   └─ run_examples.ps1            # optionale Beispiele/One-liners
```

---

## Voraussetzungen

* **Python** ≥ 3.10
* **Pip**/**venv**
* **Ultralytics** (YOLOv8)
* **Pillow**, **Matplotlib**, **PyYAML**
* **TensorFlow** (optional, für die einfache Farb-Augmentierung)

  * Hinweis: Unter Windows kann die TF-Installation je nach Python-Version abweichen. Alternativ kannst du Augmentierungen mit Albumentations o. Ä. umsetzen.

---

## Installation

```bash
# Repository klonen
git clone https://github.com/yous89/Cone_classification.git CONE-PROJECT
cd CONE-PROJECT


#  mit pyproject.toml
pip install -e .

> **Tipp:** Verwende ein virtuelles Environment (`python -m venv .venv` & `source .venv/bin/activate` bzw. `./.venv/Scripts/activate`).

---

## Konfiguration

Alle Pfade und Hyperparameter zentral in `config/config.yaml`:

```yaml
dataset_path: "C:/Users/youse/OneDrive/Desktop/Hütchen/fsoco_bounding_boxes_train"
output:
  images_train: "images/train2"
  labels_train: "labels/train2"
  images_sel:   "images/orange_augment"
  labels_sel:   "labels/orange_augment"
  merged_images: "dataset_merged/images"
  merged_labels: "dataset_merged/labels"
classes:
  blue_cone: 0
  yellow_cone: 1
  orange_cone: 2
  large_orange_cone: 3
targets_for_selection: [2, 3]
yolo:
  imgsz: 640
  batch: 16
  epochs: 50
  model: "yolov8n.pt"
  best_weights: "runs/detect/train9/weights/best.pt"
  split_root: "dataset_split"
  test_image: "dataset_split/test/images/pwrrt_00148.png"
```

---

## Schnellstart

```bash
# 1) Datensatz vorbereiten: Bilder+Labels in Zielordner schreiben
coneset prepare

# 2) Klassen zählen & Plot anzeigen
coneset stats

# 3) Bilder mit Zielklassen (z. B. orange) herausziehen
coneset select

# 4) Farb-Augmentierung erzeugen
coneset augment

# 5) Original + Augmentiert mergen
coneset merge

# 6) YOLO-Datafile schreiben & trainieren
coneset train --epochs 10

# 7) Prediction mit annotiertem Output
coneset predict --img "dataset_split/test/images/amz_00733_1.png" --out runs/predict_cli

# 8) Evaluation (mAP etc.)
coneset eval
```

---

## CLI-Befehle

Die CLI ist über den Konsolenbefehl `coneset` verfügbar. Subcommands:

| Befehl    | Beschreibung                                                                        |
| --------- | ----------------------------------------------------------------------------------- |
| `prepare` | Liest Rohdaten (img/ann), speichert Bilder & YOLO-Labels in konfigurierten Ordnern. |
| `stats`   | Zählt Instanzen pro Klasse und zeigt ein Balkendiagramm.                            |
| `select`  | Selektiert Bilder, die bestimmte Klassen-IDs enthalten (z. B. 2 & 3).               |
| `augment` | Einfache Farb-Augmentierung (TF) erzeugt und Labels mitkopieren.                    |
| `merge`   | Führt Original- und Augment-Daten kollisionsfrei in `dataset_merged/` zusammen.     |
| `train`   | Schreibt `dataset.yaml` und startet YOLOv8-Training (Ultralytics).                  |
| `predict` | Führt Inferenz aus und speichert annotierte Bilder.                                 |
| `eval`    | Validiert das Modell und druckt mAP-Werte.                                          |

> Details zu Pfaden/Klassen stammen aus `config/config.yaml`.

---

## Datenformate

### Eingabe (Rohdaten)

Erwartete Ordnerstruktur pro Subset:

```
<subset>/
  ├─ img/   # *.png
  └─ ann/   # *.png.json (Annotationen)
```

**JSON-Felder (vereinfachtes Beispiel):**

```json
{
  "size": {"width": 1920, "height": 1080},
  "objects": [
    {
      "classTitle": "orange_cone",
      "points": {"exterior": [[x1, y1], [x2, y2]]}
    }
  ]
}
```

### Ausgabe (YOLO-Labels)

Datei pro Bild: `<bildname>.txt` mit Zeilen:

```
<class_id> <x_center> <y_center> <width> <height>
```

Alle Werte **normalisiert** auf \[0,1].

---
