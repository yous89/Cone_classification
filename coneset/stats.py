from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

def count_instances(labels_dir: Path, id_to_class: dict[int, str]) -> dict[str,int]:
    counts = Counter()
    for txt in labels_dir.rglob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip(): 
                continue
            try:
                cid = int(line.split()[0])
            except Exception:
                continue
            counts[id_to_class.get(cid, "unknown")] += 1
    return dict(counts)

def plot_counts(counts: dict[str,int], title: str = "Anzahl Instanzen pro Klasse") -> None:
    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    plt.figure(figsize=(8,6))
    plt.bar(classes, values)
    plt.xlabel("Klasse"); plt.ylabel("Anzahl Instanzen")
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(i, v + max(1, max(values))*0.01, str(v), ha="center")
    plt.tight_layout()
    plt.show()
