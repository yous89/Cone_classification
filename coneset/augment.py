from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import shutil

def augment_tf(img: Image.Image) -> Image.Image:
    x = tf.convert_to_tensor(np.array(img))
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.random_brightness(x, 0.15)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.image.random_hue(x, 0.03)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = (x.numpy() * 255).astype(np.uint8)
    return Image.fromarray(x)

def run_augmentation(src_images: Path, src_labels: Path, out_images: Path, out_labels: Path) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src_images.rglob("*"):
        if p.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}:
            continue
        aug = augment_tf(Image.open(p).convert("RGB"))
        out_img = out_images / f"{p.stem}_aug.jpg"
        aug.save(out_img, quality=95)
        lab = src_labels / f"{p.stem}.txt"
        if lab.exists():
            shutil.copy2(lab, out_labels / f"{out_img.stem}.txt")
        count += 1
    print(f"✅ Augmentierte Bilder: {count}")
    return count
