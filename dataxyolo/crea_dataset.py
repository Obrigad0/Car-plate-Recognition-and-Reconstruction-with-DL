import os
import shutil
import random
from PIL import Image

# Configurazioni
images_path = 'F:\\progetto computer vision\\dataset\\CCPD2019\\ccpd_base'  # path immagini originali
dataset_path = 'F:\\progetto computer vision\\dataxyolo'

train_ratio = 0.5

# Crea cartelle YOLO
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)

def extract_bbox_from_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('-')
    coords = []
    for part in parts:
        items = part.split('_')
        for item in items:
            if '&' in item:
                x, y = item.split('&')
                coords.append((int(x), int(y)))
    if not coords:
        print(f"Nessuna coordinata trovata in {filename}")
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    return xmin, ymin, xmax, ymax

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# Lista immagini
images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
random.seed(42)
random.shuffle(images)

split_idx = int(len(images) * train_ratio)
train_images = images[:split_idx]
val_images = images[split_idx:]

def process_and_save(images_list, split):
    for filename in images_list:
        bbox = extract_bbox_from_filename(filename)
        if bbox is None:
            continue

        # Leggi dimensioni immagine
        img_path = os.path.join(images_path, filename)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        x_center, y_center, w, h = convert_bbox_to_yolo(*bbox, img_w, img_h)

        # Copia immagine
        dst_img_path = os.path.join(dataset_path, f'images/{split}', filename)
        shutil.copy(img_path, dst_img_path)

        # Crea label
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(dataset_path, f'labels/{split}', label_filename)
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

process_and_save(train_images, 'train')
process_and_save(val_images, 'val')

print("Dataset YOLO creato con split train/val e bounding box dinamico!")
