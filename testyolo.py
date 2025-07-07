# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 06:18:54 2025
@author: fedes
"""

import os
import subprocess
import yaml
import cv2
import time
import shutil
import sys
import pathlib

# === CONFIGURA ===
base_dir = 'F:\\progetto computer vision\\dataset\\CCPD2019'
model_weights = './best.pt'
img_size = 640
conf_threshold = 0.25
output_dir = 'test_results'
class_names = ['plate']

os.makedirs(output_dir, exist_ok=True)

def parse_filename_bbox(filename):
    try:
        parts = filename.split('-')
        box_part = parts[2]
        coords = box_part.split('_')
        x1, y1 = map(int, coords[0].split('&'))
        x2, y2 = map(int, coords[1].split('&'))
        return x1, y1, x2, y2
    except Exception as e:
        print(f" Errore parsing {filename}: {e}")
        return None

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h
    return x_center, y_center, width, height

ccpd_dirs = [d for d in os.listdir(base_dir) if d.startswith('ccpd_') and d not in ['ccpd_base', 'ccpd_np', 'ccpd_blur']]

for ccpd_name in ccpd_dirs:
    print(f'\n▶ Preparazione e test su: {ccpd_name}')

    ccpd_path = os.path.join(base_dir, ccpd_name)
    image_files = [f for f in os.listdir(ccpd_path) if f.endswith('.jpg')]

    if not image_files:
        print(f" Nessuna immagine trovata in {ccpd_name}")
        continue

    # === Crea strutture compatibili YOLOv5 ===
    images_val_dir = os.path.join(ccpd_path, 'images', 'val')
    labels_val_dir = os.path.join(ccpd_path, 'labels', 'val')
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(ccpd_path, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_val_dir, label_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f" Immagine non valida: {img_path}")
            continue

        h, w = img.shape[:2]
        bbox = parse_filename_bbox(img_name)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x_c, y_c, bw, bh = convert_to_yolo(x1, y1, x2, y2, w, h)

        with open(label_path, 'w') as f:
            f.write(f'0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n')

        dst_img_path = os.path.join(images_val_dir, img_name)
        shutil.copy(img_path, dst_img_path)

    # === YAML ===
    yaml_path = os.path.join(output_dir, f'{ccpd_name}.yaml')
    yaml_data = {
        'path': ccpd_path,
        'val': 'images/val',
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    # === YOLOv5 test ===
    print(f"🔍 Avvio test YOLOv5 su {ccpd_name}")
    command = [
        'python', 'val.py',
        '--weights', model_weights,
        '--data', yaml_path,
        '--task', 'test',
        '--conf', str(conf_threshold),
        '--img', str(img_size),
        '--project', output_dir,
        '--name', ccpd_name,
        '--exist-ok'
    ]

    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    print("[YOLOv5 STDOUT]")
    print(result.stdout)
    print("[YOLOv5 STDERR]")
    print(result.stderr)

    elapsed_time = end_time - start_time
    fps = len(image_files) / elapsed_time if elapsed_time > 0 else 0

    results_txt = os.path.join(output_dir, ccpd_name, 'results.txt')
    map_50 = 'N/A'
    if os.path.exists(results_txt):
        with open(results_txt, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split()
                map_50 = last_line[5] if len(last_line) >= 6 else 'N/A'

    print(f"\n✅ RISULTATI: {ccpd_name}")
    print(f" - Immagini processate: {len(image_files)}")
    print(f" - Tempo totale: {elapsed_time:.2f} sec")
    print(f" - FPS: {fps:.2f}")
    print(f" - mAP@0.5: {map_50}")

