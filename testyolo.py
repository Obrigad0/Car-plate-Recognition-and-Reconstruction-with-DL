# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 06:18:54 2025

@author: fedes
"""

import os
import subprocess
import yaml
import cv2

# === CONFIGURA ===
base_dir = '../Downloads/CCPD2019/CCPD2019'   
model_weights = '/percorso/al/modello/best.pt'
img_size = 640
conf_threshold = 0.25 
output_dir = 'test_results'

# Classe YOLO
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

# Scansiona tutte le cartelle CCPD eccetto CCPD_base
ccpd_dirs = [d for d in os.listdir(base_dir) if d.startswith('ccpd_') and d != 'ccpd_base']

for ccpd_name in ccpd_dirs:
    print(f'\nPreparazione e test su: {ccpd_name}')

    ccpd_path = os.path.join(base_dir, ccpd_name)
    labels_dir = os.path.join(ccpd_path, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(ccpd_path) if f.endswith('.jpg')]

    if not image_files:
        print(f" Nessuna immagine trovata in {ccpd_name}")
        continue

    # === Estrai label dai nomi dei file ===
    for img_name in image_files:
        img_path = os.path.join(ccpd_path, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        if os.path.exists(label_path):
            continue  # skip se già creato

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

    # === YAML ===
    yaml_path = os.path.join(output_dir, f'{ccpd_name}.yaml')
    yaml_data = {
        'path': ccpd_path,
        'train': '.',  # dummy
        'val': '.',
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    # === YOLOv5 test ===
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

    subprocess.run(command)

    print("\nTutti i test completati.")
