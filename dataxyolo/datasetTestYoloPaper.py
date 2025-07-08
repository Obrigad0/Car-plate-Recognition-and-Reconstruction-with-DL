import os
import shutil
from PIL import Image

# Configurazioni
dataset_path = 'F:\\progetto computer vision\\dataxyolo'
test_types_root = 'F:\\progetto computer vision\\dataset\\CCPD2019\\'  # dove hai le sottocartelle di test

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

def create_test_folders(selected_test_type):
    # Percorsi di destinazione
    test_img_dst = os.path.join(dataset_path, 'images', 'test')
    test_lbl_dst = os.path.join(dataset_path, 'labels', 'test')
    # Sorgente immagini di test
    test_src = os.path.join(test_types_root, selected_test_type)

    # Pulisci cartelle test
    for folder in [test_img_dst, test_lbl_dst]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # Elabora immagini di test
    for filename in os.listdir(test_src):
        if filename.endswith(('.jpg', '.png')):
            bbox = extract_bbox_from_filename(filename)
            if bbox is None:
                continue
            img_path = os.path.join(test_src, filename)
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            x_center, y_center, w, h = convert_bbox_to_yolo(*bbox, img_w, img_h)
            # Copia immagine
            shutil.copy(img_path, os.path.join(test_img_dst, filename))
            # Crea label
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(test_lbl_dst, label_filename)
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    print(f"Test set '{selected_test_type}' pronto!")

# Esempio di uso: scegli la sottocartella di test che vuoi usare
create_test_folders('ccpd_blur')  # oppure 'type2', 'type3', ecc.
