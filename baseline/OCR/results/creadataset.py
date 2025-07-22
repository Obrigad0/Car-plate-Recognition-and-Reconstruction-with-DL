# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:08:08 2025

@author: fedes
"""

import os
from PIL import Image

# Liste caratteri targa CCPD
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
       'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def decode_plate_from_name(name):
    parts = name.split('-')
    if len(parts) < 6:
        print(f"Nome file non conforme: {name}")
        return None
    code_str = parts[-3]  # es. '0_0_22_27_27_33_16'
    indices = code_str.split('_')
    if len(indices) != 7:
        print(f"Codice targa non ha 7 caratteri: {code_str}")
        return None
    try:
        indices_int = [int(i) for i in indices]
    except Exception as e:
        print(f"Errore decodifica targhe: {name} -> {e}")
        return None
    try:
        c1 = provinces[indices_int[0]]
        c2 = alphabets[indices_int[1]]
        c3_to_c7 = [ads[i] for i in indices_int[2:]]
    except IndexError as e:
        print(f"Indice fuori range per decodifica targa in {name}: {e}")
        return None
    plate = c1 + c2 + ''.join(c3_to_c7)
    plate = plate.replace('O', '')  # 'O' = nessun carattere
    return plate

def parse_bbox_from_name(name):
    # Nel nome es. "90_89-441&517_538&546-530&552_447&548_447&512_530&516"
    # la bbox è nel secondo gruppo (indice 1) che contiene coordinate come '441&517_538&546'
    # Qui prendo la bounding box (rectangle) per il ritaglio immagine
    
    parts = name.split('-')
    if len(parts) < 3:
        print(f"Nome file non conforme per bbox: {name}")
        return None
    bbox_str = parts[2]
    try:
        # bbox_str è tipo '441&517_538&546'
        coords = bbox_str.split('_')
        x1, y1 = map(int, coords[0].split('&'))
        x2, y2 = map(int, coords[1].split('&'))
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"Errore parsing bbox in {name}: {e}")
        return None

def process_ccpd_dataset(root_dir, output_dir):
    # root_dir: path a 'ccpd' con le sottocartelle
    # output_dir: dove salvare le immagini ritagliate e labels.txt

    exclude = {'ccpd_base', 'ccpd_np'}
    os.makedirs(output_dir, exist_ok=True)

    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as label_file:

        for subfolder in os.listdir(root_dir):
            if subfolder in exclude:
                print(f"Skip {subfolder}")
                continue
            subfolder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            print(f"Processing folder: {subfolder}")

            out_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(out_subfolder, exist_ok=True)

            for filename in os.listdir(subfolder_path):
                if not (filename.endswith('.jpg') or filename.endswith('.png')):
                    continue

                full_path = os.path.join(subfolder_path, filename)
                plate = decode_plate_from_name(filename)
                bbox = parse_bbox_from_name(filename)
                if plate is None or bbox is None:
                    print(f"Skip file {filename} for decoding issues")
                    continue

                try:
                    img = Image.open(full_path)
                    cropped = img.crop(bbox)
                    save_path = os.path.join(out_subfolder, filename)
                    cropped.save(save_path)

                    # scrivi label: path relativo + tab + plate
                    rel_path = os.path.join(subfolder, filename).replace('\\','/')
                    label_file.write(f"{rel_path}\t{plate}\n")

                except Exception as e:
                    print(f"Errore elaborazione {filename}: {e}")

    print("Processamento completato!")

if __name__ == '__main__':
    # Esempio di uso:
    ccpd_root = 'C:/Users/fedes/Downloads/CCPD2019/CCPD2019'        
    output_folder = 'C:/Users/fedes/Desktop/datasetxclaudio'    # cambia con path reale

    process_ccpd_dataset(ccpd_root, output_folder)
