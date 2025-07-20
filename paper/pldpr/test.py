# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 10:14:02 2025
@author: fedes
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from pdlpr import PDLPR  

# --- COSTANTI ---
BLANK_TOKEN = '-'
SEQ_LEN = 7  # Lunghezza standard della targa CCPD

CHARS = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣',
    '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', 'O',  # Province (34)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O',  # Letters (24 + O)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', BLANK_TOKEN  # Digits + blank
]

CHAR2IDX = {ch: i for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
BLANK_IDX = CHAR2IDX[BLANK_TOKEN]


# --- DATASET CLASSI ---
class CCPDCharCropDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        with open(labels_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, label = parts[0], parts[1]
                    self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_idx = [CHAR2IDX.get(c, BLANK_IDX) for c in label]
        if len(label_idx) < SEQ_LEN:
            label_idx += [BLANK_IDX] * (SEQ_LEN - len(label_idx))
        else:
            label_idx = label_idx[:SEQ_LEN]
        return img, torch.tensor(label_idx, dtype=torch.long)


class LPRDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, max_len=SEQ_LEN):
        self.img_dir = img_dir
        self.transform = transform
        self.max_len = max_len
        self.samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = ''.join(parts[1:])
                    self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = [CHAR2IDX.get(c, BLANK_IDX) for c in label]
        if len(label_idx) < self.max_len:
            label_idx += [BLANK_IDX] * (self.max_len - len(label_idx))
        else:
            label_idx = label_idx[:self.max_len]
        return image, torch.tensor(label_idx, dtype=torch.long)


def load_and_concat_datasets(base_dir, transform):
    train_images = os.path.join(base_dir, "train", "images")
    train_labels = os.path.join(base_dir, "train", "labels.txt")
    val_images = os.path.join(base_dir, "val", "images")
    val_labels = os.path.join(base_dir, "val", "labels.txt")
    train_dataset = CCPDCharCropDataset(train_images, train_labels, transform)
    val_dataset = CCPDCharCropDataset(val_images, val_labels, transform)
    return ConcatDataset([train_dataset, val_dataset])


# --- FUNZIONE DI VALUTAZIONE ---
def evaluate_model(model, dataset, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluation"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=2).cpu()
            all_labels.append(labels)
            all_preds.append(preds)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    n_samples = all_labels.size(0)
    exact = (all_preds == all_labels).all(dim=1).sum().item()
    total_accuracy = exact / n_samples
    print(f"Accuratezza targa perfetta: {total_accuracy*100:.2f}% ({exact}/{n_samples})")
    accuracy_per_pos = [(all_preds[:, i] == all_labels[:, i]).sum().item() / n_samples for i in range(SEQ_LEN)]
    mean_char_acc = sum(accuracy_per_pos) / SEQ_LEN
    print(f"Accuratezza media per carattere: {mean_char_acc*100:.2f}%")
    error_matrix = (all_preds != all_labels)
    mean_wrong_chars = error_matrix.sum(dim=1).float().mean().item()
    print(f"Numero medio di caratteri sbagliati per targa: {mean_wrong_chars:.2f}")
    chars_wrong_2to7 = error_matrix[:, 1:].sum(dim=1).tolist()
    error_hist = Counter(chars_wrong_2to7)
    error_hist_full = [error_hist.get(i, 0) for i in range(SEQ_LEN + 1)]
    chinese_class_set = set(range(0, 31))
    confusion_chinese, confusion_normal = Counter(), Counter()
    for i in range(n_samples):
        for j in range(SEQ_LEN):
            if all_preds[i, j] != all_labels[i, j]:
                t_idx = all_labels[i, j].item()
                p_idx = all_preds[i, j].item()
                if t_idx in chinese_class_set:
                    confusion_chinese[(CHARS[t_idx], CHARS[p_idx])] += 1
                else:
                    confusion_normal[(CHARS[t_idx], CHARS[p_idx])] += 1
    top10_chinese = confusion_chinese.most_common(10)
    top10_normal = confusion_normal.most_common(10)

    # Grafici
    import matplotlib.font_manager as fm

    font_path = 'C:/Windows/Fonts/simsun.ttc'
    simsun_font = fm.FontProperties(fname=font_path)

    def save_bar_plot(data, labels, title, filename, horizontal=False, color='blue'):
        plt.figure(figsize=(10, 6))
        if horizontal:
            plt.barh(labels[::-1], data[::-1], color=color)
            plt.xlabel('Numero errori', fontproperties=simsun_font)
        else:
            plt.bar(labels, data, color=color)
            plt.xlabel('Categoria', fontproperties=simsun_font)
        plt.title(title, fontproperties=simsun_font)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()


    save_bar_plot(error_hist_full, list(range(SEQ_LEN + 1)), 'Errori per caratteri sbagliati (pos 2-7)', 'istogramma_errori_targa.png')
    if top10_chinese:
        labels, values = zip(*[(f"{k[0]}→{k[1]}", v) for k, v in top10_chinese])
        save_bar_plot(values, list(labels), 'Top 10 confusioni caratteri cinesi', 'istogramma_cinesi_piu_confusi.png', horizontal=True, color='blue')
    if top10_normal:
        labels, values = zip(*[(f"{k[0]}→{k[1]}", v) for k, v in top10_normal])
        save_bar_plot(values, list(labels), 'Top 10 confusioni lettere/numeri', 'istogramma_normal_piu_confusi.png', horizontal=True, color='green')
    save_bar_plot(accuracy_per_pos, list(range(SEQ_LEN)), 'Accuratezza per posizione carattere', 'accuracy_per_posizione.png', color='purple')

    # Report TXT
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('--- RISULTATI EVALUATION OCR TARGA ---\n')
        f.write(f'Campioni testati: {n_samples}\n')
        f.write(f'Accuratezza targa perfetta: {total_accuracy*100:.2f}% ({exact}/{n_samples})\n')
        f.write(f'Accuratezza media per carattere: {mean_char_acc*100:.2f}%\n')
        f.write(f'Numero medio caratteri sbagliati: {mean_wrong_chars:.2f}\n')
        f.write('\nDistribuzione errori caratteri sbagliati per targa (pos 2-7):\n')
        for i, v in enumerate(error_hist_full):
            f.write(f'  {i} errori: {v} ({v / n_samples * 100:.2f}%)\n')
        f.write('\nTop 10 confusioni caratteri cinesi:\n')
        for (c, v) in top10_chinese:
            f.write(f'  {c[0]} → {c[1]}: {v}\n')
        f.write('\nTop 10 confusioni lettere/numeri:\n')
        for (c, v) in top10_normal:
            f.write(f'  {c[0]} → {c[1]}: {v}\n')
        f.write('\nAccuratezza per posizione:\n')
        for i, acc in enumerate(accuracy_per_pos):
            f.write(f'  Pos {i}: {acc*100:.2f}%\n')

    print("\nValutazione completata.")
    print(f"Risultati salvati in: {save_dir}\n")


# --- MAIN ---
def main():
    base_dir = "C:/Users/fedes/Desktop/datibellissimi/ccpd_weather"
    model_weights = "./models/pdlpr_final.pth"
    save_dir = "./results/ccpd_weather"
    transform = transforms.Compose([transforms.ToTensor()])

    # ✅ Usa train + val
    full_dataset = load_and_concat_datasets(base_dir, transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PDLPR(
    in_channels=3,
    base_channels=256,          # ← come nel checkpoint
    encoder_d_model=256,        # ← come nel checkpoint
    encoder_nhead=8,
    encoder_height=16,
    encoder_width=16,
    decoder_num_layers=2,
    num_classes=69,             # ← come nel checkpoint
    seq_len=7                   # ← come nel checkpoint
).to(device)

    model.load_state_dict(torch.load(model_weights, map_location=device))
    evaluate_model(model, full_dataset, device, save_dir)



if __name__ == '__main__':
    main()
