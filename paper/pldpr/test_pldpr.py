# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 10:14:02 2025

@author: fedes
"""

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from torchvision import transforms

from train_file import PDLPR, LicensePlateDataset, CHARS, CHAR2IDX  # <-- Assicurati che questi nomi siano corretti

# --- Costanti e mapping ---
index_to_char = CHARS
char_to_index = CHAR2IDX
NUM_CLASSES = len(index_to_char)
NUM_CHARS = 7  # lunghezza della targa

# --- Evaluation function ---
def evaluate_model(model, dataset, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_labels, all_preds = [], []
    model.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluation"):
            imgs = imgs.to(device)
            outputs = model(imgs)                    # [B, 7, 68]
            preds = outputs.argmax(dim=2).cpu()      # [B, 7]
            all_labels.append(labels)
            all_preds.append(preds)

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    n_samples = all_labels.shape[0]

    exact = (all_preds == all_labels).all(dim=1).sum().item()
    total_accuracy = exact / n_samples
    print(f"Accuratezza targa perfetta: {total_accuracy*100:.2f}% ({exact} su {n_samples})")

    accuracy_per_pos = []
    for pos in range(NUM_CHARS):
        n_correct = (all_preds[:, pos] == all_labels[:, pos]).sum().item()
        accuracy_per_pos.append(n_correct / n_samples)
    mean_char_acc = sum(accuracy_per_pos) / NUM_CHARS
    print(f"Accuratezza media per carattere: {mean_char_acc*100:.2f}%")

    error_matrix = (all_preds != all_labels)
    chars_wrong_per_sample = error_matrix.sum(dim=1).tolist()
    mean_wrong_chars = sum(chars_wrong_per_sample) / n_samples
    print(f"Numero medio di caratteri sbagliati a targa: {mean_wrong_chars:.2f}")

    chars_wrong_per_sample_2to7 = error_matrix[:, 1:].sum(dim=1).tolist()
    error_hist = Counter(chars_wrong_per_sample_2to7)
    error_hist_full = [error_hist.get(i, 0) for i in range(7)]

    chinese_class_set = set(range(0, 31))
    confusion_chinese = Counter()
    confusion_normal = Counter()
    for i in range(all_preds.shape[0]):
        for pos in range(NUM_CHARS):
            if all_preds[i, pos] != all_labels[i, pos]:
                t_idx = all_labels[i, pos].item()
                p_idx = all_preds[i, pos].item()
                if t_idx in chinese_class_set:
                    confusion_chinese[(index_to_char[t_idx], index_to_char[p_idx])] += 1
                else:
                    confusion_normal[(index_to_char[t_idx], index_to_char[p_idx])] += 1
    top10_chinese = confusion_chinese.most_common(10)
    top10_normal = confusion_normal.most_common(10)

    # --- GRAFICI ---
    plt.figure(figsize=(8,6))
    plt.bar(range(7), error_hist_full, color='red')
    plt.xlabel('Numero caratteri sbagliati (posizioni 2-7)')
    plt.ylabel('Frequenza')
    plt.title('Errori per numero caratteri sbagliati in pos 2-7')
    plt.xticks(range(7))
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'istogramma_errori_targa.png'))
    plt.close()

    if top10_chinese:
        labels = [f"{x[0]}→{x[1]}" for x, _ in top10_chinese]
        values = [v for _, v in top10_chinese]
        plt.figure(figsize=(10, 6))
        plt.barh(labels[::-1], values[::-1], color='blue')
        plt.xlabel('Numero errori')
        plt.title('Top 10 confusioni caratteri cinesi')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'istogramma_cinesi_piu_confusi.png'))
        plt.close()

    if top10_normal:
        labels = [f"{x[0]}→{x[1]}" for x, _ in top10_normal]
        values = [v for _, v in top10_normal]
        plt.figure(figsize=(10, 6))
        plt.barh(labels[::-1], values[::-1], color='green')
        plt.xlabel('Numero errori')
        plt.title('Top 10 confusioni lettere e numeri')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'istogramma_normal_piu_confusi.png'))
        plt.close()

    plt.figure(figsize=(8,6))
    plt.bar(range(NUM_CHARS), accuracy_per_pos, color='purple')
    plt.xlabel('Posizione carattere (0=Cinese)')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza per posizione carattere')
    plt.xticks(range(NUM_CHARS))
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'accuracy_per_posizione.png'))
    plt.close()

    # --- Report TXT ---
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('--- RISULTATI EVALUATION OCR TARGA ---\n')
        f.write(f'Campioni testati: {n_samples}\n')
        f.write(f'Accuratezza targa perfetta: {total_accuracy*100:.2f}%  ({exact} su {n_samples})\n')
        f.write(f'Accuratezza media per carattere: {mean_char_acc*100:.2f}%\n')
        f.write(f'Numero medio di caratteri sbagliati per targa: {mean_wrong_chars:.2f}\n')
        f.write('Distribuzione errori caratteri sbagliati per targa (pos 2-7):\n')
        for i, v in enumerate(error_hist_full):
            perc = v / n_samples * 100
            f.write(f'  {i} caratteri sbagliati: {v} ({perc:.2f}%)\n')
        f.write('\nTop 10 confusioni caratteri cinesi:\n')
        for (c, v) in top10_chinese:
            f.write(f'  {c[0]} → {c[1]}: {v}\n')
        f.write('\nTop 10 confusioni lettere e numeri:\n')
        for (c, v) in top10_normal:
            f.write(f'  {c[0]} → {c[1]}: {v}\n')
        f.write('\nAccuratezza per posizione carattere:\n')
        for pos, acc in enumerate(accuracy_per_pos):
            f.write(f'  Posizione {pos}: {acc:.3f}\n')

    print(f"\n--------------------------")
    print(f"Campioni valutati: {n_samples}")
    print(f"Accuratezza targa perfetta: {total_accuracy*100:.2f}%  ({exact} su {n_samples})")
    print(f"Accuratezza media per carattere: {mean_char_acc*100:.2f}%")
    print(f"Numero medio di caratteri sbagliati per targa: {mean_wrong_chars:.2f}")
    print(f"Risultati e grafici salvati in: {save_dir}")
    print(f"--------------------------\n")

# --- MAIN ---
def main():
    test_images = "F:/progetto computer vision/dataxricChar/evaluation/ccpd_blur/test/images"
    test_labels = "F:/progetto computer vision/dataxricChar/evaluation/ccpd_blur/test/labels.txt"
    model_weights = "best_pdlpr.pth"
    save_dir = "risultati_evaluation_pdlpr/ccpd_blur"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = LicensePlateDataset(
        img_dir=test_images,
        label_file=test_labels,
        img_size=(96, 288),
        transform=transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PDLPR(
        in_channels=3,
        d_model=512,
        n_heads=8,
        num_units=3,
        seq_len=7,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(model_weights, map_location=device))
    evaluate_model(model, test_dataset, device, save_dir)

if __name__ == '__main__':
    main()
