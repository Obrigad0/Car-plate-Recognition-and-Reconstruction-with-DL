import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from model import UnifiedResNetModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

index_to_char = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
    "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]
idx2char = {i: c for i, c in enumerate(index_to_char)}
NUM_CLASSES = len(index_to_char)
NUM_CHARS = 7

class CCPDTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        labels_file = os.path.join(root_dir, 'labels.txt')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"labels.txt non trovato in {root_dir}")

        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f]

        for img_rel_path, plate in lines:
            full_path = os.path.join(root_dir, img_rel_path)
            if os.path.exists(full_path):
                subfolder = img_rel_path.split('/')[0] if '/' in img_rel_path else img_rel_path.split('\\')[0]
                self.samples.append((full_path, plate, subfolder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, plate, subfolder = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, plate, subfolder, img_path

def decode_output(logits_list):
    pred_plate = ''.join(idx2char[logits.argmax(dim=-1).item()] for logits in logits_list)
    return pred_plate.rstrip('O')  # Rimuove padding 'O' in fondo

def char_accuracy(true_str, pred_str):
    correct = sum(t == p for t, p in zip(true_str, pred_str))
    return correct / max(len(true_str), len(pred_str))

def evaluate_model(model, dataloader):
    accuracies = {}
    model.eval()
    with torch.no_grad():
        for imgs, plates, subfolders, _ in tqdm(dataloader, desc="Valutazione modello"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            batch_size = imgs.size(0)
            for i in range(batch_size):
                true_plate = plates[i]
                pred_logits = [out[i].unsqueeze(0) for out in outputs]
                pred_plate = decode_output(pred_logits)
                acc = char_accuracy(true_plate, pred_plate)
                sf = subfolders[i]
                accuracies.setdefault(sf, []).append(acc)
    return accuracies

def char_error_distribution(model, dataset):
    error_counts = Counter()
    loader = DataLoader(dataset, batch_size=16)
    model.eval()
    with torch.no_grad():
        for imgs, plates, _, _ in tqdm(loader, desc="Distribuzione errori"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            for i in range(imgs.size(0)):
                true_plate = plates[i]
                pred_logits = [out[i].unsqueeze(0) for out in outputs]
                pred_plate = decode_output(pred_logits)
                errors = sum(t != p for t, p in zip(true_plate, pred_plate))
                error_counts[errors] += 1
    return error_counts

def chinese_char_accuracy(dataset, model):
    correct = 0
    total = 0
    mistakes = Counter()
    loader = DataLoader(dataset, batch_size=16)
    model.eval()
    with torch.no_grad():
        for imgs, plates, _, _ in tqdm(loader, desc="Valutazione primo carattere"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            for i in range(imgs.size(0)):
                true_plate = plates[i]
                pred_logits = [out[i].unsqueeze(0) for out in outputs]
                pred_plate = decode_output(pred_logits)
                total += 1
                if pred_plate and pred_plate[0] == true_plate[0]:
                    correct += 1
                else:
                    mistakes[true_plate[0]] += 1
    return correct / total, mistakes

def plot_and_save_results(accuracies, save_dir, window=10):
    os.makedirs(save_dir, exist_ok=True)
    for sf, acc_list in accuracies.items():
        acc_array = np.array(acc_list)
        if len(acc_array) >= window:
            smooth_acc = np.convolve(acc_array, np.ones(window)/window, mode='valid')
        else:
            smooth_acc = acc_array

        plt.figure(figsize=(10, 4))
        plt.plot(smooth_acc, label=f'{sf} (media mobile)', color='tab:blue')
        plt.title(f'OCR Accuracy Smoothed – {sf}')
        plt.xlabel('Indice immagine')
        plt.ylabel('Accuratezza caratteri')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sf}_accuracy_smoothed.png'))
        plt.close()

def plot_accuracy_boxplot(accuracies, save_path):
    labels = list(accuracies.keys())
    data = [accuracies[label] for label in labels]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title('Distribuzione accuracy per cartella')
    plt.ylabel('Accuratezza per immagine')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(error_counts, save_path):
    keys = sorted(error_counts.keys())
    values = [error_counts[k] for k in keys]

    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, color='crimson')
    plt.xlabel("Numero di caratteri sbagliati")
    plt.ylabel("Numero di immagini")
    plt.title("Distribuzione errori OCR per immagine")
    plt.xticks(range(max(keys)+1))
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_chinese_char_mistakes(mistakes, save_path):
    chars = list(mistakes.keys())
    counts = [mistakes[c] for c in chars]

    plt.figure(figsize=(10, 5))
    plt.bar(chars, counts, color='darkorange')
    plt.title("Errori sul primo carattere (cinese)")
    plt.xlabel("Carattere")
    plt.ylabel("Numero di errori")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_accuracies_to_csv(accuracies, csv_path):
    all_data = [(sf, acc) for sf, acc_list in accuracies.items() for acc in acc_list]
    df = pd.DataFrame(all_data, columns=['folder', 'accuracy'])
    df.to_csv(csv_path, index=False)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data_root = 'C:/Users/fedes/Desktop/datasetxclaudio'
    results_dir = './claudioresults'
    os.makedirs(results_dir, exist_ok=True)

    model = UnifiedResNetModel(head_type='ocr', pretrained=False,
                               num_chars=NUM_CHARS, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("claudio_best_model.pth", map_location=device))
    print("✅ Modello caricato con successo")

    dataset = CCPDTestDataset(data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    accuracies = evaluate_model(model, dataloader)
    plot_and_save_results(accuracies, results_dir, window=10)
    plot_accuracy_boxplot(accuracies, os.path.join(results_dir, 'accuracy_boxplot.png'))
    save_accuracies_to_csv(accuracies, os.path.join(results_dir, 'accuracies_by_image.csv'))

    error_counts = char_error_distribution(model, dataset)
    plot_error_distribution(error_counts, os.path.join(results_dir, 'error_distribution.png'))

    acc_chinese, chinese_mistakes = chinese_char_accuracy(dataset, model)
    plot_chinese_char_mistakes(chinese_mistakes, os.path.join(results_dir, 'chinese_char_errors.png'))

    print(f"Valutazione OCR completata.")
    print(f"🎯 Accuratezza primo carattere (cinese): {acc_chinese:.2%}")
    print(f"📁 Risultati salvati in: {os.path.abspath(results_dir)}")

if __name__ == "__main__":
    main()
