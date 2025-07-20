# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 08:53:19 2025
@author: fedes
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pldpr import PLDPR  # Assicurati che il tuo file PLDPR.py sia corretto

# --- Configurazioni globali ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
             "警", "学", "O"]
alphabets = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
ads = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]

base_chars = sorted(set(provinces + alphabets + ads))
base_chars.append("<BLANK>")

CHAR2IDX = {ch: i for i, ch in enumerate(base_chars)}
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}
vocab_size = len(base_chars)
BLANK_IDX = CHAR2IDX["<BLANK>"]

SEQ_LEN = 7
BATCH_SIZE = 128
VAL_BATCH = 5
EPOCHS = 20
INIT_LR = 1e-3
LR_MIN = 1e-5
IMG_SZ = (48, 144)
NUM_CLASSES = vocab_size
DATA_ROOT = 'C:/Users/fedes/Desktop/ccpd_dataset'
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels.txt")
VAL_LABELS = os.path.join(VAL_DIR, "labels.txt")


# --- Dataset e trasformazioni ---
class LicensePlateDataset(data.Dataset):
    def __init__(self, dir, labf, img_size, transform=None):
        self.samples = [(l.split('\t')[0], l.split('\t')[1].strip()) for l in open(labf, encoding='utf-8')]
        self.dir = dir  # es: 'ccpd_dataset/train'
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        n, lb = self.samples[i]
        img_path = os.path.join(self.dir, 'images', n)
        img = Image.open(img_path).convert('RGB').resize(self.img_size[::-1])
        if self.transform:
            img = self.transform(img)
        lab = torch.LongTensor([CHAR2IDX[c] for c in lb])
        return img, lab, len(lab)


def get_transforms(is_train):
    aug = []
    if is_train:
        aug = [transforms.ColorJitter(.3, .3, .2, .02), transforms.RandomAffine(4, (.04, .12), (.92, 1.08))]
    aug += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])]
    return transforms.Compose(aug)

def ctc_collate_fn(batch):
    images, labels, _ = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.cat([label for label in labels])
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    input_lengths = torch.full(size=(len(labels),), fill_value=18, dtype=torch.long)
    return images, targets, input_lengths, target_lengths

# --- Decodifica CTC greedy ---
def ctc_greedy_decoder(logits, blank_idx=BLANK_IDX):
    probs = logits.softmax(2)
    preds = probs.argmax(2).permute(1, 0)
    decoded = []
    for pred in preds:
        out = []
        prev = -1
        for p in pred:
            p = p.item()
            if p != blank_idx and p != prev:
                out.append(IDX2CHAR[p])
            prev = p
        decoded.append("".join(out))
    return decoded

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # shape [B, C, T]
        outputs = outputs.permute(1, 0, 2)  # ora [T, B, C]
        print(outputs.shape)

        log_probs = outputs.log_softmax(2)
        
        batch_size = images.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long).to(device)
        

        # Debug stampe dimensioni
        print(f"Batch idx: {batch_idx}")
        print(f"images.shape: {images.shape}")           # Expected: [B, 3, H, W]
        print(f"outputs.shape: {outputs.shape}")         # Expected: [T, B, C]
        print(f"labels.shape: {labels.shape}")           # Expected: [B, max_label_len]
        print(f"label_lengths.shape: {label_lengths.shape}") # Expected: [B]
        print(f"input_lengths.shape: {input_lengths.shape}") # Expected: [B]        

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
    return avg_loss

def evaluate_model(model, dataloader, device, verbose=True, print_errors=False):
    model.eval()
    total = 0
    correct = 0
    total_chars = 0
    correct_chars = 0
    length_errors = 0
    province_correct = 0
    alphabet_correct = 0

    incorrect_samples = []

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            pred_strings = ctc_greedy_decoder(outputs)

            labels_cpu = labels.cpu().numpy()
            lengths_cpu = label_lengths.cpu().numpy()

            idx = 0
            gt_strings = []
            for i, l in enumerate(lengths_cpu):
                label_seq = labels_cpu[i][:l]
                # Forza conversione a lista di interi puri
                label_seq = label_seq.tolist()  # garantisce che siano numeri base Python
                gt = ''.join([IDX2CHAR[c] for c in label_seq])
                gt_strings.append(gt)


            for pred, gt in zip(pred_strings, gt_strings):
                total += 1
                if pred == gt:
                    correct += 1
                else:
                    if print_errors:
                        incorrect_samples.append((pred, gt))

                min_len = min(len(pred), len(gt))
                correct_chars += sum(p == g for p, g in zip(pred[:min_len], gt[:min_len]))
                total_chars += len(gt)

                if len(pred) != len(gt):
                    length_errors += 1

                if len(pred) > 0 and len(gt) > 0 and pred[0] == gt[0]:
                    province_correct += 1

                if len(pred) > 1 and len(gt) > 1 and pred[1] == gt[1]:
                    alphabet_correct += 1

    acc = correct / total if total > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    length_error_rate = length_errors / total if total > 0 else 0
    province_acc = province_correct / total if total > 0 else 0
    alphabet_acc = alphabet_correct / total if total > 0 else 0

    if verbose:
        print(f"Evaluation results:")
        print(f"  Exact plate accuracy: {acc:.4f}")
        print(f"  Character accuracy: {char_acc:.4f}")
        print(f"  Length error rate: {length_error_rate:.4f}")
        print(f"  Province character accuracy: {province_acc:.4f}")
        print(f"  Alphabet character accuracy: {alphabet_acc:.4f}")
        print(f"  Total samples: {total}")

        if print_errors and incorrect_samples:
            print("\nIncorrect predictions samples:")
            for i, (pred, gt) in enumerate(incorrect_samples[:10]):
                print(f"  Sample {i+1}: Predicted: '{pred}' | Ground Truth: '{gt}'")

    return {
        'plate_acc': acc,
        'char_acc': char_acc,
        'length_error_rate': length_error_rate,
        'province_acc': province_acc,
        'alphabet_acc': alphabet_acc,
        'total_samples': total,
        'incorrect_samples': incorrect_samples
    }

def plot_training_curves(train_losses, val_accuracies):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# --- Main ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    train_ds = LicensePlateDataset(TRAIN_DIR, TRAIN_LABELS, IMG_SZ, get_transforms(is_train=True))
    val_ds = LicensePlateDataset(VAL_DIR, VAL_LABELS, IMG_SZ, get_transforms(is_train=False))


    model = PLDPR(num_classes=NUM_CLASSES, dropout=0.1).to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate_fn, drop_last=True)
        val_loader = data.DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, collate_fn=ctc_collate_fn)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        metrics = evaluate_model(model, val_loader, device, verbose=True, print_errors=True)

        scheduler.step()
        train_losses.append(train_loss)
        val_accuracies.append(metrics['plate_acc'])

    plot_training_curves(train_losses, val_accuracies)
