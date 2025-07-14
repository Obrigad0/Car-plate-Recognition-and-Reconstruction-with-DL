# train_pdlpr.py

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pdlpr import PDLPR

# -------------------------------
# 1. Dataset personalizzato CCPD
# -------------------------------

class CCPDDataset(Dataset):
    def __init__(self, images_dir, labels_path, char2idx, seq_len=18, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.seq_len = seq_len
        self.char2idx = char2idx

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.samples = [line.strip().split('\t') for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def encode_label(self, label):
        label_idx = [self.char2idx.get(c, self.char2idx['[UNK]']) for c in label]
        if len(label_idx) < self.seq_len:
            label_idx += [self.char2idx['[PAD]']] * (self.seq_len - len(label_idx))
        else:
            label_idx = label_idx[:self.seq_len]
        return torch.tensor(label_idx, dtype=torch.long)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_encoded = self.encode_label(label)
        return image, label_encoded, img_name

# -------------------------------
# 2. Dizionario caratteri
# -------------------------------

ALL_CHARS = (
    ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
     "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"] +
    ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','O'] +
    ['0','1','2','3','4','5','6','7','8','9','O']
)
ALL_CHARS = list(dict.fromkeys(ALL_CHARS))
ALL_CHARS = ['[PAD]', '[UNK]'] + ALL_CHARS

char2idx = {c: i for i, c in enumerate(ALL_CHARS)}
idx2char = {i: c for c, i in char2idx.items()}

# -------------------------------
# 3. Trasformazioni immagini
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((96, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------------
# 4. Hyperparametri (dal paper)
# -------------------------------

BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
SEQ_LEN = 18
NUM_CLASSES = len(char2idx)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# 5. Caricamento dati
# -------------------------------

train_img_dir = 'F:/progetto computer vision/dataxricChar/train/images'
train_label_path = 'F:/progetto computer vision/dataxricChar/train/labels.txt'
val_img_dir = 'F:/progetto computer vision/dataxricChar/val/images'
val_label_path = 'F:/progetto computer vision/dataxricChar/val/labels.txt'

train_dataset = CCPDDataset(train_img_dir, train_label_path, char2idx, seq_len=SEQ_LEN, transform=transform)
val_dataset = CCPDDataset(val_img_dir, val_label_path, char2idx, seq_len=SEQ_LEN, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------------------
# 6. Modello, loss, ottimizzatore
# -------------------------------

model = PDLPR(in_channels=3, d_model=512, n_heads=8, num_units=3, seq_len=SEQ_LEN, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=char2idx['[PAD]'])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 7. Training loop con decodifica predizioni
# -------------------------------

def decode_sequence(seq_tensor, idx2char):
    # seq_tensor: [seq_len] o [B, seq_len]
    if seq_tensor.dim() == 1:
        return ''.join([idx2char[idx.item()] for idx in seq_tensor if idx2char[idx.item()] not in ['[PAD]', '[UNK]']])
    else:
        return [''.join([idx2char[idx.item()] for idx in seq if idx2char[idx.item()] not in ['[PAD]', '[UNK]']]) for seq in seq_tensor]

def train_one_epoch(model, loader, criterion, optimizer, device, idx2char):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        # print(labels[0])
        optimizer.zero_grad()
        outputs = model(images)  # [B, seq_len, num_classes]
        outputs = outputs.permute(0, 2, 1)  # [B, num_classes, seq_len]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
        # Decodifica e stampa alcune predizioni per debug (solo il primo batch per epoca)
        if progress_bar.n == 0:
            preds = outputs.permute(0, 2, 1).argmax(dim=2)  # [B, seq_len]
            pred_strs = decode_sequence(preds.cpu(), idx2char)
            label_strs = decode_sequence(labels.cpu(), idx2char)
            for i in range(min(3, len(pred_strs))):
                print(f"[DEBUG] GT: {label_strs[i]} | Pred: {pred_strs[i]}")
    return total_loss / len(loader)

def validate(model, loader, criterion, device, idx2char):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    shown = 0
    progress_bar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # [B, seq_len, num_classes]
            loss = criterion(outputs.permute(0, 2, 1), labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=2)  # [B, seq_len]
            mask = (labels != char2idx['[PAD]'])
            total_correct += ((preds == labels) & mask).sum().item()
            total_samples += mask.sum().item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    acc = total_correct / total_samples if total_samples > 0 else 0
    return total_loss / len(loader), acc


# -------------------------------
# 8. Main training script
# -------------------------------

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, idx2char)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, idx2char)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if (epoch+1) % 20 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.9

torch.save(model.state_dict(), 'pdlpr_trained.pth')

# -------------------------------
# 9. Funzione di test post-training
# -------------------------------

def test_random_val_image(model, val_dataset, idx2char, device):
    model.eval()
    idx = random.randint(0, len(val_dataset)-1)
    image, label, img_name = val_dataset[idx]
    input_img = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_img)  # [1, seq_len, num_classes]
        pred_seq = output.argmax(dim=2).squeeze(0).cpu().numpy()
    label_str = ''.join([idx2char[i] for i in label if idx2char[i] not in ['[PAD]', '[UNK]']])
    pred_str = ''.join([idx2char[i] for i in pred_seq if idx2char[i] not in ['[PAD]', '[UNK]']])
    img_np = image.permute(1,2,0).numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0,1)
    plt.imshow(img_np)
    plt.title(f"GT: {label_str}\nPred: {pred_str}")
    plt.axis('off')
    plt.show()
    print(f"Immagine: {img_name}\nGT: {label_str}\nPred: {pred_str}")

# Esempio di utilizzo dopo il training:
test_random_val_image(model, val_dataset, idx2char, DEVICE)
