# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:04:29 2025

@author: fedes
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- CHAR MAPPING ----------
index_to_char = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津',
    '晋', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏',
    '皖', '湘', '新', '渝', '豫', '粤', '云', '浙'
]
char_to_index = {c: i for i, c in enumerate(index_to_char)}

# ---------- DATASET ----------
class CCPDCharacters(Dataset):
    def __init__(self, root_dir, files, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            fname = self.files[idx]
            img_path = os.path.join(self.root_dir, fname)
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                parts = fname.split('-')
                if len(parts) < 5:
                    raise Exception("Formato filename errato")
                points_str = parts[3].split('_')
                points = [tuple(map(int, p.split('&'))) for p in points_str]
                if len(points) != 4:
                    raise Exception(f"Punti non validi: {points}")
                warped = self.warp_plate(img, points)
                chars = self.segment_characters(warped)
                labels = list(map(int, parts[4].split('_')))
                if len(labels) < len(chars):
                    raise Exception(f"Etichette troppo corte: {labels}")
                labels = labels[:len(chars)]
                if self.transform:
                    chars = [self.transform(c) for c in chars]
                return torch.stack(chars), torch.tensor(labels)
            except Exception as e:
                print(f"[ERRORE]: {fname} → {e}")
                idx = (idx + 1) % len(self.files)

    def warp_plate(self, image, points):
        pts = np.array(points, dtype='float32')
        dst_pts = np.array([[0, 0], [136, 0], [136, 36], [0, 36]], dtype='float32')
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        return cv2.warpPerspective(image, M, (136, 36))

    def segment_characters(self, plate_img, n_chars=7):
        h, w = plate_img.shape[:2]
        char_width = w // n_chars
        chars = []
        for i in range(n_chars):
            x = i * char_width
            char = plate_img[:, x:x+char_width]
            char = cv2.resize(char, (48, 48))  # Dimensione per VGG
            chars.append(char)
        return chars

# ---------- MODEL VGG ----------
def get_vgg_model(num_classes):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

# ---------- TRAINING ----------
def train(model, train_loader, val_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            b, n, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w).to(device)
            labels = labels.view(-1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                b, n, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w).to(device)
                labels = labels.view(-1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

# ---------- TEST RANDOM IMAGE ----------
def test_on_random_image(dataset, model, device, index_to_char):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    chars_imgs, _ = dataset[idx]  # chars_imgs: [n_chars, 3, 48, 48]
    chars_imgs = chars_imgs.to(device)
    with torch.no_grad():
        outputs = model(chars_imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
    pred_str = ''.join([index_to_char[i] for i in preds])
    # Visualizza i caratteri segmentati
    fig, axes = plt.subplots(1, len(chars_imgs), figsize=(12, 2))
    for i, ax in enumerate(axes):
        img = chars_imgs[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f'Predicted: {pred_str}')
    plt.show()
    print('Caratteri riconosciuti:', pred_str)

# ---------- MAIN ----------
def main():
    data_dir = 'F:\\progetto computer vision\\dataset\\CCPD2019\\ccpd_base'  # Cambia con il tuo path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = CCPDCharacters(data_dir, train_files, transform=transform)
    val_dataset = CCPDCharacters(data_dir, val_files, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

    model = get_vgg_model(num_classes=len(index_to_char))
    model.to(device)
    train(model, train_loader, val_loader, device, epochs=15)
    torch.save(model.state_dict(), 'char_vgg.pth')
    print("Modello salvato in char_vgg.pth")
    test_on_random_image(val_dataset, model, device, index_to_char)

if __name__ == '__main__':
    main()
