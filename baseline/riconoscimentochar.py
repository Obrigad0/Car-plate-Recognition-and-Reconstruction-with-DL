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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

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

# ---------- CNN MODEL ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------- DATASET ----------
class CCPDCharacters(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

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
    
                # Punti prospettici (parte -3)
                points_str = parts[3].split('_')
                points = [tuple(map(int, p.split('&'))) for p in points_str]
                if len(points) != 4:
                    raise Exception(f"Punti non validi: {points}")
    
                # Warp
                warped = self.warp_plate(img, points)
                chars = self.segment_characters(warped)
    
                # Etichette (parte -4)
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
            char = cv2.resize(char, (32, 32))
            chars.append(char)
        return chars

# ---------- TRAINING ----------
def train(model, dataloader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            b, n, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w).to(device)
            labels = labels.view(-1).to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# ---------- MAIN ----------
def main():
    data_dir = '../Downloads/CCPD2019/CCPD2019/ccpd_base'  # Cambia con il tuo path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = CCPDCharacters(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = SimpleCNN(num_classes=len(index_to_char))
    model.to(device)

    train(model, dataloader, device, epochs=5)

    torch.save(model.state_dict(), 'char_cnn.pth')
    print("Modello salvato in char_cnn.pth")

if __name__ == '__main__':
    main()
