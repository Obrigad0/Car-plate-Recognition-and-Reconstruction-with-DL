import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from crnn_model import CRNN
from tqdm import tqdm

# Lista ufficiale dei 68 caratteri CCPD
CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '辽', '吉', '黑', '苏', '浙',
    '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '琼', '川', '贵',
    '云', '陕', '甘', '青', '蒙', '桂', '宁', '新', '藏', '使', '领',
    '警', '学', '港', '澳',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

char2idx = {char: i + 1 for i, char in enumerate(CHARS)}  # partono da 1
idx2char = {i + 1: char for i, char in enumerate(CHARS)}
blank_idx = 0  # CTC blank index

# Salva mapping per inferenza futura
with open('char_map.json', 'w', encoding='utf-8') as f:
    json.dump(char2idx, f, ensure_ascii=False, indent=2)

class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(label_file, encoding='utf-8') as f:
            self.data = [line.strip().split('\t') for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img = Image.open(os.path.join(self.image_dir, img_name)).convert('L')
        if self.transform:
            img = self.transform(img)
        label_encoded = torch.tensor([char2idx[c] for c in label], dtype=torch.long)
        return img, label_encoded, len(label)

def collate_batch(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(labels)
    lengths = torch.tensor(lengths)
    return images, targets, lengths

# Trasformazioni
transform = transforms.Compose([
    transforms.Resize((32, 100)),  # altezza 32, larghezza fissa
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset e DataLoader
train_dataset = OCRDataset('ccpd_dataset/train/labels.txt', 'ccpd_dataset/train/images', transform)
val_dataset = OCRDataset('ccpd_dataset/val/labels.txt', 'ccpd_dataset/val/images', transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)

# Dispositivo e modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(32, 1, len(CHARS) + 1, 256).to(device)  # +1 per blank
criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        images, targets, lengths = collate_batch(batch)
        images, targets = images.to(device), targets.to(device)

        preds = model(images)  # output: [w, b, classes]
        preds_log_softmax = preds.log_softmax(2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)

        loss = criterion(preds_log_softmax, targets, input_lengths, lengths.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}")

    # Valutazione
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            images, targets, lengths = collate_batch(batch)
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            preds_log_softmax = preds.log_softmax(2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)

            loss = criterion(preds_log_softmax, targets, input_lengths, lengths.to(device))
            val_loss += loss.item()

    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

    # Salvataggio modello
    torch.save(model.state_dict(), f'crnn_chinese_plate_epoch{epoch}.pth')
