import os
import torch
import torch.nn as nn
from PLDPR import PLDPR 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.optim as optim


VOCAB = [
    "<PAD>",
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for i, c in enumerate(VOCAB)}
PAD_IDX = char2idx["<PAD>"]

def encode_label(label, max_len=7):
    indices = [char2idx[c] for c in label]
    indices += [PAD_IDX] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)

class CCPDDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, max_len=7):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        self.max_len = max_len
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_name, label = line.strip().split('\t')
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_tensor = encode_label(label, self.max_len)
        return image, label_tensor

transform = transforms.Compose([
    transforms.Resize((48, 144)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CCPDDataset(
    img_dir='./ccpd_dataset/train/images',
    label_file='./ccpd_dataset/train/labels.txt',
    transform=transform
)
val_dataset = CCPDDataset(
    img_dir='./ccpd_dataset/val/images',
    label_file='./ccpd_dataset/val/labels.txt',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=4)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        inputs = labels[:, :-1]
        targets = labels[:, 1:]
        outputs = model(images, inputs)
        outputs = outputs.permute(0, 2, 1)  # [batch, num_classes, seq_len]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            inputs = labels[:, :-1]
            targets = labels[:, 1:]
            outputs = model(images, inputs)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PLDPR(num_classes=len(VOCAB), max_len=7).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.9)

    num_epochs = 10
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_pldpr.pth')

