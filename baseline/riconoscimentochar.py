import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# ----------------- MAPPING CARATTERI -----------------
index_to_char = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
    "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]
char_to_index = {c: i for i, c in enumerate(index_to_char)}
NUM_CLASSES = len(index_to_char)
NUM_CHARS = 7  # Numero di caratteri targa

# ----------------- DATASET -----------------
class CCPDCharCropDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        with open(labels_path, encoding='utf-8') as f:
            for line in f:
                img_name, label = line.strip().split('\t')
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_indices = torch.tensor([char_to_index[c] for c in label])
        return img, label_indices

# ----------------- MODELLO OCR PROFONDO -----------------
class OCRResNetMultiHead(nn.Module):
    def __init__(self, backbone, num_chars=NUM_CHARS, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone  # backbone già caricato e senza fc
        if hasattr(self.backbone, 'fc') and not isinstance(self.backbone.fc, nn.Identity):
            num_features = self.backbone.fc.in_features
        else:
                num_features = 512  # o il valore corretto per il tuo backbone

        # Testa OCR profonda
        self.ocr_head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.ocr_class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            ) for _ in range(num_chars)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        ocr_feats = self.ocr_head(feats)
        outs = [head(ocr_feats) for head in self.ocr_class_heads]
        return outs  # lista di [batch, num_classes] per ogni carattere

# ----------------- LOSS OCR MULTI-CHAR -----------------
def multi_char_loss(outputs, labels):
    loss = 0
    for i, out in enumerate(outputs):
        loss += F.cross_entropy(out, labels[:, i])
    return loss / len(outputs)

# ----------------- TRAINING E VALIDAZIONE -----------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        ocr_outputs = model(imgs)
        loss = multi_char_loss(ocr_outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            ocr_outputs = model(imgs)
            loss = multi_char_loss(ocr_outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# ----------------- DECODE PREDICTION -----------------
def decode_prediction(outputs):
    preds = [torch.argmax(out, dim=1) for out in outputs]
    preds = torch.stack(preds, dim=1)
    return [''.join([index_to_char[i] for i in row]) for row in preds.cpu().numpy()]

# ----------------- TEST SU IMMAGINE RANDOM -----------------
def test_on_random_image(dataset, model, device):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    img, label = dataset[idx]
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        ocr_outputs = model(img_input)
        pred_indices = [torch.argmax(out, dim=1).item() for out in ocr_outputs]
    pred_str = ''.join([index_to_char[i] for i in pred_indices])
    gt_str = ''.join([index_to_char[i] for i in label.numpy()])
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)
    plt.imshow(img_np)
    plt.title(f"Pred: {pred_str} | GT: {gt_str}")
    plt.axis('off')
    plt.show()
    print('Caratteri riconosciuti:', pred_str)
    print('Ground Truth:', gt_str)

# ----------------- MAIN -----------------
def main():
    # Sostituisci con i tuoi path reali
    train_images = 'F:/progetto computer vision/dataxricChar/train/images'
    train_labels = 'F:/progetto computer vision/dataxricChar/train/labels.txt'
    val_images = 'F:/progetto computer vision/dataxricChar/val/images'
    val_labels = 'F:/progetto computer vision/dataxricChar/val/labels.txt'

    transform = transforms.Compose([
        transforms.Resize((48, 168)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = CCPDCharCropDataset(train_images, train_labels, transform)
    val_dataset = CCPDCharCropDataset(val_images, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------- CARICA BACKBONE DA FILE ----------
    backbone = models.resnet18(pretrained=False)
    backbone.fc = nn.Identity()
    # Carica i pesi del backbone addestrato per detection
    checkpoint = torch.load('./modelli/best_model.pth', map_location='cuda')
    backbone.load_state_dict(checkpoint, strict=False)

    # --------- CREA MODELLO OCR PROFONDO ----------
    model = OCRResNetMultiHead(backbone, num_chars=NUM_CHARS, num_classes=NUM_CLASSES).to(device)
    for name, param in model.backbone.named_parameters():
        param.requires_grad = 'layer4' in name
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), #model.parameters(),
        lr=0.001,
        weight_decay=3e-4)
    epochs = 20

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'ocr_from_detection_backbone.pth')
    print("Modello OCR salvato in ocr_from_detection_backbone.pth")

    # Test su un'immagine random del validation set
    test_on_random_image(val_dataset, model, device)

if __name__ == '__main__':
    main()
