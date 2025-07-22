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
import torchvision.transforms.functional as TF
import random
import io
from model import UnifiedResNetModel

from PIL import Image, ImageFilter
# ------ MAPPING CARATTERI -------
index_to_char = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
    "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]
char_to_index = {c: i for i, c in enumerate(index_to_char)}
NUM_CLASSES = len(index_to_char)
NUM_CHARS = 7

# ------ DATASET -------
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

# ------ MODELLO OCR MULTI-HEAD -------
class FullRobustAugmentation:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.Resize((48, 144)),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __call__(self, img):
        img = self.base(img)
        if random.random() < 0.5:
            img = self.random_motion_blur(img)
        if random.random() < 0.5:
            factor = random.uniform(0.3, 1.8)
            img = TF.adjust_brightness(img, factor)
        if random.random() < 0.5:
            img = self.random_occlusion(img)
        if random.random() < 0.5:
            img = self.random_compression(img)
        if random.random() < 0.5:
            img = self.add_fog(img)
        return TF.to_tensor(img)

    def random_motion_blur(self, img):
        kernel_size = random.choice([5, 9, 15])
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size / 5))

    def add_fog(self, img):
        fog = Image.new("RGB", img.size, color=(200, 200, 200))
        return Image.blend(img, fog, alpha=random.uniform(0.1, 0.4))

    def random_occlusion(self, img):
        draw = img.copy()
        w, h = draw.size
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        x1 = x0 + random.randint(10, 40)
        y1 = y0 + random.randint(10, 20)
        color = random.choice([(0, 0, 0), (255, 255, 255)])
        for x in range(x0, min(x1, w)):
            for y in range(y0, min(y1, h)):
                draw.putpixel((x, y), color)
        return draw

    def random_compression(self, img):
        buffer = io.BytesIO()
        quality = random.randint(10, 40)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

# ------ LOSS MULTI-CHAR -------
def multi_char_loss(outputs, labels):
    loss = 0
    for i, out in enumerate(outputs):
        loss += F.cross_entropy(out, labels[:, i])
    return loss / len(outputs)

# ------ ACCURACY PER CHAR -------
def multi_char_accuracy(outputs, labels):
    correct = 0
    total = labels.size(0) * labels.size(1)
    for i, out in enumerate(outputs):
        preds = torch.argmax(out, dim=1)
        correct += (preds == labels[:, i]).sum().item()
    return correct / total

# ------ TRAIN/VAL LOOP -------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    tqdm_bar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in tqdm_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        ocr_outputs = model(imgs)
        loss = multi_char_loss(ocr_outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tqdm_bar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    tqdm_bar = tqdm(loader, desc="Val", leave=False)
    with torch.no_grad():
        for imgs, labels in tqdm_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            ocr_outputs = model(imgs)
            loss = multi_char_loss(ocr_outputs, labels)
            acc = multi_char_accuracy(ocr_outputs, labels)
            total_loss += loss.item()
            total_acc += acc
            tqdm_bar.set_postfix(loss=loss.item(), acc=acc)
    return total_loss / len(loader), total_acc / len(loader)

# ------ DECODIFICA OUTPUT MODELLO -------
def decode_prediction(outputs):
    preds = [torch.argmax(out, dim=1) for out in outputs]
    preds = torch.stack(preds, dim=1)
    return [''.join([index_to_char[i] for i in row]) for row in preds.cpu().numpy()]

# ------ TEST VISIVO SU IMMAGINE CASUALE -------
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
    train_images = 'F:/progetto computer vision/dataxricChar/train/images'
    train_labels = 'F:/progetto computer vision/dataxricChar/train/labels.txt'
    val_images = 'F:/progetto computer vision/dataxricChar/val/images'
    val_labels = 'F:/progetto computer vision/dataxricChar/val/labels.txt'

    val_transform  = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_transform = transforms.Compose([
        FullRobustAugmentation(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dataset = CCPDCharCropDataset(train_images, train_labels, train_transform)
    val_dataset = CCPDCharCropDataset(val_images, val_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UnifiedResNetModel(head_type="ocr", pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    epochs = 30
    best_val_loss = float('inf')

    # Per plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'ocr_best_model_claudio_augum.pth')
            best_val_loss = val_loss

    print("Miglior modello salvato in ocr_best_model.pth")

   # --------- GRAFICO: LOSS ---------
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Val Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot_augum.png")
    plt.show()
    
    # --------- GRAFICO: ACCURACY ---------
    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, label="Val Accuracy per Char", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Character")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot_augum.png")
    plt.show()
    # --------- TEST SU IMMAGINE CASUALE ----------
    test_on_random_image(val_dataset, model, device)

if __name__ == '__main__':
    main()
