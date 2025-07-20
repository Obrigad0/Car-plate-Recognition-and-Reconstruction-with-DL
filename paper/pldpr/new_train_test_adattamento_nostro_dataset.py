import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pdlpr import PDLPR
# Classe custom per gestire il dataset CCPD pre-processato
class CCPDDataset(Dataset):
    def __init__(self, images_dir, labels_path, seq_len=8, transform=None):
        self.images_dir = images_dir
        with open(labels_path, encoding='utf-8') as f:
            self.samples = [line.strip().split('\t') for line in f.readlines()]
        self.seq_len = seq_len
        self.transform = transform

        # Dizionario carattere->indice, da aggiornare in base ai tuoi indici
        self.char2idx = self._build_vocab()

    def _build_vocab(self):
        # Costruisci un mapping carattere → indice coerente con il tuo preprocessing
        all_chars = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂',
                     '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学', 'O',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
                     'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        return {c: i+1 for i, c in enumerate(all_chars)}  # +1 per lasciare 0 come PAD

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image).transpose(2, 0, 1) / 255., dtype=torch.float32)

        # Conversione etichetta in sequenza di indici (padded con 0)
        label_indices = [self.char2idx.get(char, 0) for char in label]
        label_indices = label_indices + [0] * (self.seq_len - len(label_indices))
        label_indices = label_indices[:self.seq_len]
        return image, torch.tensor(label_indices, dtype=torch.long)

# Funzione per il training
def train(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # (B, seq_len, num_classes)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Training Loss: {avg_loss:.4f}")

        # Validazione
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()
                preds = outputs.argmax(-1)
                mask = (labels != 0)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}, Char Accuracy: {val_acc:.4%}")

if __name__ == "__main__":
    import numpy as np  # Assicurati che numpy sia installato
    from torchvision import transforms

    # Percorsi dei dati
    root_dir = 'F:\progetto computer vision\dataxricChar'
    train_img_dir = os.path.join(root_dir, 'train', 'images')
    val_img_dir = os.path.join(root_dir, 'val', 'images')
    train_label_path = os.path.join(root_dir, 'train', 'labels.txt')
    val_label_path = os.path.join(root_dir, 'val', 'labels.txt')

    # Parametri
    batch_size = 32
    seq_len = 8

    # Trasformazioni immagini
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Dataset e dataloader
    train_dataset = CCPDDataset(train_img_dir, train_label_path, seq_len=seq_len, transform=transform)
    val_dataset = CCPDDataset(val_img_dir, val_label_path, seq_len=seq_len, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Modello (passa tutti i parametri necessari)
    model = PDLPR(num_classes=68, seq_len=seq_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Avvio training
    train(model, train_loader, val_loader, device, epochs=20, lr=1e-4)
