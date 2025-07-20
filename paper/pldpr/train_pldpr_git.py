import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import io
import matplotlib.pyplot as plt
from pdlpr import PDLPR
from tqdm import tqdm
# ----------- COSTANTI CCPD -----------

provinces = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
    "青", "宁", "新", "警", "学", "O"
]
alphabets = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O'
]
ads = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'
]

charset = provinces + [c for c in alphabets if c not in provinces] + [str(i) for i in range(10)]
charset = list(dict.fromkeys(charset))  # Rimuove duplicati mantenendo l'ordine

# ----------- UTILITÀ PREPROCESSING -----------

def parse_box_from_filename(filename):
    # Esempio: filename = "074-153_423-245&374_263&409-..." 
    parts = filename.split('-')
    box_str = parts[2]  # "245&374_263&409"
    (x1y1, x2y2) = box_str.split('_')
    x1, y1 = map(int, x1y1.split('&'))
    x2, y2 = map(int, x2y2.split('&'))
    return x1, y1, x2, y2

def crop_plate(img_path):
    img = Image.open(img_path).convert("RGB")
    filename = os.path.basename(img_path)
    x1, y1, x2, y2 = parse_box_from_filename(filename)
    left, top = min(x1, x2), min(y1, y2)
    right, bottom = max(x1, x2), max(y1, y2)
    return img.crop((left, top, right, bottom))

# ----------- AUGMENTATION AVANZATA -----------

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

# ----------- DECODIFICA TARGA E TOKENIZZAZIONE -----------

def decode_plate(plate_code):
    try:
        province = provinces[plate_code[0]]
        letter = alphabets[plate_code[1]]
        tail = ''.join(ads[i] for i in plate_code[2:])
        return province + letter + tail
    except Exception:
        return "INVALID"

def parse_filename(filename):
    parts = filename[:-4].split('-')
    plate_code = list(map(int, parts[4].split('_')))
    return decode_plate(plate_code)

class SimplePlateTokenizer:
    def __init__(self, charset):
        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 = PAD
        self.char2idx['<PAD>'] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
    def encode(self, text):
        for c in text:
            if c not in self.char2idx:
                print(f"[Tokenizer Warning] Carattere '{c}' non nel charset! Verrà codificato come PAD (0)")
        return [self.char2idx.get(c, 0) for c in text]
    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '') for i in indices if i != 0])
    def vocab_size(self):
        return len(self.char2idx)

tokenizer = SimplePlateTokenizer(charset)
num_classes = tokenizer.vocab_size()
seq_len = 8  # Lunghezza massima targa CCPD

# ----------- DATASET CCPD -----------

class CCPDPlateDataset(Dataset):
    def __init__(self, image_folder, transform=None, max_len=8):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform else FullRobustAugmentation()
        self.max_len = max_len
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, filename)
        image = crop_plate(img_path)
        image = self.transform(image)
        label_text = parse_filename(filename)
        return image, label_text

def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)
    token_seqs = [torch.tensor(tokenizer.encode(t)[:seq_len] + [0]*(seq_len-len(t))) for t in texts]
    targets = torch.stack(token_seqs)
    if (targets >= num_classes).any() or (targets < 0).any():
        print("[ERROR] Target fuori range! Ecco alcune label e codifiche:")
        for t in texts:
            print("Label:", t, "Encoded:", tokenizer.encode(t))
        print("Target tensor:", targets)
        print("num_classes:", num_classes)
        raise ValueError("Target fuori range per CrossEntropyLoss!")
    return images, targets




# ----------- FUNZIONE TRAINING -----------

def PDLPR_training(image_folder, num_epochs, batch_size=32):
    dataset = CCPDPlateDataset(image_folder)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PDLPR(
        in_channels=3,
        base_channels=256,
        encoder_d_model=256,
        encoder_nhead=4,
        encoder_height=16,
        encoder_width=16,
        decoder_num_layers=2,
        num_classes=num_classes,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(device="cuda")

    



    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(images)
                output = output.permute(0, 2, 1)
                loss = loss_fn(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item()})
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", unit="batch"):
                images = images.to(device)
                targets = targets.to(device)
                with autocast(device_type="cuda"):
                    output = model(images)
                    output = output.permute(0, 2, 1)
                    loss = loss_fn(output, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")

        # Cambia il path dove vuoi salvare i pesi
        # torch.save(model.state_dict(), f"pdlpr_epoch{epoch + 1}.pth")

    torch.save(model.state_dict(), "pdlpr_final.pth")

    # --- Plot delle loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.close()
    print("Salvato grafico delle loss in 'loss_plot.png'")

# ----------- ESEMPIO USO -----------
PDLPR_training("F:\progetto computer vision\dataset\CCPD2019\ccpd_base", num_epochs=30)

