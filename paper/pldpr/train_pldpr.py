import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -- Mappa dei caratteri del dataset CCPD Base (modifica se hai una label map diversa) --
CHARS = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂',
         '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学', 'O', # province (34)
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
         'X', 'Y', 'Z', 'O', # alphabet (24+1 O)
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # digit (10+O?) , 'O'
CHAR2IDX = {ch:i for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
SEQ_LEN = 7 # Lunghezza delle targhe CCPD

# --- DATASET ---
class LicensePlateDataset(data.Dataset):
    def __init__(self, img_dir, label_file, img_size=(48, 144), transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        with open(label_file, encoding='utf-8') as f:
            for line in f:
                name, label = line.strip().split('\t')
                self.samples.append((name, label))

    def __len__(self):
        return len(self.samples)

    def encode_label(self, label):
        return [CHAR2IDX[c] for c in label]

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, name)
        img = Image.open(img_path).convert('RGB').resize(self.img_size[::-1], Image.BILINEAR)
        if self.transform: img = self.transform(img)
        label_encoded = self.encode_label(label)
        label_tensor = torch.LongTensor(label_encoded)
        return img, label_tensor, len(label_encoded)

# --- DATA AUGMENTATION ---
def get_transforms(is_train):
    aug = []
    if is_train:
        aug.extend([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.RandomAffine(degrees=4, translate=(.04,.12), scale=(.92,1.08))
            # Aggiungi altre randomizzazioni se vuoi
        ])
    aug.append(transforms.ToTensor())
    return transforms.Compose(aug)

# --- COLLATE FUNCTION per batch CTC ---
def ctc_collate(batch):
    imgs, labels, label_lens = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat([l for l in labels])
    label_lens = torch.tensor(label_lens)
    return imgs, labels, label_lens

# -- Training hyperparams --
BATCH_SIZE = 128
VAL_BATCH_SIZE = 5
IMG_SIZE = (96, 288)
EPOCHS = 20
INIT_LR = 1e-2
LR_MIN = 1e-5
DECAY_EVERY = 206
DECAY_FACTOR = 0.9

# -- Carica dati --
DATA_ROOT = 'F:/progetto computer vision/dataxricChar/'
train_set = LicensePlateDataset(
    img_dir=os.path.join(DATA_ROOT, 'train', 'images'),
    label_file=os.path.join(DATA_ROOT, 'train', 'labels.txt'),
    img_size=IMG_SIZE,
    transform=get_transforms(True)
)
val_set = LicensePlateDataset(
    img_dir=os.path.join(DATA_ROOT, 'val', 'images'),
    label_file=os.path.join(DATA_ROOT, 'val', 'labels.txt'),
    img_size=IMG_SIZE,
    transform=get_transforms(False)
)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate, drop_last=True, num_workers=0, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=ctc_collate, num_workers=0)

# --- MODELLI ---
from pdlpr import PDLPR # Inserisci qui il tuo modello
model = PDLPR(in_channels=3, d_model=512, n_heads=8, num_units=3, seq_len=SEQ_LEN, num_classes=NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# --- LOSS e OPTIMIZER ---
ctc_loss = nn.CTCLoss(blank=NUM_CLASSES-1, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

# --- LR Scheduler ---
def adjust_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        old_lr = param_group["lr"]
        new_lr = max(LR_MIN, old_lr * factor)
        param_group["lr"] = new_lr

# --- TRAIN LOOP ---
def train_one_epoch(epoch):
    model.train()
    total_loss, total_count = 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for imgs, labels, label_lens in pbar:
        imgs = imgs.to(device)
        targets = labels.to(device)
        label_lens = label_lens.to(device)
        batch_size = imgs.size(0)
        # Model output: [B, seq_len, num_classes] → [seq_len, B, num_classes] for CTC
        logits = model(imgs)       # [B, seq_len, num_classes]
        log_probs = logits.log_softmax(-1).transpose(0, 1)   # [seq_len, B, num_classes]
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long).to(device)
        loss = ctc_loss(log_probs, targets, input_lengths, label_lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_count += batch_size
        pbar.set_postfix(loss=loss.item())
    return total_loss / total_count

def evaluate():
    model.eval()
    total_loss, total_count = 0, 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for imgs, labels, label_lens in pbar:
            imgs = imgs.to(device)
            targets = labels.to(device)
            label_lens = label_lens.to(device)
            batch_size = imgs.size(0)
            logits = model(imgs)
            log_probs = logits.log_softmax(-1).transpose(0, 1)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long).to(device)
            loss = ctc_loss(log_probs, targets, input_lengths, label_lens)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            pbar.set_postfix(loss=loss.item())
    return total_loss / total_count


train_losses = []
val_losses = []

best_val_loss = float("inf")
epochs_since_lr_decay = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    val_loss = evaluate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    # Salva modello migliore
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_pdlpr.pth')
        print("** Miglior modello salvato! **")

    # Decresci learning rate se la validation non migliora ogni DECAY_EVERY
    epochs_since_lr_decay += 1
    if epochs_since_lr_decay >= DECAY_EVERY and val_loss >= best_val_loss:
        adjust_lr(optimizer, DECAY_FACTOR)
        print(f"** LR decayed, nuova lr: {[g['lr'] for g in optimizer.param_groups]} **")
        epochs_since_lr_decay = 0

# --- GRAFICO delle loss ---
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")  # salva il grafico su disco
plt.show()
