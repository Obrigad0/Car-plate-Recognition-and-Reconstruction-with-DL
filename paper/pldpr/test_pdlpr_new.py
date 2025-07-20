import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import io
import matplotlib.pyplot as plt
from pdlpr import PDLPR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# ----------- TRASFORMAZIONI PER TEST (SENZA AUGMENTATION) -----------

class TestTransforms:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((48, 144)),
            transforms.ToTensor()
        ])
    
    def __call__(self, img):
        return self.transforms(img)

# ----------- DECODIFICA TARGA E TOKENIZZAZIONE -----------

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

# ----------- DATASET PER TEST -----------

class TestLPRDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=8):
        self.image_dir = os.path.join(root_dir, "images")
        self.labels_path = os.path.join(root_dir, "labels.txt")
        self.transform = transform if transform else TestTransforms()
        self.max_len = max_len

        # Verifica che il file esista
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"File labels.txt non trovato in: {self.labels_path}")

        with open(self.labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename, label = parts[0], ''.join(parts[1:])
                self.samples.append((filename, label))
            else:
                print(f"[Warning] Riga ignorata (malformata): {line}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    @staticmethod
    def load_and_concat_datasets(base_dir, transform=None, max_len=8):
        """
        Metodo statico per concatenare dataset train e val
        """
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "val")
        
        # Verifica che le directory esistano
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Directory train non trovata: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Directory val non trovata: {val_dir}")
        
        train_dataset = TestLPRDataset(train_dir, transform=transform, max_len=max_len)
        val_dataset = TestLPRDataset(val_dir, transform=transform, max_len=max_len)
        
        print(f"Dataset train caricato: {len(train_dataset)} campioni")
        print(f"Dataset val caricato: {len(val_dataset)} campioni")
        
        concat_dataset = ConcatDataset([train_dataset, val_dataset])
        print(f"Dataset concatenato: {len(concat_dataset)} campioni totali")
        
        return concat_dataset

def load_and_concat_test_datasets(base_dir, transform=None, max_len=8):
    """
    Carica e concatena i dataset di train e validation in un unico dataset per il testing.
    
    Args:
        base_dir (str): Directory base che contiene le cartelle 'train' e 'val'
        transform: Trasformazioni da applicare alle immagini
        max_len (int): Lunghezza massima delle sequenze
        
    Returns:
        ConcatDataset: Dataset concatenato di train + val
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # Verifica che le directory esistano
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory train non trovata: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Directory val non trovata: {val_dir}")
    
    # Crea i due dataset usando la tua classe TestLPRDataset
    train_dataset = TestLPRDataset(train_dir, transform=transform, max_len=max_len)
    val_dataset = TestLPRDataset(val_dir, transform=transform, max_len=max_len)
    
    print(f"Dataset train caricato: {len(train_dataset)} campioni")
    print(f"Dataset val caricato: {len(val_dataset)} campioni")
    
    # Concatena i due dataset
    concat_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"Dataset concatenato: {len(concat_dataset)} campioni totali")
    
    return concat_dataset

def test_collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)
    token_seqs = [torch.tensor(tokenizer.encode(t)[:seq_len] + [0]*(seq_len-len(t))) for t in texts]
    targets = torch.stack(token_seqs)
    return images, targets

# ----------- METRICHE DI VALUTAZIONE -----------

def calculate_sequence_accuracy(outputs, targets):
    """
    Calcola l'accuratezza per sequenza completa (targa intera corretta)
    """
    batch_size, seq_len, num_classes = outputs.shape
    predicted = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]
    
    # Confronta sequenza per sequenza
    correct_sequences = 0
    for i in range(batch_size):
        # Considera solo i token non-padding (target != 0)
        mask = targets[i] != 0
        if torch.all(predicted[i][mask] == targets[i][mask]):
            correct_sequences += 1
    
    return correct_sequences / batch_size

def calculate_character_accuracy(outputs, targets):
    """
    Calcola l'accuratezza per singolo carattere
    """
    predicted = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]
    
    # Maschera per escludere i token di padding
    mask = targets != 0
    
    # Conta i caratteri corretti (escludendo padding)
    correct_chars = ((predicted == targets) & mask).sum().item()
    total_chars = mask.sum().item()
    
    return correct_chars / total_chars if total_chars > 0 else 0.0

def calculate_edit_distance(pred_str, gt_str):
    """
    Calcola la distanza di edit (Levenshtein) tra due stringhe
    """
    len_pred, len_gt = len(pred_str), len(gt_str)
    dp = [[0] * (len_gt + 1) for _ in range(len_pred + 1)]
    
    for i in range(len_pred + 1):
        dp[i][0] = i
    for j in range(len_gt + 1):
        dp[0][j] = j
    
    for i in range(1, len_pred + 1):
        for j in range(1, len_gt + 1):
            if pred_str[i-1] == gt_str[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len_pred][len_gt]

# ----------- VISUALIZZAZIONE RISULTATI -----------

def visualize_predictions(dataset, model, device, num_samples=10, save_path="test_predictions.png"):
    """
    Visualizza le predizioni del modello su un campione di immagini
    """
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_input = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_input)
            predicted_indices = torch.argmax(output, dim=-1).squeeze(0)
            pred_str = tokenizer.decode(predicted_indices.cpu().numpy())
        
        # Prepara l'immagine per la visualizzazione
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.229 + 0.485).clip(0, 1)  # Denormalizza
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Pred: {pred_str}\nGT: {label}", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predizioni salvate in: {save_path}")

def plot_confusion_matrix_per_position(all_predictions, all_targets, save_path="position_confusion_matrices.png"):
    """
    Crea matrici di confusione per ogni posizione della targa
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for pos in range(seq_len):
        pos_pred = [pred[pos] if pos < len(pred) else 0 for pred in all_predictions]
        pos_target = [target[pos] if pos < len(target) else 0 for target in all_targets]
        
        # Filtra i padding (0)
        filtered_pred = [p for p, t in zip(pos_pred, pos_target) if t != 0]
        filtered_target = [t for t in pos_target if t != 0]
        
        if filtered_pred and filtered_target:
            cm = confusion_matrix(filtered_target, filtered_pred)
            
            axes[pos].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[pos].set_title(f'Posizione {pos + 1}')
            axes[pos].set_xlabel('Predetto')
            axes[pos].set_ylabel('Reale')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Matrici di confusione salvate in: {save_path}")

# ----------- FUNZIONE PRINCIPALE DI TEST -----------

def PDLPR_testing(test_data_folder, model_path="pdlpr_best_model.pth", batch_size=32, use_concat=True):
    import os

    # Carica il dataset di test
    if use_concat:
        test_dataset = load_and_concat_test_datasets(test_data_folder, transform=TestTransforms())
    else:
        test_dataset = TestLPRDataset(test_data_folder, transform=TestTransforms())

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=test_collate_fn, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

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

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modello caricato da: {model_path}")
    else:
        print(f"ERRORE: File del modello non trovato: {model_path}")
        return

    model.eval()
    total_loss = 0.0
    total_char_acc = 0.0
    total_seq_acc = 0.0
    total_edit_distance = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []
    all_pred_strings = []
    all_gt_strings = []

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    print("Iniziando valutazione del modello...")
    pbar = tqdm(test_loader, desc="Testing", unit="batch")

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        with autocast(device_type="cuda"):
            outputs = model(images)

            output_reshaped = outputs.view(-1, outputs.size(-1))
            targets_reshaped = targets.view(-1)
            loss = loss_fn(output_reshaped, targets_reshaped)

            char_acc = calculate_character_accuracy(outputs, targets)
            seq_acc = calculate_sequence_accuracy(outputs, targets)

            predicted = torch.argmax(outputs, dim=-1)

            total_loss += loss.item()
            total_char_acc += char_acc
            total_seq_acc += seq_acc

            for i in range(predicted.shape[0]):
                pred_indices = predicted[i].cpu().numpy()
                target_indices = targets[i].cpu().numpy()

                pred_str = tokenizer.decode(pred_indices)
                gt_str = tokenizer.decode(target_indices)

                edit_dist = calculate_edit_distance(pred_str, gt_str)
                total_edit_distance += edit_dist
                total_samples += 1

                all_predictions.append(pred_indices.tolist())
                all_targets.append(target_indices.tolist())
                all_pred_strings.append(pred_str)
                all_gt_strings.append(gt_str)

        pbar.set_postfix({
            "loss": loss.item(),
            "char_acc": char_acc,
            "seq_acc": seq_acc
        })

    avg_loss = total_loss / len(test_loader)
    avg_char_acc = total_char_acc / len(test_loader)
    avg_seq_acc = total_seq_acc / len(test_loader)
    avg_edit_distance = total_edit_distance / total_samples

    print("\n" + "=" * 60)
    print("RISULTATI DEL TEST")
    print("=" * 60)
    print(f"Numero totale di campioni: {total_samples}")
    print(f"Loss medio: {avg_loss:.4f}")
    print(f"Accuratezza per carattere: {avg_char_acc:.4f} ({avg_char_acc * 100:.2f}%)")
    print(f"Accuratezza per sequenza: {avg_seq_acc:.4f} ({avg_seq_acc * 100:.2f}%)")
    print(f"Distanza di edit media: {avg_edit_distance:.4f}")
    print("=" * 60)

    # Analisi errori
    error_count = {}
    for pred_str, gt_str in zip(all_pred_strings, all_gt_strings):
        if pred_str != gt_str:
            error_type = f"{gt_str} -> {pred_str}"
            error_count[error_type] = error_count.get(error_type, 0) + 1

    sorted_errors = sorted(error_count.items(), key=lambda x: x[1], reverse=True)
    print("Errori più comuni (Top 10):")
    for i, (error, count) in enumerate(sorted_errors[:10]):
        print(f"{i + 1:2d}. {error} ({count} volte)")

    # Crea cartella risultati
    output_dir = "./results/rotate"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Predizioni esempio
    visualize_predictions(test_dataset, model, device, num_samples=10,
                          save_path=os.path.join(output_dir, "test_predictions.png"))

    # 2. Matrici di confusione
    plot_confusion_matrix_per_position(all_predictions, all_targets,
                                       save_path=os.path.join(output_dir, "position_confusion_matrices.png"))

    # 3. Edit distance distribution
    edit_distances = [calculate_edit_distance(pred, gt) for pred, gt in zip(all_pred_strings, all_gt_strings)]
    plt.figure(figsize=(10, 6))
    plt.hist(edit_distances, bins=range(max(edit_distances) + 2), alpha=0.7, edgecolor='black')
    plt.xlabel('Distanza di Edit')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione delle Distanze di Edit')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "edit_distance_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Accuratezza per posizione
    position_accuracies = []
    for pos in range(seq_len):
        pos_correct = 0
        pos_total = 0
        for pred, target in zip(all_predictions, all_targets):
            if pos < len(target) and target[pos] != 0:
                if pos < len(pred) and pred[pos] == target[pos]:
                    pos_correct += 1
                pos_total += 1
        acc = pos_correct / pos_total if pos_total > 0 else 0.0
        position_accuracies.append(acc)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, seq_len + 1), position_accuracies, alpha=0.7)
    plt.xlabel('Posizione del Carattere')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza per Posizione del Carattere')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    for i, acc in enumerate(position_accuracies):
        plt.text(i + 1, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, "position_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Confusioni top-10
    def top_confusions(all_predictions, all_targets, tokenizer, output_path):
        confusion_dict = {}
        for pred_seq, target_seq in zip(all_predictions, all_targets):
            for p, t in zip(pred_seq, target_seq):
                if t != 0 and p != t:
                    true_char = tokenizer.idx2char.get(t, '')
                    pred_char = tokenizer.idx2char.get(p, '')
                    if true_char and pred_char:
                        key = f"{true_char} -> {pred_char}"
                        confusion_dict[key] = confusion_dict.get(key, 0) + 1

        sorted_conf = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)

        chinese_conf = [c for c in sorted_conf if c[0][0] in provinces][:10]
        alphanum_conf = [c for c in sorted_conf if c[0][0] in alphabets + [str(i) for i in range(10)]][:10]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Top-10 confusioni caratteri cinesi:\n")
            for c, n in chinese_conf:
                f.write(f"{c}: {n} volte\n")
            f.write("\nTop-10 confusioni alfanumeriche:\n")
            for c, n in alphanum_conf:
                f.write(f"{c}: {n} volte\n")

    top_confusions(all_predictions, all_targets, tokenizer,
                   output_path=os.path.join(output_dir, "top_confusions.txt"))

    # 6. Report testuale
    report_path = os.path.join(output_dir, "report_finale.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("====== REPORT TEST PDLPR ======\n")
        f.write(f"Totale campioni: {total_samples}\n")
        f.write(f"Loss medio: {avg_loss:.4f}\n")
        f.write(f"Accuratezza per carattere: {avg_char_acc:.4f} ({avg_char_acc * 100:.2f}%)\n")
        f.write(f"Accuratezza per sequenza: {avg_seq_acc:.4f} ({avg_seq_acc * 100:.2f}%)\n")
        f.write(f"Distanza di edit media: {avg_edit_distance:.4f}\n\n")
        f.write("Errori più comuni (Top 10):\n")
        for i, (error, count) in enumerate(sorted_errors[:10]):
            f.write(f"{i + 1:2d}. {error} ({count} volte)\n")

        f.write("\nAccuratezza per posizione:\n")
        for i, acc in enumerate(position_accuracies):
            f.write(f"Pos {i + 1}: {acc:.4f}\n")

    print(f"\nRisultati salvati nella cartella '{output_dir}'")

    return {
        "avg_loss": avg_loss,
        "avg_char_acc": avg_char_acc,
        "avg_seq_acc": avg_seq_acc,
        "avg_edit_distance": avg_edit_distance,
        "error_count": error_count,
        "position_accuracies": position_accuracies,
    }





if __name__ == '__main__':
    # Test del modello sul dataset di test con dataset concatenato
    results = PDLPR_testing(
        test_data_folder="C:/Users/fedes/Desktop/datibellissimi/ccpd_rotate",
        model_path="C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/paper/pldpr/models/pdlpr_best_model.pth",
        batch_size=32,
        use_concat=True  # Usa train + val concatenati
    )
    
    # Esempio di test su singola immagine (opzionale)
    # pred = test_single_image("path/to/your/test/image.jpg", "pdlpr_best_model.pth")
