import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import generalized_box_iou, box_iou
from dataset import CCPDDataset, create_splits
from model import UnifiedResNetModel
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# Funzione di loss combinata: MSE + (1 - GIoU)
def bbox_giou_loss(pred_boxes, true_boxes):
    # pred/true in formato [x1,y1,x2,y2], normalized 0–1
    giou = generalized_box_iou(pred_boxes, true_boxes)  # [B]
    l1 = nn.functional.l1_loss(pred_boxes, true_boxes, reduction='none').mean(dim=1)
    return (l1 + (1 - giou)).mean()

def calculate_iou(pred_boxes, true_boxes):
    with torch.no_grad():
        pred = torch.clamp(pred_boxes, 0, 1)
        ious = box_iou(pred, true_boxes)  # [B, B]
        diag_ious = ious.diag()            # IoU per coppia pred-vera corrispondente
        return diag_ious.mean().item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = total_iou = 0
    for imgs, bboxes in tqdm(dataloader, desc="Train", leave=False):
        imgs, bboxes = imgs.to(device), bboxes.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, bboxes)
        loss.backward()
        optimizer.step()
        iou = calculate_iou(preds, bboxes)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_iou  += iou * bs
    return total_loss / len(dataloader.dataset), total_iou / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = total_iou = 0
    with torch.no_grad():
        for imgs, bboxes in tqdm(dataloader, desc="Val", leave=False):
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            preds = model(imgs)
            loss = criterion(preds, bboxes)
            iou = calculate_iou(preds, bboxes)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_iou  += iou * bs
    return total_loss / len(dataloader.dataset), total_iou / len(dataloader.dataset)

def plot_metrics(epochs, train_vals, val_vals, metric_name, save_path):
    plt.figure()
    plt.plot(epochs, train_vals, label=f'Train {metric_name}')
    plt.plot(epochs, val_vals, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds_path = "C:/Users/fedes/Downloads/CCPD2019/CCPD2019"
    if not os.path.exists(ds_path):
        raise FileNotFoundError(ds_path)

    dataset = CCPDDataset(ds_path, transform=transform)
    train_ds, val_ds, test_ds = create_splits(dataset)
    train_dl = DataLoader(train_ds, batch_size=50, shuffle=True, num_workers=10, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=50, shuffle=False, num_workers=10, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=50, shuffle=False, num_workers=10, pin_memory=True)

    model = UnifiedResNetModel(head_type='bbox', pretrained=True).to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    writer = SummaryWriter("tb_logs")
    best_loss = float('inf')
    early_stop = 0
    max_patience = 7
    start_epoch = 1
    
    epochs = []
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    # Per riprendere l'addestramento da un checkpoint si può usare questo codice
    # if os.path.exists("best_model_bbox.pth"):
    #     checkpoint = torch.load("best_model_bbox.pth", map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     best_loss = checkpoint['best_loss']
    #     start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, 26):
        print(f"\nEpoch {epoch}/25")
        train_loss, train_iou = train_one_epoch(model, train_dl, bbox_giou_loss, optimizer, device)
        val_loss, val_iou     = validate(model, val_dl, bbox_giou_loss, device)

        scheduler.step(val_loss)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("IoU", {"train": train_iou, "val": val_iou}, epoch)

        print(f"Train Loss {train_loss:.4f} | IoU {train_iou:.4f}")
        print(f"Val   Loss {val_loss:.4f} | IoU {val_iou:.4f}")
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, "best_model_bbox.pth")
            print("✅ Saved best model")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= max_patience:
                print("🛑 Early stopping")
                break
            
    # Grafici bellissimi
    plot_metrics(epochs, train_losses, val_losses, 'Loss', 'loss_plot.png')
    plot_metrics(epochs, train_ious, val_ious, 'IoU', 'iou_plot.png')
    print("📈 Plots saved: loss_plot.png, iou_plot.png")
    plt.show()


    # Test evaluation
    checkpoint = torch.load("best_model_bbox.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loss, test_iou = validate(model, test_dl, bbox_giou_loss, device)
    print(f"\nTest Loss {test_loss:.4f} | Test IoU {test_iou:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
