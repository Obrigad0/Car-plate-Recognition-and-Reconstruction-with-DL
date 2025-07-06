import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CCPDDataset, create_splits  # Importa create_splits
from model import ResNetBBoxModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np

def calculate_iou(pred_boxes, true_boxes):
    """Calcola l'IoU (Intersection over Union) per batch di bounding box"""
    # Clamp per assicurare valori validi [0,1]
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    
    # Calcolo area di intersezione
    x1_i = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1_i = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2_i = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2_i = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    
    # Calcolo aree dei bounding box
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    # Calcolo IoU
    union = pred_area + true_area - intersection
    iou = intersection / (union + 1e-6)  # Evita divisioni per zero
    return iou.mean().item()  # Media dell'IoU per il batch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Addestra il modello per un'epoca con metriche"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    loop = tqdm(dataloader, leave=False)
    
    for images, bboxes in loop:
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()

        # Calcola IoU (senza gradienti)
        with torch.no_grad():
            batch_iou = calculate_iou(outputs, bboxes)
        
        # Aggiorna metriche
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_iou += batch_iou * batch_size
        
        # Aggiorna progress bar
        avg_loss = running_loss / ((loop.n + 1) * batch_size)
        avg_iou = running_iou / ((loop.n + 1) * batch_size)
        loop.set_description(f"Train Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_iou

def validate(model, dataloader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        loop = tqdm(dataloader, leave=False, desc="Validation")
        for images, bboxes in loop:
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            batch_iou = calculate_iou(outputs, bboxes)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_iou += batch_iou * batch_size
            
            # Aggiorna progress bar
            avg_loss = running_loss / ((loop.n + 1) * batch_size)
            avg_iou = running_iou / ((loop.n + 1) * batch_size)
            loop.set_description(f"Val Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Percorso dataset (modificare con il proprio)
    dataset_path = "F:\\progetto computer vision\\dataset\\CCPD2019"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Percorso dataset non trovato: {dataset_path}")
    
    # Caricamento dataset e creazione split
    dataset = CCPDDataset(dataset_path, transform=transform)
    train_data, val_data, test_data = create_splits(dataset)  # Usa la funzione importata
    
    # Dataloader
    batch_size  = 50 
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=False, num_workers=10)

    # Inizializzazione modello (stessa istanza per tutte le epoche)
    model = ResNetBBoxModel(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"EPOCA [{epoch+1}/{epochs}]")
        print(f"{'='*50}")
        
        # Training
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validazione
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Salvataggio metriche
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Report epoca
        print(f"\nRIEPILOGO EPOCA {epoch+1}:")
        print(f"  Training:   Loss = {train_loss:.4f} | IoU = {train_iou:.4f}")
        print(f"  Validation: Loss = {val_loss:.4f} | IoU = {val_iou:.4f}")
        print(f"  Delta Loss: {train_loss - val_loss:.4f}")
        print(f"  Delta IoU:  {val_iou - train_iou:.4f}")

        # Salvataggio miglior modello
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  Salvato nuovo miglior modello!")
    
    # Salvataggio finale
    torch.save(model.state_dict(), "resnet_bbox_model.pth")
    print("\nTraining completato!")
    print(f"Miglior validation loss: {best_val_loss:.4f}")
    print(f"Miglior validation IoU: {max(history['val_iou']):.4f}")
    print("Modello finale salvato in 'resnet_bbox_model.pth'")

if __name__ == "__main__":
    main()
