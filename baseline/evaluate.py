import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CCPDDataset
from model import ResNetBBoxModel
import time
from tqdm import tqdm
import os

def calculate_iou(pred_boxes, true_boxes):
    """Calcola IoU per batch (versione ottimizzata)"""
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    
    x1_i = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1_i = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2_i = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2_i = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    union = pred_area + true_area - intersection
    return intersection / (union + 1e-6)

def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    correct = 0
    total = 0
    total_time = 0.0

    with torch.no_grad():
        for images, gt_bboxes in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            start = time.time()
            outputs = model(images)
            end = time.time()
            total_time += (end - start) * images.size(0)
            
            # Calcolo IoU in batch
            ious = calculate_iou(outputs, gt_bboxes)
            total_iou += ious.sum().item()
            correct += (ious >= 0.7).sum().item()
            total += images.size(0)

    avg_iou = total_iou / total
    accuracy = correct / total
    avg_time = total_time / total / 1000  # ms per immagine
    fps = total / total_time

    print(f"\n Evaluation results:")
    print(f"- Average IoU: {avg_iou:.4f}")
    print(f"- Accuracy (IoU > 0.7): {accuracy:.4f}")
    print(f"- Inference time per image: {avg_time:.2f} ms")
    print(f"- FPS: {fps:.2f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Percorso base del dataset
    base_dataset_path = "F:\\progetto computer vision\\dataset\\CCPD2019"
    
    # Carica modello
    model = ResNetBBoxModel(pretrained=False).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Model loaded successfully")

    # Valutazione su tutte le sottocartelle
    dataset = CCPDDataset(
        img_dir=base_dataset_path,
        mode='evaluate',
        transform=transform
    )
    
    print(f"Total evaluation images: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=8)
    
    # Esegui valutazione
    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
