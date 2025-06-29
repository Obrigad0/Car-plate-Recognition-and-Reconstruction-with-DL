import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CCPDDataset
from model import ResNetBBoxModel
import time
from tqdm import tqdm

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2] in formato normalizzato
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    correct = 0
    total = 0
    total_time = 0.0

# qui la valutazione viene fatta basandomi sempre sul paper a pag 13, sezione 4.3
    with torch.no_grad():
        for images, gt_bboxes in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            start = time.time()
            outputs = model(images).cpu()
            end = time.time()
            total_time += (end - start)

            for pred_box, gt_box in zip(outputs, gt_bboxes):
                iou = compute_iou(pred_box.tolist(), gt_box.tolist())
                total_iou += iou
                if iou >= 0.7:
                    correct += 1
                total += 1

    avg_iou = total_iou / total
    accuracy = correct / total
    avg_time = total_time / total
    fps = 1 / avg_time

    print(f"\n Evaluation results:")
    print(f"- Average IoU: {avg_iou:.4f}")
    print(f"- Accuracy (IoU > 0.7): {accuracy:.4f}")
    print(f"- Inference time per image: {avg_time * 1000:.2f} ms")
    print(f"- FPS: {fps:.2f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Carica l'intero dataset
    # Modifica anche qui con il tuo path
    dataset = CCPDDataset("F:\progetto computer vision\dataset\CCPD2019\ccpd_base", transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)

    # Carica il modello
    model = ResNetBBoxModel(pretrained=False).to(device)
    model.load_state_dict(torch.load("resnet_bbox_model.pth", map_location=device))

    # Esegui valutazione
    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
