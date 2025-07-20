import torch
from torchvision import transforms
from PIL import Image

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mappa indici caratteri per OCR (stesso ordine del training)
index_to_char = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
    "青", "宁", "新", "警", "学", "O",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# --- Trasformazioni ---
transform_bbox = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

transform_ocr = transforms.Compose([
    transforms.Resize((48, 168)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Funzione di decodifica OCR ---
def decode_ocr(outputs):
    preds = [torch.argmax(o, dim=1) for o in outputs]
    preds = torch.cat(preds).cpu().numpy()
    return ''.join(index_to_char[i] for i in preds)

# --- Carica i modelli ---
import sys
sys.path.append("C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/baseline/bisio")
from model import UnifiedResNetModel

# Modello bbox (salvato da training bbox)
model_bbox = UnifiedResNetModel(pretrained=True).to(device)
model_bbox.load_state_dict(torch.load('C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/baseline/bisio/models/bisio_bbox_model.pth', map_location=device))
model_bbox.eval()

# Modello OCR (salvato da training OCR)
model_ocr = UnifiedResNetModel(head_type='ocr', pretrained=False).to(device)
model_ocr.load_state_dict(torch.load('C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/baseline/claudio/models/nuovo_ocr_Claudio_modello_aggiornato.pth', map_location=device))
model_ocr.eval()

# --- Pipeline ---
def recognize_plate_full(image_path):
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size

    # Prepara immagine per bbox
    input_bbox = transform_bbox(img).unsqueeze(0).to(device)

    # Predict bbox [x1, y1, x2, y2] normalizzati (0-1)
    with torch.no_grad():
        pred_bbox = model_bbox(input_bbox).squeeze(0)  # tensor 4 elementi

    # Denormalizza coordinate bbox
    x1 = int(pred_bbox[0].item() * img_width)
    y1 = int(pred_bbox[1].item() * img_height)
    x2 = int(pred_bbox[2].item() * img_width)
    y2 = int(pred_bbox[3].item() * img_height)

    # Clamp per sicurezza
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    # Crop targa
    cropped = img.crop((x1, y1, x2, y2))

    # Prepara immagine per OCR
    input_ocr = transform_ocr(cropped).unsqueeze(0).to(device)

    # Predict OCR
    with torch.no_grad():
        outputs_ocr = model_ocr(input_ocr)  # lista di tensori per carattere

    plate_text = decode_ocr(outputs_ocr)
    return plate_text, (x1, y1, x2, y2)

# --- Uso esempio ---
if __name__ == "__main__":
    img_path = 'targa.jpg'
    plate, bbox = recognize_plate_full(img_path)
    print(f"Targa riconosciuta: {plate}")
    print(f"BBox predetta: {bbox}")
