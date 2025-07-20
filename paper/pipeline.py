import torch
from torchvision import transforms
from PIL import Image
import sys

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Configurazioni ---
BLANK_TOKEN = '-'
SEQ_LEN = 7

CHARS = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣',
    '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', 'O',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', BLANK_TOKEN
]

# --- Trasformazioni ---
transform_pdlpr = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Funzione di decodifica ---
def decode_prediction(prediction, chars=CHARS, blank_token=BLANK_TOKEN):
    decoded = []
    for idx in prediction:
        char = chars[idx]
        if char != blank_token:
            decoded.append(char)
    return ''.join(decoded)

# --- Caricamento modelli ---
# YOLOv5
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/paper/yolo/models/best.pt')
yolo_model.conf = 0.4
yolo_model.to(device)
yolo_model.eval()

# PDLPR
sys.path.append("C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/paper/pldpr")
from pdlpr import PDLPR

model = PDLPR(
    in_channels=3,
    base_channels=256,
    encoder_d_model=256,
    encoder_nhead=8,
    encoder_height=16,
    encoder_width=16,
    decoder_num_layers=2,
    num_classes=69,
    seq_len=7
).to(device)

model.load_state_dict(torch.load("C:/Users/fedes/Desktop/fishing-game/Car-plate-Recognition-and-Reconstruction-with-DL/paper/pldpr/models/pdlpr_final.pth", map_location=device))
model.eval()

# --- Pipeline completa ---
def recognize_license_plate(image_path):
    img = Image.open(image_path).convert('RGB')

    # YOLOv5 detection
    results = yolo_model(img)
    detections = results.pandas().xyxy[0]

    if len(detections) == 0:
        print("Nessuna targa rilevata.")
        return None

    # Croppa prima targa
    box = detections.iloc[0]
    x1, y1, x2, y2 = map(int, [box['xmin'], box['ymin'], box['xmax'], box['ymax']])
    cropped_img = img.crop((x1, y1, x2, y2))

    # Preprocessing per PDLPR
    input_tensor = transform_pdlpr(cropped_img).unsqueeze(0).to(device)

    # Inferenza PDLPR
    with torch.no_grad():
        output = model(input_tensor)  # [1, seq_len, num_classes]
        prediction = output.argmax(dim=-1).squeeze(0)  # [seq_len]

    # Decodifica
    plate_string = decode_prediction(prediction.cpu().numpy())
    return plate_string

# --- Esempio d’uso ---
image_path = "targa.jpg"
targa = recognize_license_plate(image_path)
print("Targa rilevata:", targa)
