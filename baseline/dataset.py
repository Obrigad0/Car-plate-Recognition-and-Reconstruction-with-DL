import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CCPDDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('RGB')
        
        x1, y1, x2, y2 = self.extract_bbox(fname)
        w, h = image.size
        bbox = torch.tensor([x1/w, y1/h, x2/w, y2/h], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, bbox
    
    def extract_bbox(self, filename):
        name = filename.replace(".jpg", "")
        parts = name.split('-')
        point1, point2 = parts[2].split('_')
        x1, y1 = map(int, point1.split('&'))
        x2, y2 = map(int, point2.split('&'))
        return x1, y1, x2, y2
