import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import numpy as np

class CCPDDataset(Dataset):
    def __init__(self, img_dir, mode='train', transform=None):
        """
        Args:
            img_dir (str): Percorso della cartella principale del dataset
            mode (str): 'train' per training/validation, 'evaluate' per test esterni
            transform (callable, optional): Trasformazioni da applicare
        """
        self.transform = transform
        self.filenames = []
        
        if mode == 'train':
            # Modalità training: usa solo ccpd_base
            base_path = os.path.join(img_dir, 'ccpd_base')
            self.filenames = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.jpg')]
        elif mode == 'evaluate':
            # Modalità evaluate: usa tutte le sottocartelle tranne ccpd_base
            subfolders = ['ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']
            for folder in subfolders:
                folder_path = os.path.join(img_dir, folder)
                if os.path.exists(folder_path):
                    self.filenames.extend([
                        os.path.join(folder_path, f) 
                        for f in os.listdir(folder_path) 
                        if f.endswith('.jpg')
                    ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Estrai bbox dal nome del file
        filename = os.path.basename(img_path)
        x1, y1, x2, y2 = self.extract_bbox(filename)
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

def create_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    return train_set, val_set, test_set
