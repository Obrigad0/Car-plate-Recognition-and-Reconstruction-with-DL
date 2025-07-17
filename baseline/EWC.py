import torch
from model import UnifiedResNetModel

import torch.nn as nn
import torchvision.transforms as transforms
from dataset import CCPDDataset, create_splits  # Importa create_splits
class EWC:
    def __init__(self, model, dataset, device='cpu', fisher_samples=200):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.fisher_samples = fisher_samples
        self.params = {n: p for n, p in model.backbone.named_parameters() if p.requires_grad}
        self._means = {n: p.detach().clone().to(device) for n, p in self.params.items()}
        self._precision_matrices = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}

    def estimate_fisher(self, loss_fn):
        self.model.eval()
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        total = 0
        for i, (inputs, targets) in enumerate(loader):
            if total >= self.fisher_samples:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            for n, p in self.model.backbone.named_parameters():
                if p.grad is not None:
                    self._precision_matrices[n] += p.grad.data.clone().pow(2)
            total += 1
        for n in self._precision_matrices:
            self._precision_matrices[n] /= total

    def save(self, path):
        torch.save({'means': self._means,
                    'fisher': self._precision_matrices}, path)

# USO: esegui dopo il primo training, in un file a sé stante

if __name__ == "__main__":
    # Definisci/importa il modello e la classe (deve essere identica a quella usata per training)
    model = UnifiedResNetModel(head_type='bbox')
    checkpoint = torch.load("best_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
        
    # Data e loss del primo task
    dataset_task1 = CCPDDataset(
        img_dir="F:\\progetto computer vision\\dataset\\CCPD2019",
        mode='train',
        transform=transform   # Uguale al training
    )   
    loss_fn_task1 =  nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ewc = EWC(model, dataset_task1, device=device, fisher_samples=800)
    print("Stima Fisher (può richiedere qualche minuto)...")
    ewc.estimate_fisher(loss_fn_task1)
    ewc.save("ewc_parameters.pth")
    print("EWC parameters salvati!")

