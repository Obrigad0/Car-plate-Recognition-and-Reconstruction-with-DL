import torch.nn as nn
import torchvision.models as models

class ResNetBBoxModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Carica ResNet18 senza il classificatore finale
        self.backbone = models.resnet18(pretrained=pretrained)
        # Rimuovi l'ultimo fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # sostituisci con identità

        # Testa di regressione bbox
        self.bbox_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # output normalizzato [0,1]
        )
    
    def forward(self, x):
        features = self.backbone(x)
        bbox = self.bbox_head(features)
        return bbox
