import torch
import torch.nn as nn
import torchvision.models as models

class UnifiedResNetModel(nn.Module):
    def __init__(
        self,
        head_type='bbox',
        pretrained=True,
        num_chars=7, # 7 caratteri nella targa di cui il primo la provincia (ideogramma) + valori targa
        num_classes=68   #68 caratteri possibili 
    ):
        """
        head_type: 'bbox' per la regressione bounding box, 'ocr' per multi-head OCR
        pretrained: se usare pesi pre-addestrati per la ResNet
        num_chars: necessario solo per head_type='ocr'
        num_classes: necessario solo per head_type='ocr'
        """
        super().__init__()
        self.head_type = head_type
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if head_type == 'bbox':
            self.bbox_head = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid()
            )
        elif head_type == 'ocr':
            assert num_chars is not None and num_classes is not None, \
                "num_chars e num_classes devono essere specificati per OCR"
            self.ocr_head = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU()
            )
            self.ocr_class_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                ) for _ in range(num_chars)
            ])
        else:
            raise ValueError("head_type deve essere 'bbox' o 'ocr'")

    def forward(self, x):
        features = self.backbone(x)
        if self.head_type == 'bbox':
            return self.bbox_head(features)
        elif self.head_type == 'ocr':
            ocr_feats = self.ocr_head(features)
            outs = [head(ocr_feats) for head in self.ocr_class_heads]
            return outs  # lista di [batch, num_classes] per ogni carattere

def split_model_and_save(model, head_path, backbone_path='backbone.pth'):
    """
    Salva separatamente backbone e parte lineare del modello.
    Args:
        model: Istanza di UnifiedResNetModel.
        backbone_path: Path dove salvare il backbone.
        head_path: Path dove salvare la parte lineare.
    """
    # Salva backbone
    torch.save(model.backbone.state_dict(), backbone_path)
    # Salva la parte lineare
    if model.head_type == 'bbox':
        torch.save(model.bbox_head.state_dict(), head_path)
    elif model.head_type == 'ocr':
        # Salva sia la ocr_head che le class_heads
        torch.save({
            'ocr_head': model.ocr_head.state_dict(),
            'ocr_class_heads': [h.state_dict() for h in model.ocr_class_heads]
        }, head_path)
    else:
        raise ValueError("Tipo di testa non supportato")

def rebuild_model_from_parts(
    head_type,
    backbone_path,
    head_path,
    num_chars=7,
    num_classes=68,
    device='cpu'
):
    """
    Ricostruisce un modello UnifiedResNetModel dato un backbone e una parte lineare salvati.
    Args:
        head_type: 'bbox' o 'ocr'
        backbone_path: Path dei pesi del backbone.
        head_path: Path dei pesi della parte lineare.
        num_chars: Solo per OCR.
        num_classes: Solo per OCR.
        device: 'cpu' o 'cuda'
    Returns:
        modello UnifiedResNetModel completo.
    """
    # Istanzia il modello
    model = UnifiedResNetModel(
        head_type=head_type,
        pretrained=False,
        num_chars=7,
        num_classes=68
    )
    # Carica backbone
    model.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    # Carica parte lineare
    if head_type == 'bbox':
        model.bbox_head.load_state_dict(torch.load(head_path, map_location=device))
    elif head_type == 'ocr':
        head_dict = torch.load(head_path, map_location=device)
        model.ocr_head.load_state_dict(head_dict['ocr_head'])
        for h, sd in zip(model.ocr_class_heads, head_dict['ocr_class_heads']):
            h.load_state_dict(sd)
    else:
        raise ValueError("Tipo di testa non supportato")
    model.to(device)
    model.eval()
    return model
