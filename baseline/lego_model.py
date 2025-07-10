import torch
from model import UnifiedResNetModel
from model import split_model_and_save
from model import rebuild_model_from_parts
# --- Assicurati che queste import siano corrette rispetto alla struttura dei tuoi file ---
# from tuo_file_modello import UnifiedResNetModel, split_model_and_save, rebuild_model_from_parts

# Parametri di esempio
head_type = 'bbox'   # puoi cambiare in 'bbox'
num_chars = 7
num_classes = 68

# # 1. Crea il modello
# model = UnifiedResNetModel(
#     head_type=head_type,
#     pretrained=False,
#     num_chars=num_chars,
#     num_classes=num_classes
# )

# checkpoint = torch.load('./modelli/best_model.pth', map_location='cpu')
# model.load_state_dict(checkpoint)
# # 2. Scomponi e salva
# split_model_and_save(model, head_path='bbox_head.pth', backbone_path='backbone_vecchio_della_parte_solo_base_yolo.pth')
# print("Backbone e parte lineare salvati separatamente.")

# 3. Ricomponi il modello dai file salvati
model_loaded = rebuild_model_from_parts(
    head_type='bbox',
    backbone_path='backbone.pth',
    head_path='bbox_head.pth',
    num_chars=num_chars,
    num_classes=num_classes,
    device='cpu'
)

torch.save(model_loaded.state_dict(), 'bbox_+_backbone_model.pth')
print("Modello salvato con successo.")

# # 4. Test rapido con input dummy
# dummy_input = torch.randn(2, 3, 224, 224)
# with torch.no_grad():
#     output = model_loaded(dummy_input)
# print("Output del modello ricostruito:")
# if head_type == 'bbox':
#     print(output.shape)  # [batch, 4]
# else:
#     print(f"Lista di {len(output)} tensori, shape primo: {output[0].shape}")
