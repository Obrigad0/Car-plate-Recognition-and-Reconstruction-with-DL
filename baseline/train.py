import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CCPDDataset
from model import ResNetBBoxModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, leave=False)
    for images, bboxes in loop:
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / ((loop.n + 1) * images.size(0))
        loop.set_description(f"Train Loss: {avg_loss:.4f}")
    return running_loss / len(dataloader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # modifica il percorso (poi ne metteremo uno assoluto, anche se non credo possiamo caricare l'intero dataset su git)
    full_dataset = CCPDDataset("../Downloads/CCPD2019/CCPD2019/ccpd_base", transform=transform)
    # batch_size=50 nel paper
    dataloader = DataLoader(full_dataset, batch_size=50, shuffle=True, num_workers=4)

    model = ResNetBBoxModel(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5 # da cambiare (300 nel paper)
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    torch.save(model.state_dict(), "resnet_bbox_model.pth")
    print("Modello salvato in resnet_bbox_model.pth")

if __name__ == "__main__":
    main()
