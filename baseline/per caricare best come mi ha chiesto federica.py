import torch




checkpoint = torch.load('C://Users//flavi//OneDrive//Documenti//GitHub//Car-plate-Recognition-and-Reconstruction-with-DL//paper model//yolo model trained//weights//best.pt', map_location='cpu')
print(checkpoint)
torch.save(checkpoint['model'].float().state_dict(), 'best_clean.pt')