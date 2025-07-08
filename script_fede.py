import torch
checkpoint = torch.load('best.pt', map_location='cpu')
torch.save(checkpoint['model'].float().state_dict(), 'best_clean.pt')