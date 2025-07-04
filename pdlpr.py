# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:53:34 2025

@author: fedes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CTCLoss

#import per la singola immagine (probabile non serva per dopo)
import torchvision.transforms as T
import cv2



# -----------------------------
# FOCUS MODULE (IGFE)
# -----------------------------
class FocusModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FocusModule, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        patch1 = x[:, :, 0::2, 0::2]
        patch2 = x[:, :, 0::2, 1::2]
        patch3 = x[:, :, 1::2, 0::2]
        patch4 = x[:, :, 1::2, 1::2]
        x = torch.cat((patch1, patch2, patch3, patch4), dim=1)
        return self.conv(x)


# -----------------------------
# Downsampling & Residual Block
# -----------------------------
class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(x + self.block(x))


# -----------------------------
# IGFE
# -----------------------------
class IGFE(nn.Module):
    def __init__(self):
        super(IGFE, self).__init__()
        self.focus = FocusModule(3, 64)
        self.down1 = ConvDownSampling(64, 128)
        self.res1 = ResBlock(128)
        self.down2 = ConvDownSampling(128, 256)
        self.res2 = ResBlock(256)

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        return x  # (B, 256, H', W')


# -----------------------------
# Encoder Block
# -----------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=n_heads, batch_first=True)
        self.conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x_attn, _ = self.mha(x, x, x)
        x = self.conv2(x_attn.transpose(1, 2)).transpose(1, 2)
        return self.norm(residual + x)


class Encoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, num_layers=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------
# Decoder Block
# -----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(DecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self_attn_out)
        cross_attn_out, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_out)
        return self.ffn(x)


class ParallelDecoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, seq_len=10, num_classes=37):
        super(ParallelDecoder, self).__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Parameter(torch.randn(seq_len, d_model))
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(3)])
        self.classifier = nn.Linear(d_model // 2, num_classes)

    def forward(self, encoder_output):
        B = encoder_output.size(0)
        tokens = self.token_embedding.unsqueeze(0).repeat(B, 1, 1)
        for layer in self.layers:
            tokens = layer(tokens, encoder_output)
        return self.classifier(tokens)


# -----------------------------
# Full Model PDLPR
# -----------------------------
class PDLPR(nn.Module):
    def __init__(self, seq_len=10, num_classes=37):
        super(PDLPR, self).__init__()
        self.igfe = IGFE()
        self.encoder_to_decoder = nn.Linear(256, 128)

        self.encoder = Encoder()
        self.decoder = ParallelDecoder(seq_len=seq_len, num_classes=num_classes)

    def forward(self, x):
        
        features = self.igfe(x)                      # (B, 256, H, W)
        print("Shape features after IGFE:", features.shape)

        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)  # (B, seq_len, 256)
    
        features = self.encoder_to_decoder(features) # (B, seq_len, 128) - riduce canali da 256 a 128
        print("Shape features after encoder_to_decoder:", features.shape)

        encoded = self.encoder(features)   
        print("Shape encoded after encoder:", encoded.shape)
          # (B, seq_len, 128)
        
        # Interpola a seq_len token
        seq_len = self.decoder.seq_len
        encoded = F.interpolate(encoded.permute(0, 2, 1), size=seq_len, mode="linear", align_corners=False).permute(0, 2, 1)
           
        decoded = self.decoder(encoded)
        return decoded  # (B, seq_len, num_classes)


# -----------------------------
# ESEMPIO DI INFERENZA
# -----------------------------
if __name__ == "__main__":
    idx_to_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', '-']
    model = PDLPR()
    transform = T.Compose([
    T.ToPILImage(),
    T.Resize((48, 144)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # perché usi immagini RGB
    ])
    
    # Carica immagine
    img = cv2.imread("auto2.jpg")  # oppure "dataset_crop/00001.jpg"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 48, 144)
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=2)[0]
        targa = ''.join([idx_to_char[i] for i in pred])
        print("Targa riconosciuta:", targa)
