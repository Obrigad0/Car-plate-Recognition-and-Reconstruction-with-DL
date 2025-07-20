import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Focus Structure Module ---
class FocusStructure(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1):
        super(FocusStructure, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # slicing
        x1 = x[..., ::2, ::2]
        x2 = x[..., ::2, 1::2]
        x3 = x[..., 1::2, ::2]
        x4 = x[..., 1::2, 1::2]
        #concatenation
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class RESBLOCK(nn.Module):
    def __init__(self, channels):
        super(RESBLOCK, self).__init__()
        self.cnn1 = CNNBlock(channels, channels)
        self.cnn2 = CNNBlock(channels, channels)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        return out + x  # residual connection

class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class IGFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = FocusStructure()            # [3, 48, 144] → [32, 24, 72]
        self.res1 = RESBLOCK(32)
        self.res2 = RESBLOCK(32)
        self.down1 = ConvDownSampling(32, 256)   # [12, 24, 72] → [256, 12, 36]
        self.res3 = RESBLOCK(256)
        self.res4 = RESBLOCK(256)
        self.down2 = ConvDownSampling(256, 512)  # [256, 12, 36] → [512, 6, 18]

    def forward(self, x):
        x = self.focus(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down1(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down2(x)
        return x  # Final output: [B, 512, 6, 18]
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.height = height
        self.width = width

        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")

        pe = torch.zeros(d_model, height, width)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        pos_w = torch.arange(0, width).unsqueeze(1)
        pos_h = torch.arange(0, height).unsqueeze(1)

        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, d_model, H, W]

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
    
class EncoderUnit(nn.Module):
    def __init__(self, d_model=512, d_mha=1024, n_heads=8, height=6, width=18):
        super(EncoderUnit, self).__init__()
        self.height = height
        self.width = width
        self.d_mha = d_mha

        self.conv1 = nn.Conv2d(d_model, d_mha, kernel_size=1)
        self.mha = nn.MultiheadAttention(d_mha, n_heads, batch_first=True)
        self.conv2 = nn.Conv2d(d_mha, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)  # **LayerNorm**

    def forward(self, x):
        B, C, H, W = x.shape
        # Conv1 → [B, 1024, 6, 18]
        x1 = self.conv1(x)

        # Flatten per MHA → [B, 108, 1024]
        x1_seq = x1.view(B, self.d_mha, H * W).permute(0, 2, 1)

        # Multi-Head Attention
        attn_out, _ = self.mha(x1_seq, x1_seq, x1_seq)  # [B, 108, 1024]

        # Reshape a feature map and CNN BLOCK 2
        attn_out = attn_out.permute(0, 2, 1).view(B, self.d_mha, H, W)
        x2 = self.conv2(attn_out)  # [B, 512, 6, 18]

        # Residual + LayerNorm **(token-wise)**
        # Flatten of x e x2 → [B, 108, 512]
        x2_seq = x2.view(B, C, H * W).permute(0, 2, 1)
        x_res_seq = x.view(B, C, H * W).permute(0, 2, 1)

        norm_out = self.norm(x2_seq + x_res_seq)

        # Return to the previous dimensions [B, 512, 6, 18]
        out = norm_out.permute(0, 2, 1).view(B, C, H, W)
        return out

class Encoder(nn.Module):
    def __init__(self, d_model=512, d_mha=1024, n_heads=8, num_units=3, height=6, width=18):
        super(Encoder, self).__init__()
        self.pe2d = PositionalEncoding(d_model, height, width)
        self.encoder_units = nn.ModuleList([
            EncoderUnit(d_model=d_model, d_mha=d_mha, n_heads=n_heads, height=height, width=width)
            for _ in range(num_units)
        ])

    def forward(self, x):
        # x: [B, 512, 6, 18]
        x = self.pe2d(x)
        for unit in self.encoder_units:
            x = unit(x)
        return x  # [B, 512, 6, 18]
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=18):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):  # x: [B, T, D]
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_len={self.pe.size(1)} in PositionalEncoding1D.")
        return x + self.pe[:, :x.size(1)].to(x.device)

# === Masked Self-Attention ===
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        return self.norm(x + attn_output)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        attn_output, _ = self.attn(x, context, context)
        return self.norm(x + attn_output)


# === Feed Forward Network ===
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = MaskedMultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, dropout=dropout)

    def forward(self, x, encoder_out):
        x = self.self_attn(x)
        x = self.cross_attn(x, encoder_out)
        x = self.ffn(x)
        return x



# CNN BLOCK3 (2x1 kernel, stride=(3,1), padding=1)
class CNNBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=(2, 1), stride=(3, 1), padding=(1,0))
        self.norm = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))  # [B, 512, 3, 18] → approx

class CNNBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=(3, 1), padding=(0, 0))
        self.norm = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))  # [B, 512, 1, 18]

class ParallelTransformerDecoder(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=3, max_seq_len=18, dropout=0.1):
        super().__init__()
        self.block3 = CNNBlock3()
        self.block4 = CNNBlock4()
        self.pos_encoder = PositionalEncoding1D(embed_dim, max_len=max_seq_len)

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, encoder_out):
        x = self.block3(encoder_out)
        x = self.block4(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, x)
        logits = self.output_layer(x)
        return logits


class PLDPR(nn.Module):
    def __init__(self, num_classes, seq_len=18, dropout=0.1):
        super().__init__()
        self.igfe = IGFE()
        self.encoder = Encoder()
        self.decoder = ParallelTransformerDecoder(num_classes=num_classes, max_seq_len=seq_len, dropout=dropout)

    def forward(self, x):
        x = self.igfe(x)
        x = self.encoder(x)
        logits = self.decoder(x)
        return logits
