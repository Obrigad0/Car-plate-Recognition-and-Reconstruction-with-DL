import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Focus Structure Module ---
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # print(f"[Focus] input: {x.shape}")
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # print(f"[Focus] after slicing+concat: {x.shape}")
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # print(f"[Focus] after conv: {x.shape}")
        return x

# --- Residual Block ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.act2(out)
        return out

# --- ConvDownSampling Module ---
class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # print(f"[ConvDownSampling] output: {x.shape}")
        return x

# --- Improved Global Feature Extractor (IGFE) ---
class IGFE(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super(IGFE, self).__init__()
        self.focus = Focus(in_channels, 64)
        self.down1 = ConvDownSampling(64, 128)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.down2 = ConvDownSampling(128, 256)
        self.res3 = ResBlock(256)
        self.res4 = ResBlock(256)
        self.down3 = ConvDownSampling(256, out_channels)
        self.res5 = ResBlock(out_channels)
        self.res6 = ResBlock(out_channels)

    def forward(self, x):
        # print(f"[IGFE] input: {x.shape}")
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down3(x)
        x = self.res5(x)
        x = self.res6(x)
        # print(f"[IGFE] output: {x.shape}")
        return x  # [B, 512, 6, 18]

# --- Positional Encoding 2D ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model

        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        x_pos = torch.arange(0, width, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        for i in range(0, d_model, 2):
            pe[i, :, :] = torch.sin(y_pos * div_term[i // 2]).repeat(1, width)
            pe[i + 1, :, :] = torch.cos(x_pos * div_term[i // 2]).repeat(height, 1)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # print(f"[PositionalEncoding2D] input: {x.shape}")
        x = x + self.pe[:, :, :x.size(2), :x.size(3)]
        # print(f"[PositionalEncoding2D] output: {x.shape}")
        return x

# --- Encoder Unit ---
class EncoderUnit(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.mha = nn.MultiheadAttention(d_model * 2, n_heads, batch_first=True)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, D]
        # print(f"[EncoderUnit] input: {x.shape}")
        x = self.fc1(x)
        attn_output, _ = self.mha(x, x, x)
        x = self.fc2(attn_output)
        x = self.norm(x)
        # print(f"[EncoderUnit] output: {x.shape}")
        return x

# --- Encoder (3 stacked units) ---
class Encoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, num_units=3, height=6, width=18):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.units = nn.ModuleList([
            EncoderUnit(d_model, n_heads) for _ in range(num_units)
        ])

    def forward(self, x):
        # print(f"[Encoder] input: {x.shape}")
        x = self.pos_enc(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        # print(f"[Encoder] after flatten: {x.shape}")
        for i, unit in enumerate(self.units):
            x = unit(x)
            # print(f"[Encoder] after unit {i}: {x.shape}")
        return x

# --- Parallel Decoder Unit ---
class ParallelDecoderUnit(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

    def forward(self, tgt, memory):
        # print(f"[ParallelDecoderUnit] tgt: {tgt.shape}, memory: {memory.shape}")
        attn_output1, _ = self.mha1(tgt, tgt, tgt, attn_mask=self.generate_square_subsequent_mask(self.seq_len).to(tgt.device))
        x = self.norm1(tgt + attn_output1)
        attn_output2, _ = self.mha2(x, memory, memory)
        x = self.norm2(x + attn_output2)
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        # print(f"[ParallelDecoderUnit] output: {x.shape}")
        return x

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

# --- Parallel Decoder (3 stacked units) ---
class ParallelDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, num_units=3, seq_len=18, num_classes=68):
        super().__init__()
        self.units = nn.ModuleList([
            ParallelDecoderUnit(d_model, n_heads, seq_len) for _ in range(num_units)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, memory):
        # print(f"[ParallelDecoder] input: {memory.shape}")
        B, N, d_model = memory.shape
        tgt = torch.zeros(B, N, d_model, device=memory.device)
        for i, unit in enumerate(self.units):
            tgt = unit(tgt, memory)
            # print(f"[ParallelDecoder] after unit {i}: {tgt.shape}")
        logits = self.classifier(tgt)  # [B, seq_len, num_classes]
        # print(f"[ParallelDecoder] logits: {logits.shape}")
        return logits

# --- PDLPR Model ---
class PDLPR(nn.Module):
    def __init__(self, in_channels=3, d_model=512, n_heads=8, num_units=3, seq_len=18, num_classes=68):
        super().__init__()
        self.igfe = IGFE(in_channels, d_model)
        self.encoder = Encoder(d_model, n_heads, num_units, height=6, width=18)
        self.decoder = ParallelDecoder(d_model, n_heads, num_units, seq_len, num_classes)

    def forward(self, x):
        # print(f"[PDLPR] input: {x.shape}")
        features = self.igfe(x)  # [B, 512, 6, 18]
        # print(f"[PDLPR] after IGFE: {features.shape}")
        encoded = self.encoder(features)  # [B, N, d_model], N=6*18=108
        # print(f"[PDLPR] after Encoder: {encoded.shape}")
        B = x.size(0)
        # Pooling per ottenere [B, 18, 512]
        encoded_seq = encoded.view(B, 6, 18, 512).mean(dim=1)
        # print(f"[PDLPR] encoded_seq for decoder: {encoded_seq.shape}")
        logits = self.decoder(encoded_seq)  # [B, 18, num_classes]
        # print(f"[PDLPR] output logits: {logits.shape}")
        return logits
