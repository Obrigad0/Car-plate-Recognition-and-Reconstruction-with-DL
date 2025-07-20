import torch
import torch.nn as nn
import torch.nn.functional as F


# --- IGFE Components ---
class FocusStructure(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnn_block1 = CNNBlock(in_channels, out_channels)
        self.cnn_block2 = CNNBlock(out_channels, out_channels)
        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        identity = self.identity(x)
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        return out + identity


class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.conv(x)
        return x


class IGFE(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.focus = FocusStructure()
        self.layer1 = ResBlock(4 * in_channels, base_channels)
        self.layer2 = ResBlock(base_channels, base_channels)
        self.down1 = ConvDownSampling(base_channels, base_channels, stride=2)
        self.layer3 = ResBlock(base_channels, base_channels)
        self.layer4 = ResBlock(base_channels, base_channels)
        self.down2 = ConvDownSampling(base_channels, base_channels, stride=2)
    
    def forward(self, x):
        x = self.focus(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.down1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.down2(x)
        return x


# --- Encoder Components ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")
        
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))
        
        pe[0::4, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::4, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[2::4, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[3::4, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]


class EncoderModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.cnn_block1 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.cnn_block2 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.add_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x.clone()
        x = self.pos_enc(x)
        x = self.cnn_block1(x)
        B, C, H, W = x.shape
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        attn_out, _ = self.mha(x_, x_, x_)
        x = attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        x = self.cnn_block2(x)
        out = residual + x
        out = out.permute(0, 2, 3, 1)
        out = self.add_norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderModule(d_model, nhead, height, width) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --- Decoder Components ---
class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)


class DecodingModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(d_model, height, width)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.cross_cnn1 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_cnn2 = CNNBlock(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(d_model, d_model * 4, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model * 4, d_model, kernel_size=1),
        )
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)
    
    def forward(self, x, encoder_out):
        x = self.pos_enc(x)
        B, C, H, W = x.shape
        
        # Self attention
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        self_attn_out, _ = self.self_attn(x_, x_, x_)
        self_attn_out = self.addnorm1(x_.permute(1, 0, 2), self_attn_out.permute(1, 0, 2))
        self_attn_out = self_attn_out.permute(1, 0, 2)
        x = self_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        
        # Cross attention
        enc = self.cross_cnn1(encoder_out)
        enc = self.cross_cnn2(enc)
        B_enc, C_enc, H_enc, W_enc = enc.shape
        enc_ = enc.permute(2, 3, 0, 1).reshape(H_enc*W_enc, B_enc, C_enc)
        x_ = x.permute(2, 3, 0, 1).reshape(H*W, B, C)
        cross_attn_out, _ = self.cross_attn(x_, enc_, enc_)
        cross_attn_out = self.addnorm2(x_.permute(1, 0, 2), cross_attn_out.permute(1, 0, 2))
        cross_attn_out = cross_attn_out.permute(1, 0, 2)
        x = cross_attn_out.reshape(H, W, B, C).permute(2, 3, 0, 1)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        out = self.addnorm3(x.permute(0, 2, 3, 1).reshape(B, -1, C), ff_out.permute(0, 2, 3, 1).reshape(B, -1, C))
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class Decoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, height=16, width=16, num_layers=3, num_classes=68, seq_len=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DecodingModule(d_model=d_model, nhead=nhead, height=height, width=width)
            for _ in range(num_layers)
        ])
        self.seq_len = seq_len
        self.classifier = nn.Linear(d_model, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, seq_len))  # (B, C, 1, seq_len)
    
    def forward(self, x, encoder_out):
        for layer in self.layers:
            x = layer(x, encoder_out)
        x = self.pool(x)  # (B, C, 1, seq_len)
        x = x.squeeze(2)  # (B, C, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, C)
        logits = self.classifier(x)  # (B, seq_len, num_classes)
        return logits


# --- PDLPR Main Model ---
class PDLPR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=512,
                 encoder_d_model=512,
                 encoder_nhead=8,
                 encoder_height=16,
                 encoder_width=16,
                 decoder_num_layers=3,
                 num_classes=68,
                 seq_len=8):
        super().__init__()
        self.igfe = IGFE(in_channels, base_channels)
        self.pool = nn.AdaptiveAvgPool2d((encoder_height, encoder_width))
        self.encoder = Encoder(d_model=encoder_d_model, nhead=encoder_nhead, height=encoder_height, width=encoder_width)
        self.decoder = Decoder(
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            height=encoder_height,
            width=encoder_width,
            num_layers=decoder_num_layers,
            num_classes=num_classes,
            seq_len=seq_len
        )
    
    def forward(self, x):
        x = self.igfe(x)
        x = self.pool(x)
        x = self.encoder(x)
        decoder_input = torch.zeros_like(x)
        x = self.decoder(decoder_input, x)
        return x


