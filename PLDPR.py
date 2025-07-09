import torch
import torch.nn as nn

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # Slicing: divide l’immagine in 4 parti e concatena sui canali
        x = torch.cat([
            x[..., ::2, ::2],  # top-left
            x[..., ::2, 1::2], # top-right
            x[..., 1::2, ::2], # bottom-left
            x[..., 1::2, 1::2] # bottom-right
        ], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RESBLOCK(nn.Module):
    def __init__(self, channels):
        super(RESBLOCK, self).__init__()
        self.block1 = CNNBlock(channels, channels)
        self.block2 = CNNBlock(channels, channels)

    def forward(self, x):
        return x + self.block2(self.block1(x))
    
class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class IGFE(nn.Module):
    def __init__(self):
        super(IGFE, self).__init__()
        self.focus = Focus(3, 64)
        self.down1 = ConvDownSampling(64, 128)
        self.res1 = RESBLOCK(128)
        self.down2 = ConvDownSampling(128, 256)
        self.res2 = RESBLOCK(256)
        self.down3 = ConvDownSampling(256, 512)
        self.res3 = RESBLOCK(512)
        self.res4 = RESBLOCK(512)

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = self.res3(x)
        x = self.res4(x)
        return x  # dimensione finale: [batch, 512, 6, 18]

class Encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch, 512, 6, 18] → [batch, 108, 512]
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [batch, 108, 512]
        x = self.encoder(x)
        return x

class ParallelDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, num_classes=..., max_len=18):
        super(ParallelDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, memory, tgt):
        # memory: [batch, 108, 512]
        # tgt: [batch, seq_len, d_model] 
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.fc(out)
        return out

class PLDPR(nn.Module):
    def __init__(self, num_classes, max_len=18):
        super(PLDPR, self).__init__()
        self.igfe = IGFE()
        self.encoder = Encoder()
        self.embedding = nn.Embedding(num_classes, 512)
        self.decoder = ParallelDecoder(num_classes=num_classes, max_len=max_len)

    def forward(self, img, tgt_seq):
        features = self.igfe(img)
        encoded = self.encoder(features)
        tgt_emb = self.embedding(tgt_seq)  # [batch, seq_len, 512]
        output = self.decoder(encoded, tgt_emb)
        return output
