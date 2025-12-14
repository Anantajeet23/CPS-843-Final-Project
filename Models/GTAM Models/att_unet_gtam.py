import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from attention import AttentionGate
from gtam import GaborTAM

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class AttentionUNet_GTAM(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(64,128,256,512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)

        # ---------------- Encoder ----------------
        self.enc1 = DoubleConv(in_channels, features[0])
        self.gtam1 = GaborTAM(features[0])  # 64

        self.enc2 = DoubleConv(features[0], features[1])    # 128
        self.enc3 = DoubleConv(features[1], features[2])    # 256
        self.enc4 = DoubleConv(features[2], features[3])    # 512

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3]*2)  # 1024

        # Upconvs
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)  # 1024→512
        self.up3 = nn.ConvTranspose2d(features[3],   features[2], 2, 2)  # 512→256
        self.up2 = nn.ConvTranspose2d(features[2],   features[1], 2, 2)  # 256→128
        self.up1 = nn.ConvTranspose2d(features[1],   features[0], 2, 2)  # 128→64

        # Attention Gates (skip Cx, gate Cg, inter = Cx//2)
        self.ag4 = AttentionGate(in_channels_x=features[3], in_channels_g=features[3],   inter_channels=features[3]//2)
        self.ag3 = AttentionGate(in_channels_x=features[2], in_channels_g=features[2],   inter_channels=features[2]//2)
        self.ag2 = AttentionGate(in_channels_x=features[1], in_channels_g=features[1],   inter_channels=features[1]//2)
        self.ag1 = AttentionGate(in_channels_x=features[0], in_channels_g=features[0],   inter_channels=features[0]//2)

        # Decoder DoubleConvs (concat skip + up)
        self.dec4 = DoubleConv(features[3] + features[3], features[3])   # (512+512)->512
        self.dec3 = DoubleConv(features[2] + features[2], features[2])   # (256+256)->256
        self.dec2 = DoubleConv(features[1] + features[1], features[1])   # (128+128)->128
        self.dec1 = DoubleConv(features[0] + features[0], features[0])   # (64+64)->64

        # Head
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        # -------- Encoder --------
        e1 = self.enc1(x)
        e1 = self.gtam1(e1) # 64
        e2 = self.enc2(self.pool(e1))     # 128
        e3 = self.enc3(self.pool(e2))     # 256
        e4 = self.enc4(self.pool(e3))     # 512

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))   # 1024

        # Decoder stage 4 (512)
        g4 = self.up4(b)                      # gate @ 512
        if g4.shape[2:] != e4.shape[2:]:
            g4 = TF.resize(g4, e4.shape[2:])
        e4g = self.ag4(e4, g4)
        d4  = self.dec4(torch.cat([g4, e4g], dim=1))

        # Decoder stage 3 (256)
        g3 = self.up3(d4)
        if g3.shape[2:] != e3.shape[2:]:
            g3 = TF.resize(g3, e3.shape[2:])
        e3g = self.ag3(e3, g3)
        d3  = self.dec3(torch.cat([g3, e3g], dim=1))

        # Decoder stage 2 (128)
        g2 = self.up2(d3)
        if g2.shape[2:] != e2.shape[2:]:
            g2 = TF.resize(g2, e2.shape[2:])
        e2g = self.ag2(e2, g2)
        d2  = self.dec2(torch.cat([g2, e2g], dim=1))

        # Decoder stage 1 (64)
        g1 = self.up1(d2)
        if g1.shape[2:] != e1.shape[2:]:
            g1 = TF.resize(g1, e1.shape[2:])
        e1g = self.ag1(e1, g1)
        d1  = self.dec1(torch.cat([g1, e1g], dim=1))

        return self.head(d1)
