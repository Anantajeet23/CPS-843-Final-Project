'''
Part of this code is borrowed from aladdinpersson/Machine-Learning-Collection
'''
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from gtam import GaborTAM

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class UNET_GTAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super(UNET_GTAM,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # ===============================================
        # ---- GTAM on the first encoder level ----
        self.gtam1 = GaborTAM(features[0])
        # ===============================================

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self,x):
        skip_connections = []

        # ---- Encoder Level 1 (with GTAM) ----
        x1 = self.downs[0](x)
        x1 = self.gtam1(x1)      
        skip_connections.append(x1)
        x = self.pool(x1)

        # ---- Encoder Levels 2, 3, 4 (NO GTAM) ----
        for down in self.downs[1:]:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
         
        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x) 
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)

