import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from gtam import GaborTAM


class DeepLabV3_ResNet50_GTAM(nn.Module):
    """DeepLabV3-ResNet50 with GTAM inserted after layer1."""

    def __init__(self, num_classes=2):
        super().__init__()

        base = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            aux_loss=False
        )


        # Backbone layers (ResNet50)
        self.conv1   = base.backbone['conv1']
        self.bn1     = base.backbone['bn1']
        self.relu    = base.backbone['relu']
        self.maxpool = base.backbone['maxpool']

        self.layer1 = base.backbone['layer1']   # 256 channels
        self.layer2 = base.backbone['layer2']   # 512 channels
        self.layer3 = base.backbone['layer3']   # 1024 channels
        self.layer4 = base.backbone['layer4']   # 2048 channels


        # Insert GTAM after layer1
        self.gtam = GaborTAM(in_channels=256)

        self.aspp = base.classifier[0]   # ASPP module

        # Replace final classifier conv
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)


    def forward(self, x):
        input_shape = x.shape[-2:]

        # Backbone Encoder 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)    # (B, 256, H/4, W/4)
        x = self.gtam(x)      # Inserting GTAM 

        x = self.layer2(x)    # 512
        x = self.layer3(x)    # 1024
        x = self.layer4(x)    # 2048

        x = self.aspp(x)

        x = self.head(x)

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}
