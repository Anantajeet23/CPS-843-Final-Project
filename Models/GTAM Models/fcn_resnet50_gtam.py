import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from gtam import GaborTAM


class FCN_ResNet50_GTAM(nn.Module):
    """FCN-ResNet50 with GTAM inserted after layer1."""

    def __init__(self, num_classes=2):
        super().__init__()

        base = fcn_resnet50(weights=None, weights_backbone=None, aux_loss=False)

        self.conv1 = base.backbone['conv1']
        self.bn1   = base.backbone['bn1']
        self.relu  = base.backbone['relu']
        self.maxpool = base.backbone['maxpool']

        self.layer1 = base.backbone['layer1']   # 256 channels
        self.layer2 = base.backbone['layer2']   # 512 channels
        self.layer3 = base.backbone['layer3']   # 1024 channels
        self.layer4 = base.backbone['layer4']   # 2048 channels

        self.gtam = GaborTAM(in_channels=256)


        # Replace classifier head for 2 classes
        self.classifier = base.classifier
        self.classifier[4] = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # -------- Encoder ----------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)     # (B,256,H/4,W/4)
        x = self.gtam(x)       # GTAM 

        x = self.layer2(x)     # 512
        x = self.layer3(x)     # 1024
        x = self.layer4(x)     # 2048

        # -------- FCN classifier --------
        out = self.classifier(x)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": out}
