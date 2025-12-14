import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Additive attention gate from Oktay et al. (Attention U-Net).
    x: skip feature (B, Cx, Hx, Wx)
    g: gating feature from decoder (B, Cg, Hg, Wg)
    returns: gated skip with same spatial size as x
    """
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        # linear projections
        self.theta_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=2, stride=2, bias=False)
        self.phi_g   = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, bias=True)
        self.psi     = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        self.bn_x = nn.BatchNorm2d(inter_channels)
        self.bn_g = nn.BatchNorm2d(inter_channels)

        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()

        # upsample attention to skip size
        self.upsample_mode = "bilinear"

    def forward(self, x, g):
        # project skip and gate
        theta_x = self.bn_x(self.theta_x(x))            
        phi_g   = self.bn_g(self.phi_g(g))              

        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode=self.upsample_mode, align_corners=True)

        f = self.relu(theta_x + phi_g)                  
        att = self.sigm(self.psi(f))                    

        # upsample attention to x's spatial size
        att = F.interpolate(att, size=x.shape[2:], mode=self.upsample_mode, align_corners=True)

        # gate skip
        return x * att
