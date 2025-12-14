import torch
import torch.nn as nn
import math


class GaborTAM(nn.Module):
    """
    Gabor-based Texture Attention Module (GTAM)

    Input:  feature map F of shape (B, C, H, W)
    Output: F' of shape (B, C, H, W) with texture-based spatial attention

    Steps:
      1. Project F -> 1 channel (texture map).
      2. Convolve with fixed Gabor filter bank.
      3. Fuse Gabor responses into a 1-channel attention map.
      4. Apply sigmoid to get spatial attention A (B, 1, H, W).
      5. Return F * A + F (residual).
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        n_orientations: int = 4,  # 0°, 45°, 90°, 135°
        n_frequencies: int = 2,   # low + medium spatial frequency
        gamma: float = 0.5,       # aspect ratio for Gabor
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.num_kernels = n_orientations * n_frequencies

        # Project feature map to single-channel 
        self.project = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        # Fixed Gabor conv: 1 -> num_kernels 
        padding = kernel_size // 2
        self.gabor_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_kernels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
    
        for param in self.gabor_conv.parameters():
            param.requires_grad = False

        # Initialize Gabor kernels
        gabor_kernels = self._create_gabor_kernels(
            kernel_size=kernel_size,
            n_orientations=n_orientations,
            n_frequencies=n_frequencies,
            gamma=gamma,
        )
        with torch.no_grad():
            self.gabor_conv.weight.copy_(gabor_kernels)

        # Fuse Gabor responses -> attention map
        hidden_channels = max(self.num_kernels // 2, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.num_kernels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),  # spatial attention in [0, 1]
        )

        self.last_attn = None   # will store (B,1,h,w) attention from the last forward pass

    def _create_gabor_kernels(
        self,
        kernel_size: int,
        n_orientations: int,
        n_frequencies: int,
        gamma: float,
    ) -> torch.Tensor:
        """
        Create a bank of 2D Gabor filters as a tensor of shape:
            (num_kernels, 1, kernel_size, kernel_size)

        - orientations evenly spaced in [0, π)
        - frequencies chosen from a small set of wavelengths
        """
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, steps=kernel_size),
            torch.linspace(-half, half, steps=kernel_size),
            indexing="ij",
        )

    
        base_lambda = 3.0  
        lambdas = [base_lambda, base_lambda * 1.7]  # two frequencies

        thetas = [i * math.pi / n_orientations for i in range(n_orientations)]
        num_kernels = n_orientations * n_frequencies

        kernels = torch.zeros(num_kernels, 1, kernel_size, kernel_size)

        idx = 0
        for lam in lambdas[:n_frequencies]:
            # Sigma scales with wavelength
            sigma = 0.56 * lam  # heuristic
            for theta in thetas:
                # Rotate coordinates
                x_theta = x * math.cos(theta) + y * math.sin(theta)
                y_theta = -x * math.sin(theta) + y * math.cos(theta)

                # Gabor formula (real part only)
                gb = torch.exp(
                    -0.5 * (x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (sigma ** 2)
                ) * torch.cos(2.0 * math.pi * x_theta / lam)

                # Normalize kernel to zero mean and unit L1 norm
                gb = gb - gb.mean()
                norm = gb.abs().sum()
                if norm > 0:
                    gb = gb / norm

                kernels[idx, 0, :, :] = gb
                idx += 1

        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W) with texture-based attention applied
        """
        # Project to single-channel "texture base"
        tex = self.project(x)          # (B, 1, H, W)

        # Apply fixed Gabor filter bank
        gabor_feats = self.gabor_conv(tex)   # (B, num_kernels, H, W)

        # Fuse Gabor responses into spatial attention
        attn = self.fuse(gabor_feats)  # (B, 1, H, W), values in [0, 1]

        # store attention for visualization
        self.last_attn = attn.detach()
        
        # Apply attention with residual connection
        out = x * attn + x             # (B, C, H, W)
        return out

        
