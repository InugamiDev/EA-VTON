# intent: composition U-Net that fuses agnostic person, warped garment, and
#         pose heatmaps into a final try-on image with a learned blending mask
# status: done
# next: experiment with spectral-norm or attention in bottleneck
# confidence: high

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, apply_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        ]
        if apply_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CompositionUNet(nn.Module):
    """U-Net that outputs a 3-channel rendered image and a 1-channel
    composition mask, then blends with the warped garment.

    Input channels: I_agnostic(3) + I_warped(3) + H_pose(18) = 24
    """

    def __init__(self, in_channels: int = 24):
        super().__init__()

        # ---- Encoder ----
        self.e1 = _EncoderBlock(in_channels, 64)    # /2
        self.e2 = _EncoderBlock(64, 128)             # /4
        self.e3 = _EncoderBlock(128, 256)            # /8
        self.e4 = _EncoderBlock(256, 512)             # /16
        self.e5 = _EncoderBlock(512, 512)             # /32

        # ---- Decoder ----
        self.d5 = _DecoderBlock(512, 512)             # /16  -> cat(e4) => 1024
        self.d4 = _DecoderBlock(1024, 256)            # /8   -> cat(e3) => 512
        self.d3 = _DecoderBlock(512, 128)             # /4   -> cat(e2) => 256
        self.d2 = _DecoderBlock(256, 64)              # /2   -> cat(e1) => 128
        self.d1 = _DecoderBlock(128, 4, apply_norm=False)  # /1  (no norm, raw output)

    def forward(
        self,
        agnostic: torch.Tensor,
        warped_garment: torch.Tensor,
        pose_heatmaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        agnostic       : (B, 3, H, W) person with garment region removed
        warped_garment : (B, 3, H, W) TPS-warped garment
        pose_heatmaps  : (B, 18, H, W) pose joint heatmaps

        Returns
        -------
        result : (B, 3, H, W) blended try-on image
        mask   : (B, 1, H, W) composition mask (after sigmoid)
        """
        x = torch.cat([agnostic, warped_garment, pose_heatmaps], dim=1)  # (B,24,H,W)

        # encode
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # decode with skip connections
        d5 = self.d5(e5)
        d5 = torch.cat([d5, e4], dim=1)   # 512 + 512 = 1024

        d4 = self.d4(d5)
        d4 = torch.cat([d4, e3], dim=1)   # 256 + 256 = 512

        d3 = self.d3(d4)
        d3 = torch.cat([d3, e2], dim=1)   # 128 + 128 = 256

        d2 = self.d2(d3)
        d2 = torch.cat([d2, e1], dim=1)   # 64 + 64 = 128

        out = self.d1(d2)                  # (B, 4, H, W)

        # split into render (3ch) and mask (1ch)
        render = torch.tanh(out[:, :3])
        mask = torch.sigmoid(out[:, 3:4])

        # composite: blend warped garment with rendered image
        result = mask * warped_garment + (1.0 - mask) * render

        return result, mask
