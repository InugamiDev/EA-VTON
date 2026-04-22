# intent: trainable nn.Module wrapping feature encoders + TPS + composition for training
# status: done
# next: use from training/scripts/train_vton.py
# confidence: high

"""Trainable VTON module for curriculum training.

Wraps person/garment ResNet-18 encoders, TPS warping, and composition U-Net
into a single nn.Module with proper .parameters(), .train(), .state_dict().

Separate from the inference-only VTONModel wrapper in model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .tps import TPSWarping
from .composition import CompositionUNet


class TrainableVTON(nn.Module):
    """Full VTON pipeline as a trainable nn.Module.

    Components:
        person_encoder:  ResNet-18 truncated at layer3 → 256-ch feature map
        garment_encoder: ResNet-18 truncated at layer3 → 256-ch feature map
        tps_module:      TPS warping with learnable control points
        composition_net: U-Net fusing agnostic + warped garment + pose → result
    """

    def __init__(
        self,
        n_control_points_per_side: int = 5,
        composition_channels: int = 24,
        warp_reg_lambda: float = 0.0,
    ) -> None:
        super().__init__()

        self.person_encoder = self._build_encoder()
        self.garment_encoder = self._build_encoder()
        self.tps_module = TPSWarping(
            in_channels=256,
            n_control_points_per_side=n_control_points_per_side,
            reg_lambda=warp_reg_lambda,
        )
        self.composition_net = CompositionUNet(in_channels=composition_channels)

    @staticmethod
    def _build_encoder() -> nn.Sequential:
        """ResNet-18 truncated after layer3 → (B, 256, H/8, W/8)."""
        backbone = models.resnet18(weights=None)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

    def warp(
        self,
        person: torch.Tensor,
        garment: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: warp garment to match person pose.

        Args:
            person:  (B, 3, H, W) person image tensor
            garment: (B, 3, H, W) garment image tensor

        Returns:
            warped_garment: (B, 3, H, W) TPS-warped garment
        """
        f_person = self.person_encoder(person)
        f_garment = self.garment_encoder(garment)
        return self.tps_module(f_garment, f_person, garment)

    def forward(
        self,
        person: torch.Tensor,
        garment: torch.Tensor,
        heatmaps: torch.Tensor,
        parsing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline: warp + compose.

        Args:
            person:   (B, 3, H, W) person image tensor
            garment:  (B, 3, H, W) garment image tensor
            heatmaps: (B, 18, H, W) pose heatmaps
            parsing:  (B, 1, H, W) parsing map (optional, used for agnostic)

        Returns:
            result: (B, 3, H, W) final try-on image
            mask:   (B, 1, H, W) composition mask
        """
        warped = self.warp(person, garment)

        # Generate agnostic: zero out garment region if parsing available,
        # otherwise use person image directly (paired training assumption)
        if parsing is not None:
            agnostic = person.clone()
            # Zero upper-clothes region (ATR labels 4,5,6,7)
            garment_mask = (
                (parsing == 4) | (parsing == 5) |
                (parsing == 6) | (parsing == 7)
            ).float().unsqueeze(1)  # (B, 1, H, W)
            agnostic = agnostic * (1.0 - garment_mask)
        else:
            agnostic = person

        # Resize heatmaps to match spatial dims if needed
        H, W = person.shape[2], person.shape[3]
        if heatmaps.shape[2] != H or heatmaps.shape[3] != W:
            heatmaps = torch.nn.functional.interpolate(
                heatmaps, size=(H, W), mode="bilinear", align_corners=False,
            )

        result, mask = self.composition_net(agnostic, warped, heatmaps)
        return result, mask
