# intent: thin-plate-spline warping module — predicts control-point offsets
#         from person+garment features, solves the TPS system, and warps
#         the garment image via grid_sample
# status: done
# next: train jointly with composition U-Net; tune lambda regularisation
# confidence: high

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlPointNet(nn.Module):
    """CNN that regresses 2 x N_CTRL TPS offsets from concatenated feature maps."""

    def __init__(self, in_channels: int, n_control_points: int = 25):
        super().__init__()
        self.n_control_points = n_control_points

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 2 offsets (dx, dy) per control point
        self.fc = nn.Linear(64, n_control_points * 2)

        # initialise offsets near zero so warping starts as identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, N_CTRL, 2) offset tensor."""
        h = self.conv(x)
        h = self.pool(h).flatten(1)          # (B, 64)
        offsets = self.fc(h)                  # (B, 50)
        return offsets.view(-1, self.n_control_points, 2)


class TPSWarping(nn.Module):
    """Thin-Plate-Spline warping with learnable control-point offsets.

    Parameters
    ----------
    in_channels : int
        Number of channels in *each* feature map (person and garment are
        concatenated along the channel axis, so the ControlPointNet receives
        ``2 * in_channels``).
    n_control_points_per_side : int
        Grid side length; total control points = n^2.  Default 5 -> 25 pts.
    reg_lambda : float
        Tikhonov regularisation added to the TPS kernel diagonal.
    """

    def __init__(
        self,
        in_channels: int = 256,
        n_control_points_per_side: int = 5,
        reg_lambda: float = 0.0,
    ):
        super().__init__()
        self.n_side = n_control_points_per_side
        self.n_ctrl = n_control_points_per_side ** 2  # 25
        self.reg_lambda = reg_lambda

        self.control_point_net = ControlPointNet(
            in_channels=in_channels * 2,
            n_control_points=self.n_ctrl,
        )

        # canonical source grid in [-1, 1]
        grid_1d = torch.linspace(-1.0, 1.0, self.n_side)
        gy, gx = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        # (N_CTRL, 2)
        self.register_buffer(
            "source_points", torch.stack([gx.flatten(), gy.flatten()], dim=1)
        )

    # ------------------------------------------------------------------
    # TPS math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _U(r: torch.Tensor) -> torch.Tensor:
        """Radial basis U(r) = r^2 * log(r), with U(0)=0."""
        return torch.where(r > 0, r.pow(2) * r.log(), torch.zeros_like(r))

    def compute_tps_grid(
        self,
        control_points: torch.Tensor,
        target_points: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Solve TPS and return a sampling grid of shape (B, H, W, 2).

        Parameters
        ----------
        control_points : (B, N, 2)  source control points
        target_points  : (B, N, 2)  target (= source + offset)
        output_size    : (H, W)
        """
        B, N, _ = control_points.shape
        device = control_points.device
        dtype = control_points.dtype

        # --- build K matrix  (B, N, N) ---
        # pairwise distances between control points
        diff = control_points.unsqueeze(2) - control_points.unsqueeze(1)  # (B,N,N,2)
        dist = diff.norm(dim=-1)  # (B, N, N)
        K = self._U(dist)
        if self.reg_lambda > 0:
            K = K + self.reg_lambda * torch.eye(N, device=device, dtype=dtype).unsqueeze(0)

        # --- build P matrix  (B, N, 3) : [1, x, y] ---
        ones = torch.ones(B, N, 1, device=device, dtype=dtype)
        P = torch.cat([ones, control_points], dim=2)  # (B, N, 3)

        # --- assemble system matrix L  (B, N+3, N+3) ---
        zeros_33 = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        top = torch.cat([K, P], dim=2)                              # (B, N, N+3)
        bottom = torch.cat([P.transpose(1, 2), zeros_33], dim=2)    # (B, 3, N+3)
        L = torch.cat([top, bottom], dim=1)                         # (B, N+3, N+3)

        # --- RHS  (B, N+3, 2) ---
        zeros_32 = torch.zeros(B, 3, 2, device=device, dtype=dtype)
        rhs = torch.cat([target_points, zeros_32], dim=1)           # (B, N+3, 2)

        # --- solve for TPS weights  (B, N+3, 2) ---
        weights = torch.linalg.solve(L, rhs)  # (B, N+3, 2)

        # --- build output sampling grid ---
        H, W = output_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (HW, 2)
        grid_pts = grid_pts.unsqueeze(0).expand(B, -1, -1)                  # (B, HW, 2)

        # distances from every grid point to every control point
        diff_g = grid_pts.unsqueeze(2) - control_points.unsqueeze(1)  # (B, HW, N, 2)
        dist_g = diff_g.norm(dim=-1)  # (B, HW, N)
        K_g = self._U(dist_g)         # (B, HW, N)

        ones_g = torch.ones(B, H * W, 1, device=device, dtype=dtype)
        P_g = torch.cat([ones_g, grid_pts], dim=2)  # (B, HW, 3)

        # full basis: (B, HW, N+3)
        basis = torch.cat([K_g, P_g], dim=2)

        # mapped grid points: (B, HW, 2)
        mapped = torch.bmm(basis, weights)

        return mapped.view(B, H, W, 2)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        garment_features: torch.Tensor,
        person_features: torch.Tensor,
        garment_image: torch.Tensor,
    ) -> torch.Tensor:
        """Warp ``garment_image`` according to learned TPS transformation.

        Parameters
        ----------
        garment_features : (B, C, H', W')
        person_features  : (B, C, H', W')
        garment_image    : (B, 3, H, W)

        Returns
        -------
        warped_garment : (B, 3, H, W)
        """
        concat_feats = torch.cat([person_features, garment_features], dim=1)
        offsets = self.control_point_net(concat_feats)  # (B, N, 2)

        B = garment_image.size(0)
        source = self.source_points.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        target = source + offsets

        H, W = garment_image.shape[2], garment_image.shape[3]
        tps_grid = self.compute_tps_grid(source, target, (H, W))

        warped = F.grid_sample(
            garment_image, tps_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped
