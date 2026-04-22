"""RAFT-Small optical flow estimation.

// intent: estimate dense optical flow between frames for garment warping
// status: done
// confidence: high

Uses torchvision's RAFT-Small (already installed). Operates at 512x384
for speed (~8-12 FPS on M2 CPU, ~21 FPS GPU).

Includes forward-backward consistency check for occlusion detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Singleton model
_model = None
_device = None

# Operating resolution (must be divisible by 8)
FLOW_H = 384
FLOW_W = 512


@dataclass
class FlowResult:
    """Optical flow estimation result."""
    flow: np.ndarray           # (2, H, W) flow in pixel units at FLOW_H x FLOW_W
    occlusion_mask: np.ndarray  # (H, W) float32 [0,1], 1 = occluded
    original_size: tuple[int, int]  # (W, H) of input frames


def _get_model(device: str = "cpu"):
    """Get or create singleton RAFT-Small model."""
    global _model, _device
    if _model is not None and _device == device:
        return _model

    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    logger.info("Loading RAFT-Small on %s...", device)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights)
    model = model.to(device).eval()
    _model = model
    _device = device
    logger.info("RAFT-Small ready on %s", device)
    return model


def _preprocess_frame(frame: Image.Image, device: str) -> torch.Tensor:
    """Convert PIL Image to RAFT input tensor.

    RAFT expects (N, 3, H, W) in [0, 255] uint8 or [-1, 1] float.
    We normalize to [-1, 1] at FLOW_H x FLOW_W.
    """
    frame_resized = frame.resize((FLOW_W, FLOW_H), Image.BILINEAR)
    arr = np.array(frame_resized, dtype=np.float32)  # (H, W, 3)
    # Normalize to [-1, 1]
    arr = (arr / 255.0) * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def estimate_flow(
    frame1: Image.Image,
    frame2: Image.Image,
    occlusion_threshold: float = 2.0,
    device: str = "cpu",
) -> FlowResult:
    """Estimate optical flow from frame1 to frame2.

    Args:
        frame1: Source frame (PIL Image RGB).
        frame2: Target frame (PIL Image RGB).
        occlusion_threshold: Pixel distance for forward-backward consistency.
        device: torch device.

    Returns:
        FlowResult with flow field and occlusion mask at FLOW_H x FLOW_W.
    """
    model = _get_model(device)
    original_size = frame1.size  # (W, H)

    img1 = _preprocess_frame(frame1, device)
    img2 = _preprocess_frame(frame2, device)

    with torch.no_grad():
        # RAFT returns list of flow predictions at different iterations
        # Last element is the final prediction
        flow_fwd_list = model(img1, img2)
        flow_fwd = flow_fwd_list[-1]  # (1, 2, H, W)

        flow_bwd_list = model(img2, img1)
        flow_bwd = flow_bwd_list[-1]  # (1, 2, H, W)

    # Occlusion detection via forward-backward consistency
    occlusion = _forward_backward_consistency(
        flow_fwd[0], flow_bwd[0], threshold=occlusion_threshold
    )

    flow_np = flow_fwd[0].cpu().numpy()  # (2, H, W)
    occlusion_np = occlusion.cpu().numpy()  # (H, W)

    return FlowResult(
        flow=flow_np,
        occlusion_mask=occlusion_np,
        original_size=original_size,
    )


def _forward_backward_consistency(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Check forward-backward consistency for occlusion detection.

    For each pixel x: check if ||flow_fwd(x) + flow_bwd(x + flow_fwd(x))|| > threshold.
    Returns binary mask where 1 = occluded.

    Args:
        flow_fwd: (2, H, W) forward flow.
        flow_bwd: (2, H, W) backward flow.
        threshold: Consistency threshold in pixels.

    Returns:
        (H, W) float32 tensor, 1.0 = occluded, 0.0 = visible.
    """
    _, h, w = flow_fwd.shape

    # Create coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow_fwd.device, dtype=torch.float32),
        torch.arange(w, device=flow_fwd.device, dtype=torch.float32),
        indexing="ij",
    )

    # Warp coordinates by forward flow
    warped_x = xx + flow_fwd[0]
    warped_y = yy + flow_fwd[1]

    # Normalize to [-1, 1] for grid_sample
    warped_x_norm = 2.0 * warped_x / (w - 1) - 1.0
    warped_y_norm = 2.0 * warped_y / (h - 1) - 1.0
    grid = torch.stack([warped_x_norm, warped_y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Sample backward flow at warped locations
    flow_bwd_4d = flow_bwd.unsqueeze(0)  # (1, 2, H, W)
    sampled_bwd = F.grid_sample(
        flow_bwd_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]  # (2, H, W)

    # Consistency check
    diff = flow_fwd + sampled_bwd  # Should be ~0 if consistent
    error = torch.norm(diff, dim=0)  # (H, W)

    return (error > threshold).float()


def backward_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """Warp image using backward flow via grid_sample.

    Args:
        image: (C, H, W) or (1, C, H, W) image tensor.
        flow: (2, H, W) flow field in pixel units.

    Returns:
        Warped image tensor, same shape as input.
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    _, _, h, w = image.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )

    # Add flow to create sampling coordinates
    sample_x = xx + flow[0]
    sample_y = yy + flow[1]

    # Normalize to [-1, 1]
    sample_x = 2.0 * sample_x / (w - 1) - 1.0
    sample_y = 2.0 * sample_y / (h - 1) - 1.0
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    warped = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    if squeeze:
        warped = warped.squeeze(0)

    return warped


def release():
    """Release the RAFT model to free memory."""
    global _model, _device
    if _model is not None:
        del _model
        _model = None
        _device = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("RAFT-Small released")
