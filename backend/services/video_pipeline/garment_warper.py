"""Garment-masked optical flow warping.

// intent: warp keyframe garment pixels using flow field, composite onto current frame
// status: done
// confidence: high

Novel contribution: flow applied only within semantic garment mask, extending
the two-mask compositing approach to the temporal domain.

Occlusion-aware: forward-backward consistency check zeros out occluded regions,
falling back to original frame pixels at occlusion boundaries.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from .optical_flow import FlowResult, backward_warp, FLOW_H, FLOW_W

logger = logging.getLogger(__name__)


def warp_garment(
    keyframe_result: Image.Image,
    keyframe_mask: np.ndarray,
    current_frame: Image.Image,
    flow_result: FlowResult,
) -> tuple[Image.Image, np.ndarray]:
    """Warp keyframe's garment onto the current frame using optical flow.

    Steps:
    1. Resize keyframe result + mask to flow resolution
    2. Warp garment pixels + mask using backward flow
    3. Zero out occluded regions (fall back to original)
    4. Composite warped garment onto current frame
    5. Resize back to original resolution

    Args:
        keyframe_result: VTON result at the keyframe (full image).
        keyframe_mask: Float32 garment mask [0,1] at keyframe resolution.
        current_frame: Current video frame (original, no VTON).
        flow_result: Flow from keyframe to current frame.

    Returns:
        (composited_frame, warped_mask) — composited PIL Image and warped mask.
    """
    orig_w, orig_h = current_frame.size

    # Resize inputs to flow resolution
    kf_resized = keyframe_result.resize((FLOW_W, FLOW_H), Image.BILINEAR)
    cf_resized = current_frame.resize((FLOW_W, FLOW_H), Image.BILINEAR)
    mask_resized = cv2.resize(keyframe_mask, (FLOW_W, FLOW_H), interpolation=cv2.INTER_LINEAR)

    # Convert to tensors
    device = "cpu"
    kf_tensor = torch.from_numpy(
        np.array(kf_resized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)  # (3, H, W)
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)  # (1, H, W)
    flow_tensor = torch.from_numpy(flow_result.flow)  # (2, H, W)
    occlusion = torch.from_numpy(flow_result.occlusion_mask)  # (H, W)

    # Warp garment pixels and mask
    warped_garment = backward_warp(kf_tensor.to(device), flow_tensor.to(device))  # (3, H, W)
    warped_mask = backward_warp(mask_tensor.to(device), flow_tensor.to(device))  # (1, H, W)
    warped_mask = warped_mask.squeeze(0)  # (H, W)

    # Zero out occluded regions — garment disappears where flow is inconsistent
    visible = 1.0 - occlusion.to(device)
    warped_mask = warped_mask * visible

    # Threshold mask to avoid ghosting
    warped_mask = torch.clamp(warped_mask, 0.0, 1.0)

    # Feather edges (smooth mask boundaries)
    mask_np = warped_mask.cpu().numpy()
    mask_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=2.0)

    # Composite: warped garment inside mask, current frame outside
    cf_tensor = torch.from_numpy(
        np.array(cf_resized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)  # (3, H, W)

    mask_3ch = torch.from_numpy(mask_np).unsqueeze(0).expand(3, -1, -1)
    composited = cf_tensor * (1.0 - mask_3ch) + warped_garment * mask_3ch

    # Convert back to PIL at original resolution
    composited_np = (composited.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    composited_pil = Image.fromarray(composited_np).resize((orig_w, orig_h), Image.LANCZOS)

    # Resize mask back to original resolution
    mask_out = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return composited_pil, mask_out
