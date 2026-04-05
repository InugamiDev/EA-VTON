"""Temporal smoothing near keyframe transitions.

// intent: blend outgoing and incoming keyframe VTON results to avoid popping
// status: done
// confidence: high

Near keyframe boundaries (within N frames), blends the outgoing warped result
with the incoming keyframe result using linear interpolation.
Only blends within the garment mask — non-garment pixels are always from
the original frame.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def blend_keyframe_transition(
    outgoing_frame: Image.Image,
    outgoing_mask: np.ndarray,
    incoming_frame: Image.Image,
    incoming_mask: np.ndarray,
    alpha: float,
) -> tuple[Image.Image, np.ndarray]:
    """Blend two VTON results near a keyframe boundary.

    Args:
        outgoing_frame: Warped result from the previous keyframe.
        outgoing_mask: Garment mask for outgoing (float32 [0,1]).
        incoming_frame: Result from the new keyframe (or its warp).
        incoming_mask: Garment mask for incoming (float32 [0,1]).
        alpha: Blend weight for incoming (0.0 = all outgoing, 1.0 = all incoming).

    Returns:
        (blended_frame, blended_mask)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Ensure same size
    if outgoing_frame.size != incoming_frame.size:
        incoming_frame = incoming_frame.resize(outgoing_frame.size, Image.LANCZOS)
    if outgoing_mask.shape != incoming_mask.shape:
        incoming_mask = cv2.resize(
            incoming_mask,
            (outgoing_mask.shape[1], outgoing_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    out_arr = np.array(outgoing_frame, dtype=np.float32)
    in_arr = np.array(incoming_frame, dtype=np.float32)

    # Blend pixels
    blended = out_arr * (1.0 - alpha) + in_arr * alpha
    blended_mask = outgoing_mask * (1.0 - alpha) + incoming_mask * alpha

    return Image.fromarray(blended.clip(0, 255).astype(np.uint8)), blended_mask


class TemporalSmoother:
    """Stateful temporal smoother that manages keyframe transitions.

    Tracks the current and previous keyframe results, applies blending
    within a configurable window around each keyframe boundary.
    """

    def __init__(self, blend_window: int = 3):
        """
        Args:
            blend_window: Number of frames over which to blend transitions.
        """
        self.blend_window = blend_window
        self._prev_keyframe_idx: int | None = None
        self._curr_keyframe_idx: int | None = None

    def update_keyframe(self, keyframe_idx: int) -> None:
        """Register a new keyframe."""
        self._prev_keyframe_idx = self._curr_keyframe_idx
        self._curr_keyframe_idx = keyframe_idx

    def should_blend(self, frame_idx: int) -> bool:
        """Check if a frame falls within a blend window."""
        if self._curr_keyframe_idx is None or self._prev_keyframe_idx is None:
            return False
        distance = frame_idx - self._curr_keyframe_idx
        return 0 < distance <= self.blend_window

    def get_blend_alpha(self, frame_idx: int) -> float:
        """Get the blend alpha for a frame within the transition window.

        Returns alpha in [0, 1] where 1.0 = fully incoming (new keyframe).
        """
        if self._curr_keyframe_idx is None:
            return 1.0
        distance = frame_idx - self._curr_keyframe_idx
        if distance <= 0:
            return 1.0
        if distance > self.blend_window:
            return 1.0
        # Linear ramp from 0 → 1 over blend_window frames
        return distance / self.blend_window

    def smooth_frame(
        self,
        frame_idx: int,
        current_result: Image.Image,
        current_mask: np.ndarray,
        prev_result: Image.Image | None = None,
        prev_mask: np.ndarray | None = None,
    ) -> tuple[Image.Image, np.ndarray]:
        """Apply temporal smoothing to a frame if within blend window.

        If not in a blend window or no previous result, returns current as-is.
        """
        if not self.should_blend(frame_idx) or prev_result is None or prev_mask is None:
            return current_result, current_mask

        alpha = self.get_blend_alpha(frame_idx)
        return blend_keyframe_transition(
            outgoing_frame=prev_result,
            outgoing_mask=prev_mask,
            incoming_frame=current_result,
            incoming_mask=current_mask,
            alpha=alpha,
        )
