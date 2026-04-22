"""Pose-aware keyframe detection for VTON.

// intent: trigger VTON inference only on frames with significant pose change
// status: done
// confidence: high

Novel contribution: uses garment-relevant landmark subset (shoulders, elbows,
hips) instead of generic scene change detection. Also triggers on torso
rotation (shoulder vector change).

Rules:
1. Frame 0 = always keyframe
2. Pose lost (no detection) = force keyframe
3. pose_delta > threshold = keyframe
4. torso_rotation_delta > rotation_threshold = keyframe
5. frames_since_last > max_interval = force keyframe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .pose_tracker import PoseResult, compute_pose_delta, compute_torso_rotation

logger = logging.getLogger(__name__)


@dataclass
class KeyframeDecision:
    """Result of keyframe detection for a single frame."""
    is_keyframe: bool
    reason: str  # "first_frame", "pose_delta", "torso_rotation", "max_interval", "pose_lost", "not_keyframe"
    pose_delta: float = 0.0
    rotation_delta: float = 0.0
    frames_since_last: int = 0


@dataclass
class KeyframeDetector:
    """Stateful keyframe detector tracking pose changes between frames.

    Args:
        pose_threshold: Minimum pose delta to trigger a keyframe.
        rotation_threshold: Minimum torso rotation delta (radians) to trigger.
        min_interval: Minimum frames between keyframes (cooldown).
        max_interval: Force keyframe after this many frames.
    """
    pose_threshold: float = 0.05
    rotation_threshold: float = 0.15  # ~8.6 degrees
    min_interval: int = 5
    max_interval: int = 30

    # Internal state
    _last_keyframe_idx: int = field(default=-1, init=False, repr=False)
    _last_keyframe_garment_lm: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_keyframe_rotation: float = field(default=0.0, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _keyframe_count: int = field(default=0, init=False, repr=False)

    def check(self, frame_idx: int, pose: PoseResult) -> KeyframeDecision:
        """Check if a frame should be a keyframe.

        Args:
            frame_idx: Current frame index.
            pose: PoseResult from pose_tracker.

        Returns:
            KeyframeDecision indicating if this frame is a keyframe and why.
        """
        self._frame_count += 1
        frames_since = frame_idx - self._last_keyframe_idx

        # Rule 1: First frame is always a keyframe
        if self._last_keyframe_idx < 0:
            self._mark_keyframe(frame_idx, pose)
            return KeyframeDecision(
                is_keyframe=True, reason="first_frame",
                frames_since_last=frames_since,
            )

        # Rule 2: Pose lost → force keyframe
        if not pose.detected:
            self._mark_keyframe(frame_idx, pose)
            return KeyframeDecision(
                is_keyframe=True, reason="pose_lost",
                frames_since_last=frames_since,
            )

        # Cooldown: skip if too close to last keyframe
        if frames_since < self.min_interval:
            return KeyframeDecision(
                is_keyframe=False, reason="not_keyframe",
                frames_since_last=frames_since,
            )

        # Rule 5: Max interval exceeded → force keyframe
        if frames_since >= self.max_interval:
            delta = self._compute_delta(pose)
            rot_delta = self._compute_rotation_delta(pose)
            self._mark_keyframe(frame_idx, pose)
            return KeyframeDecision(
                is_keyframe=True, reason="max_interval",
                pose_delta=delta, rotation_delta=rot_delta,
                frames_since_last=frames_since,
            )

        # Rule 3: Pose delta exceeds threshold
        delta = self._compute_delta(pose)
        rot_delta = self._compute_rotation_delta(pose)

        if delta > self.pose_threshold:
            self._mark_keyframe(frame_idx, pose)
            return KeyframeDecision(
                is_keyframe=True, reason="pose_delta",
                pose_delta=delta, rotation_delta=rot_delta,
                frames_since_last=frames_since,
            )

        # Rule 4: Torso rotation exceeds threshold
        if abs(rot_delta) > self.rotation_threshold:
            self._mark_keyframe(frame_idx, pose)
            return KeyframeDecision(
                is_keyframe=True, reason="torso_rotation",
                pose_delta=delta, rotation_delta=rot_delta,
                frames_since_last=frames_since,
            )

        return KeyframeDecision(
            is_keyframe=False, reason="not_keyframe",
            pose_delta=delta, rotation_delta=rot_delta,
            frames_since_last=frames_since,
        )

    def _compute_delta(self, pose: PoseResult) -> float:
        """Compute pose delta from last keyframe."""
        if self._last_keyframe_garment_lm is None or pose.garment_landmarks is None:
            return float("inf")
        return compute_pose_delta(self._last_keyframe_garment_lm, pose.garment_landmarks)

    def _compute_rotation_delta(self, pose: PoseResult) -> float:
        """Compute torso rotation delta from last keyframe."""
        if pose.garment_landmarks is None:
            return 0.0
        current_rot = compute_torso_rotation(pose.garment_landmarks)
        return current_rot - self._last_keyframe_rotation

    def _mark_keyframe(self, frame_idx: int, pose: PoseResult) -> None:
        """Update state to mark current frame as keyframe."""
        self._last_keyframe_idx = frame_idx
        self._keyframe_count += 1
        if pose.garment_landmarks is not None:
            self._last_keyframe_garment_lm = pose.garment_landmarks.copy()
            self._last_keyframe_rotation = compute_torso_rotation(pose.garment_landmarks)

    @property
    def stats(self) -> dict:
        """Return detection statistics."""
        return {
            "total_frames": self._frame_count,
            "keyframes": self._keyframe_count,
            "keyframe_ratio": (
                round(self._keyframe_count / self._frame_count, 3)
                if self._frame_count > 0 else 0.0
            ),
        }
