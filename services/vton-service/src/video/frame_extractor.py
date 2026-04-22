"""Video frame extraction via cv2.VideoCapture.

// intent: extract frames from video as a generator (memory-efficient)
// status: done
// confidence: high

Yields (frame_idx, PIL.Image, timestamp_ms) tuples.
Supports FPS normalization to a target FPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Read video metadata without extracting frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            codec=codec,
        )
    finally:
        cap.release()


def extract_frames(
    video_path: str,
    target_fps: int = 15,
    max_duration_sec: float = 30.0,
) -> Generator[tuple[int, Image.Image, float], None, None]:
    """Extract frames from a video file as a generator.

    Normalizes to target_fps by skipping frames. Yields (frame_idx, PIL.Image, timestamp_ms).
    frame_idx is the output index (0, 1, 2, ...) at target_fps.

    Args:
        video_path: Path to input video file.
        target_fps: Output frames per second (skip frames to match).
        max_duration_sec: Stop after this many seconds.

    Yields:
        (output_frame_index, PIL.Image RGB, timestamp_ms)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0
            logger.warning("Could not detect FPS, assuming %.1f", source_fps)

        # Frame interval: how many source frames to skip per output frame
        frame_interval = max(1.0, source_fps / target_fps)
        max_source_frame = int(max_duration_sec * source_fps)

        output_idx = 0
        next_source_frame = 0.0
        source_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if source_frame_idx >= max_source_frame:
                break

            if source_frame_idx >= int(next_source_frame):
                # Convert BGR → RGB → PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp_ms = (source_frame_idx / source_fps) * 1000.0

                yield (output_idx, pil_img, timestamp_ms)

                output_idx += 1
                next_source_frame += frame_interval

            source_frame_idx += 1

        logger.info(
            "Extracted %d frames from %s (source: %.1f FPS → target: %d FPS)",
            output_idx, video_path, source_fps, target_fps,
        )
    finally:
        cap.release()
