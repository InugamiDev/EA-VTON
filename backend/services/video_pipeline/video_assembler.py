"""Video assembly from processed frames.

// intent: assemble processed frames into MP4 output
// status: done
// confidence: high

Uses cv2.VideoWriter with H.264 codec. Audio preservation via ffmpeg mux.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def assemble_video(
    frames: list[Image.Image],
    output_path: str,
    fps: int = 15,
    codec: str = "mp4v",
) -> str:
    """Assemble frames into an MP4 video.

    Args:
        frames: List of PIL Images in order.
        output_path: Path for output MP4 file.
        fps: Output frame rate.
        codec: FourCC codec code.

    Returns:
        Path to the output video file.
    """
    if not frames:
        raise ValueError("No frames to assemble")

    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer at {output_path}")

    try:
        for frame in frames:
            if frame.size != (w, h):
                frame = frame.resize((w, h), Image.LANCZOS)
            arr = np.array(frame)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        logger.info(
            "Assembled %d frames → %s (%dx%d @ %d FPS)",
            len(frames), output_path, w, h, fps,
        )
    finally:
        writer.release()

    return output_path


def mux_audio(
    video_path: str,
    source_video_path: str,
    output_path: str,
) -> str:
    """Copy audio from source video onto the processed video using ffmpeg.

    If ffmpeg is not available or source has no audio, returns video_path unchanged.

    Args:
        video_path: Processed video (no audio).
        source_video_path: Original video with audio.
        output_path: Final output path with audio.

    Returns:
        Path to the final video (with audio if available).
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg not available, skipping audio mux")
            return video_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg not found, skipping audio mux")
        return video_path

    try:
        # Mux: copy video stream + audio stream from source
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", source_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path,
            ],
            capture_output=True, timeout=60,
            check=True,
        )
        # Remove the intermediate video without audio
        os.unlink(video_path)
        logger.info("Audio muxed: %s", output_path)
        return output_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Audio mux failed: %s, using video without audio", e)
        return video_path
