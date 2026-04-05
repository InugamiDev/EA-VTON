"""RT-VTON Plan A: Keyframe + Optical Flow Video Virtual Try-On.

// intent: demo/proof-of-concept video VTON via keyframe inference + flow warping
// status: done
// next: Plan B (distilled real-time model) when GPU infra is available
// confidence: high

**This is Plan A — a demo pipeline for research evaluation and benchmarking.**
It processes video offline (~5-15 FPS). For production real-time use (30 FPS),
see Plan B in services/distillation/ and docs/09-video-tryon-pipeline.md.

Architecture:
    VideoInput → FrameExtractor → PoseTracker → KeyframeDetector
                                                  ↓
                                AsyncVTON (keyframes only, ~0.2 FPS)
                                                  ↓
                                OpticalFlowWarper (every frame, ~10 FPS)
                                                  ↓
                                GarmentCompositor → TemporalSmoother → VideoOutput

Core insight: VTON runs on ~1 in 10 frames (keyframes). For the other 9,
warp the garment using optical flow. Face/skin/background are always from
the original frame.

Plan A also serves as the teacher/data generator for Plan B distillation —
it produces unlimited labeled training pairs for the student model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from PIL import Image

from services.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/fitview_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_RESULTS_DIR = Path("/tmp/fitview_video_results")
VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoPipelineConfig:
    """Configuration for the video try-on pipeline."""
    vton_config: PipelineConfig = field(default_factory=PipelineConfig)
    target_fps: int = 15
    keyframe_pose_threshold: float = 0.05
    keyframe_min_interval: int = 5
    keyframe_max_interval: int = 30
    flow_model: str = "raft_small"
    blend_window: int = 3
    occlusion_threshold: float = 2.0
    max_duration_sec: float = 30.0
    output_format: str = "mp4"


@dataclass
class VideoPipelineStage:
    """Progress of a single pipeline stage."""
    name: str
    status: str = "pending"
    duration_ms: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class VideoPipelineResult:
    """Output of the video try-on pipeline."""
    video_path: str
    total_frames: int
    keyframe_count: int
    processing_time_ms: int
    avg_fps: float
    stages: list[VideoPipelineStage] = field(default_factory=list)


StageStatus = Literal[
    "extract_frames", "pose_tracking", "keyframe_detection",
    "keyframe_vton", "flow_warping", "temporal_smoothing",
    "video_assembly", "completed", "failed",
]


async def run_video_pipeline(
    video_path: str,
    garment_image: Image.Image,
    config: VideoPipelineConfig | None = None,
    progress_callback=None,
) -> VideoPipelineResult:
    """Execute the full video try-on pipeline (offline mode).

    Processes all keyframes first, then warps all inter-frames.

    Args:
        video_path: Path to input video file.
        garment_image: Garment image to try on (PIL Image).
        config: Pipeline configuration.
        progress_callback: Optional async callback(stage, progress_pct).

    Returns:
        VideoPipelineResult with path to output video.
    """
    if config is None:
        config = VideoPipelineConfig()

    pipeline_start = time.time()
    stages: list[VideoPipelineStage] = []

    async def _report(stage_name: str, pct: float = 0.0):
        if progress_callback:
            await progress_callback(stage_name, pct)

    # ── Stage 1: Extract frames ─────────────────────────────────────
    from .frame_extractor import extract_frames, get_video_metadata

    stage = VideoPipelineStage(name="extract_frames", status="running")
    stage_start = time.time()
    await _report("extract_frames", 0.0)

    metadata = get_video_metadata(video_path)
    frames: list[tuple[int, Image.Image, float]] = []

    for frame_idx, frame, ts in extract_frames(
        video_path,
        target_fps=config.target_fps,
        max_duration_sec=config.max_duration_sec,
    ):
        frames.append((frame_idx, frame, ts))

    total_frames = len(frames)
    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {
        "total_frames": total_frames,
        "source_fps": metadata.fps,
        "target_fps": config.target_fps,
        "duration_sec": round(metadata.duration_sec, 2),
    }
    stages.append(stage)
    logger.info("Extracted %d frames from %s", total_frames, video_path)

    if total_frames == 0:
        raise ValueError("No frames extracted from video")

    # ── Stage 2: Pose tracking ──────────────────────────────────────
    from .pose_tracker import detect_pose, PoseResult

    stage = VideoPipelineStage(name="pose_tracking", status="running")
    stage_start = time.time()
    await _report("pose_tracking", 0.0)

    poses: list[PoseResult] = []
    for i, (_, frame, _) in enumerate(frames):
        pose = await asyncio.to_thread(detect_pose, frame)
        poses.append(pose)

    detected_count = sum(1 for p in poses if p.detected)
    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {
        "poses_detected": detected_count,
        "detection_rate": round(detected_count / total_frames, 3),
    }
    stages.append(stage)

    # ── Stage 3: Keyframe detection ─────────────────────────────────
    from .keyframe_detector import KeyframeDetector

    stage = VideoPipelineStage(name="keyframe_detection", status="running")
    stage_start = time.time()
    await _report("keyframe_detection", 0.0)

    detector = KeyframeDetector(
        pose_threshold=config.keyframe_pose_threshold,
        min_interval=config.keyframe_min_interval,
        max_interval=config.keyframe_max_interval,
    )

    keyframe_indices: list[int] = []
    for i, pose in enumerate(poses):
        decision = detector.check(i, pose)
        if decision.is_keyframe:
            keyframe_indices.append(i)

    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = detector.stats
    stages.append(stage)
    logger.info(
        "Detected %d keyframes out of %d frames (ratio: %.2f)",
        len(keyframe_indices), total_frames,
        len(keyframe_indices) / total_frames if total_frames > 0 else 0,
    )

    # ── Stage 4: Keyframe VTON ──────────────────────────────────────
    from .async_vton_worker import process_keyframes_batch

    stage = VideoPipelineStage(name="keyframe_vton", status="running")
    stage_start = time.time()
    await _report("keyframe_vton", 0.0)

    keyframe_frames = [(idx, frames[idx][1]) for idx in keyframe_indices]
    vton_results = await process_keyframes_batch(
        keyframe_frames=keyframe_frames,
        garment_image=garment_image,
        config=config.vton_config,
    )

    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {
        "keyframes_processed": len(vton_results),
        "avg_vton_ms": (
            int(np.mean([r.processing_time_ms for r in vton_results.values()]))
            if vton_results else 0
        ),
    }
    stages.append(stage)

    # ── Stage 5: Flow warping for inter-frames ──────────────────────
    from .optical_flow import estimate_flow
    from .garment_warper import warp_garment

    stage = VideoPipelineStage(name="flow_warping", status="running")
    stage_start = time.time()
    await _report("flow_warping", 0.0)

    # Build output frames array
    output_frames: list[Image.Image] = [None] * total_frames
    output_masks: list[np.ndarray | None] = [None] * total_frames

    # Place keyframe results directly
    for kf_idx, kf_result in vton_results.items():
        output_frames[kf_idx] = kf_result.result_image
        output_masks[kf_idx] = kf_result.garment_mask

    # Determine device for flow
    import torch
    flow_device = "cuda" if torch.cuda.is_available() else "cpu"

    # For each inter-frame, find nearest preceding keyframe and warp
    sorted_kf = sorted(keyframe_indices)
    warped_count = 0

    for frame_idx in range(total_frames):
        if output_frames[frame_idx] is not None:
            continue  # Already a keyframe

        # Find the nearest preceding keyframe
        prev_kf_idx = None
        for kf in reversed(sorted_kf):
            if kf < frame_idx:
                prev_kf_idx = kf
                break

        if prev_kf_idx is None:
            # Before first keyframe — use original frame
            output_frames[frame_idx] = frames[frame_idx][1]
            continue

        kf_result = vton_results[prev_kf_idx]

        # Estimate flow from keyframe to current frame
        flow_result = await asyncio.to_thread(
            estimate_flow,
            frames[prev_kf_idx][1],  # keyframe
            frames[frame_idx][1],     # current frame
            config.occlusion_threshold,
            flow_device,
        )

        # Warp garment from keyframe onto current frame
        warped_frame, warped_mask = await asyncio.to_thread(
            warp_garment,
            kf_result.result_image,
            kf_result.garment_mask,
            frames[frame_idx][1],
            flow_result,
        )

        output_frames[frame_idx] = warped_frame
        output_masks[frame_idx] = warped_mask
        warped_count += 1

    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {"warped_frames": warped_count}
    stages.append(stage)

    # ── Stage 6: Temporal smoothing ─────────────────────────────────
    from .temporal_smoother import TemporalSmoother

    stage = VideoPipelineStage(name="temporal_smoothing", status="running")
    stage_start = time.time()
    await _report("temporal_smoothing", 0.0)

    smoother = TemporalSmoother(blend_window=config.blend_window)
    blended_count = 0

    # Register keyframes and apply blending
    for kf_idx in sorted_kf:
        smoother.update_keyframe(kf_idx)

    # Re-pass: blend frames near keyframe boundaries
    for i in range(total_frames):
        if i in vton_results:
            continue  # Don't blend keyframes themselves

        if smoother.should_blend(i):
            # Find adjacent keyframes
            next_kf = None
            prev_kf = None
            for kf in sorted_kf:
                if kf <= i:
                    prev_kf = kf
                elif next_kf is None:
                    next_kf = kf

            if prev_kf is not None and next_kf is not None and next_kf in vton_results:
                alpha = smoother.get_blend_alpha(i)
                prev_mask = output_masks[i] if output_masks[i] is not None else np.zeros((1, 1))
                next_mask = vton_results[next_kf].garment_mask

                from .temporal_smoother import blend_keyframe_transition
                blended, blended_mask = blend_keyframe_transition(
                    output_frames[i], prev_mask,
                    vton_results[next_kf].result_image, next_mask,
                    alpha,
                )
                output_frames[i] = blended
                output_masks[i] = blended_mask
                blended_count += 1

    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {"blended_frames": blended_count}
    stages.append(stage)

    # ── Stage 7: Video assembly ─────────────────────────────────────
    from .video_assembler import assemble_video, mux_audio

    stage = VideoPipelineStage(name="video_assembly", status="running")
    stage_start = time.time()
    await _report("video_assembly", 0.0)

    output_id = uuid4().hex[:12]
    temp_video_path = str(VIDEO_RESULTS_DIR / f"{output_id}_noaudio.mp4")
    final_video_path = str(VIDEO_RESULTS_DIR / f"{output_id}.mp4")

    # Filter out any None frames (shouldn't happen, but defensive)
    valid_frames = [f for f in output_frames if f is not None]

    await asyncio.to_thread(
        assemble_video,
        valid_frames,
        temp_video_path,
        config.target_fps,
    )

    # Mux audio from source
    final_path = await asyncio.to_thread(
        mux_audio,
        temp_video_path,
        video_path,
        final_video_path,
    )

    stage.status = "completed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stage.details = {"output_path": final_path, "total_output_frames": len(valid_frames)}
    stages.append(stage)

    # ── Build result ────────────────────────────────────────────────
    total_ms = int((time.time() - pipeline_start) * 1000)
    avg_fps = (total_frames / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    result = VideoPipelineResult(
        video_path=final_path,
        total_frames=total_frames,
        keyframe_count=len(keyframe_indices),
        processing_time_ms=total_ms,
        avg_fps=round(avg_fps, 2),
        stages=stages,
    )

    logger.info(
        "Video pipeline complete: %d frames, %d keyframes, %dms, %.2f effective FPS",
        total_frames, len(keyframe_indices), total_ms, avg_fps,
    )

    # Clean up resources
    from .pose_tracker import release as release_pose
    from .optical_flow import release as release_flow
    release_pose()
    release_flow()

    return result
