"""Real end-to-end video pipeline test with actual VTON inference.

// intent: produce real try-on video from person photo + garment image
// status: done
// confidence: high

Uses FASHN VTON 1.5 example data (model.webp + garment.webp).
Creates a synthetic video from the model photo by applying slight
camera-like transforms (crop shifts, minor zoom), then runs the
full pipeline with real VTON inference on keyframes.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.chdir(Path(__file__).resolve().parent.parent)

OUTPUT_DIR = Path("/tmp/fitview_real_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_DIR = Path("/private/tmp/fashn-vton-1.5/examples/data")


def create_video_from_photo(
    photo: Image.Image,
    output_path: str,
    num_frames: int = 30,
    fps: int = 15,
    target_w: int = 576,
    target_h: int = 864,
) -> str:
    """Create a video from a single photo by simulating camera movement.

    Applies a gentle horizontal pan + slight zoom breathing to create
    realistic frame-to-frame variation from a static image.
    """
    # Work at slightly larger resolution for crop headroom
    src_w, src_h = photo.size
    # We'll crop a target_w x target_h window that pans across the image
    pad = 40  # max pixel shift in each direction

    # Ensure source is large enough
    if src_w < target_w + 2 * pad or src_h < target_h + 2 * pad:
        # Upscale if needed
        scale = max((target_w + 2 * pad) / src_w, (target_h + 2 * pad) / src_h)
        photo = photo.resize(
            (int(src_w * scale) + 1, int(src_h * scale) + 1), Image.LANCZOS
        )
        src_w, src_h = photo.size

    # Center crop origin
    cx = (src_w - target_w) // 2
    cy = (src_h - target_h) // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    photo_arr = np.array(photo)

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)

        # Horizontal pan: gentle sinusoidal sway
        dx = int(pad * 0.7 * np.sin(2 * np.pi * t))
        # Vertical: slight breathing
        dy = int(pad * 0.3 * np.sin(4 * np.pi * t))

        x1 = cx + dx
        y1 = cy + dy
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Clamp
        x1 = max(0, min(x1, src_w - target_w))
        y1 = max(0, min(y1, src_h - target_h))
        x2 = x1 + target_w
        y2 = y1 + target_h

        crop = photo_arr[y1:y2, x1:x2]
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Created video from photo: {output_path} ({num_frames} frames, {fps} FPS, {target_w}x{target_h})")
    return output_path


async def run_real_test():
    """Run full pipeline with real VTON on actual person + garment images."""
    print("=" * 60)
    print("RT-VTON Real End-to-End Test")
    print("=" * 60)

    # Load real images
    model_photo = Image.open(EXAMPLE_DIR / "model.webp").convert("RGB")
    garment_image = Image.open(EXAMPLE_DIR / "garment.webp").convert("RGB")
    print(f"Model photo: {model_photo.size}")
    print(f"Garment image: {garment_image.size}")

    # Save copies for reference
    model_photo.save(str(OUTPUT_DIR / "input_model.jpg"), quality=95)
    garment_image.save(str(OUTPUT_DIR / "input_garment.jpg"), quality=95)

    # Create video from photo (15 frames = 1 second at 15 FPS)
    video_path = str(OUTPUT_DIR / "input_video.mp4")
    create_video_from_photo(model_photo, video_path, num_frames=15, fps=15)

    stages_report = []
    pipeline_start = time.time()

    # ── Stage 1: Frame extraction ───────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.frame_extractor import extract_frames, get_video_metadata

    metadata = get_video_metadata(video_path)
    frames = list(extract_frames(video_path, target_fps=15, max_duration_sec=10))
    total_frames = len(frames)
    dt = time.time() - t0
    stages_report.append(("frame_extraction", dt, {"frames": total_frames}))
    print(f"\n[1/7] Frame extraction: {total_frames} frames in {dt*1000:.0f}ms")

    # ── Stage 2: Pose tracking ──────────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.pose_tracker import detect_pose, PoseResult

    poses: list[PoseResult] = []
    for _, frame, _ in frames:
        pose = detect_pose(frame)
        poses.append(pose)
    detected = sum(1 for p in poses if p.detected)
    dt = time.time() - t0
    stages_report.append(("pose_tracking", dt, {
        "detected": detected, "total": total_frames,
        "fps": f"{total_frames/dt:.1f}",
    }))
    print(f"[2/7] Pose tracking: {detected}/{total_frames} detected in {dt*1000:.0f}ms ({total_frames/dt:.1f} FPS)")

    # ── Stage 3: Keyframe detection ─────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.keyframe_detector import KeyframeDetector

    detector = KeyframeDetector(
        pose_threshold=0.03,
        min_interval=7,       # Fewer keyframes for speed
        max_interval=15,
    )
    keyframe_indices = []
    for i, pose in enumerate(poses):
        decision = detector.check(i, pose)
        if decision.is_keyframe:
            keyframe_indices.append(i)

    dt = time.time() - t0
    stages_report.append(("keyframe_detection", dt, {
        "keyframes": len(keyframe_indices), "indices": keyframe_indices,
    }))
    print(f"[3/7] Keyframe detection: {len(keyframe_indices)} keyframes {keyframe_indices}")

    # ── Stage 4: Real VTON on keyframes ─────────────────────────────
    t0 = time.time()
    from services.video_pipeline.async_vton_worker import (
        KeyframeVTONResult, run_vton_on_keyframe,
    )
    from services.pipeline import PipelineConfig

    vton_config = PipelineConfig(
        backend="huggingface",
        denoise_steps=4,  # Minimal steps for speed (quality tradeoff)
        seed=42,
        garment_type="upper_body",
    )

    vton_results: dict[int, KeyframeVTONResult] = {}
    for i, kf_idx in enumerate(keyframe_indices):
        print(f"     VTON keyframe {i+1}/{len(keyframe_indices)} (frame {kf_idx})...", end=" ", flush=True)
        kf_start = time.time()
        result = await run_vton_on_keyframe(
            person_frame=frames[kf_idx][1],
            garment_image=garment_image,
            frame_idx=kf_idx,
            config=vton_config,
        )
        vton_results[kf_idx] = result
        print(f"{result.processing_time_ms}ms")

        # Save keyframe results for inspection
        result.result_image.save(str(OUTPUT_DIR / f"keyframe_{kf_idx:03d}.jpg"), quality=95)

    dt = time.time() - t0
    avg_ms = int(np.mean([r.processing_time_ms for r in vton_results.values()])) if vton_results else 0
    stages_report.append(("keyframe_vton", dt, {
        "processed": len(vton_results), "avg_ms": avg_ms,
    }))
    print(f"[4/7] Keyframe VTON: {len(vton_results)} keyframes in {dt*1000:.0f}ms (avg {avg_ms}ms)")

    # ── Stage 5: Flow warping ───────────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.optical_flow import estimate_flow
    from services.video_pipeline.garment_warper import warp_garment

    output_frames: list[Image.Image | None] = [None] * total_frames
    output_masks: list[np.ndarray | None] = [None] * total_frames

    for kf_idx, kf_result in vton_results.items():
        output_frames[kf_idx] = kf_result.result_image
        output_masks[kf_idx] = kf_result.garment_mask

    import torch
    flow_device = "cuda" if torch.cuda.is_available() else "cpu"

    sorted_kf = sorted(keyframe_indices)
    warped_count = 0
    flow_times = []

    for frame_idx in range(total_frames):
        if output_frames[frame_idx] is not None:
            continue

        prev_kf_idx = None
        for kf in reversed(sorted_kf):
            if kf < frame_idx:
                prev_kf_idx = kf
                break

        if prev_kf_idx is None:
            output_frames[frame_idx] = frames[frame_idx][1]
            continue

        kf_result = vton_results[prev_kf_idx]

        ft0 = time.time()
        flow_result = estimate_flow(
            frames[prev_kf_idx][1], frames[frame_idx][1],
            occlusion_threshold=2.0, device=flow_device,
        )
        flow_times.append(time.time() - ft0)

        warped_frame, warped_mask = warp_garment(
            kf_result.result_image, kf_result.garment_mask,
            frames[frame_idx][1], flow_result,
        )
        output_frames[frame_idx] = warped_frame
        output_masks[frame_idx] = warped_mask
        warped_count += 1

        # Save a few warped frames for inspection
        if warped_count <= 3:
            warped_frame.save(str(OUTPUT_DIR / f"warped_{frame_idx:03d}.jpg"), quality=95)

    dt = time.time() - t0
    avg_flow_ms = np.mean(flow_times) * 1000 if flow_times else 0
    flow_fps = 1000 / avg_flow_ms if avg_flow_ms > 0 else 0
    stages_report.append(("flow_warping", dt, {
        "warped": warped_count, "avg_flow_ms": f"{avg_flow_ms:.0f}",
        "flow_fps": f"{flow_fps:.1f}",
    }))
    print(f"[5/7] Flow warping: {warped_count} frames in {dt*1000:.0f}ms (flow: {flow_fps:.1f} FPS)")

    # ── Stage 6: Temporal smoothing ─────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.temporal_smoother import TemporalSmoother, blend_keyframe_transition

    smoother = TemporalSmoother(blend_window=2)
    blended_count = 0

    for kf_idx in sorted_kf:
        smoother.update_keyframe(kf_idx)

    for i in range(total_frames):
        if i in vton_results:
            continue
        if smoother.should_blend(i):
            next_kf = None
            for kf in sorted_kf:
                if kf > i:
                    next_kf = kf
                    break
            if next_kf and next_kf in vton_results and output_masks[i] is not None:
                alpha = smoother.get_blend_alpha(i)
                blended, blended_mask = blend_keyframe_transition(
                    output_frames[i], output_masks[i],
                    vton_results[next_kf].result_image,
                    vton_results[next_kf].garment_mask,
                    alpha,
                )
                output_frames[i] = blended
                output_masks[i] = blended_mask
                blended_count += 1

    dt = time.time() - t0
    stages_report.append(("temporal_smoothing", dt, {"blended": blended_count}))
    print(f"[6/7] Temporal smoothing: {blended_count} blended in {dt*1000:.0f}ms")

    # ── Stage 7: Video assembly ─────────────────────────────────────
    t0 = time.time()
    from services.video_pipeline.video_assembler import assemble_video

    valid_frames = [f for f in output_frames if f is not None]
    output_video = str(OUTPUT_DIR / "output_tryon.mp4")
    assemble_video(valid_frames, output_video, fps=15)

    # Also create a side-by-side comparison video
    comparison_frames = []
    for i, f in enumerate(valid_frames):
        original = frames[i][1] if i < len(frames) else valid_frames[0]
        # Resize both to same height
        h = min(f.size[1], original.size[1])
        w_result = int(f.size[0] * h / f.size[1])
        w_orig = int(original.size[0] * h / original.size[1])
        f_resized = f.resize((w_result, h), Image.LANCZOS)
        o_resized = original.resize((w_orig, h), Image.LANCZOS)
        # Side by side
        combined = Image.new("RGB", (w_orig + w_result + 4, h), (40, 40, 40))
        combined.paste(o_resized, (0, 0))
        combined.paste(f_resized, (w_orig + 4, 0))
        comparison_frames.append(combined)

    comparison_video = str(OUTPUT_DIR / "comparison.mp4")
    assemble_video(comparison_frames, comparison_video, fps=15)

    dt = time.time() - t0
    stages_report.append(("video_assembly", dt, {"frames": len(valid_frames)}))
    print(f"[7/7] Video assembly: {len(valid_frames)} frames in {dt*1000:.0f}ms")

    # ── Quality metrics ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    from skimage.metrics import structural_similarity as ssim

    ssim_scores = []
    for i in range(1, len(valid_frames)):
        arr1 = np.array(valid_frames[i - 1])
        arr2 = np.array(valid_frames[i])
        if arr1.shape == arr2.shape:
            score = ssim(arr1, arr2, channel_axis=2)
            ssim_scores.append(score)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    min_ssim = np.min(ssim_scores) if ssim_scores else 0
    print(f"Temporal SSIM: avg={avg_ssim:.4f}, min={min_ssim:.4f} (target: >0.85)")

    coverage = [float(np.mean(m > 0.5)) for m in output_masks if m is not None]
    print(f"Garment mask coverage: avg={np.mean(coverage)*100:.1f}%")

    # ── Summary ─────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for name, dt, details in stages_report:
        pct = dt / total_time * 100
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"  {name:25s} {dt*1000:8.0f}ms ({pct:4.1f}%)  {detail_str}")
    print(f"  {'TOTAL':25s} {total_time*1000:8.0f}ms")
    print(f"  Effective throughput: {total_frames / total_time:.2f} FPS")

    print(f"\n OUTPUT FILES: {OUTPUT_DIR}/")
    print(f"  input_model.jpg       — original person photo")
    print(f"  input_garment.jpg     — garment to try on")
    print(f"  input_video.mp4       — synthetic input video (pan from photo)")
    print(f"  keyframe_NNN.jpg      — VTON results on keyframes")
    print(f"  warped_NNN.jpg        — flow-warped inter-frames")
    print(f"  output_tryon.mp4      — final try-on video")
    print(f"  comparison.mp4        — side-by-side original vs try-on")

    # Cleanup
    from services.video_pipeline.pose_tracker import release as release_pose
    from services.video_pipeline.optical_flow import release as release_flow
    release_pose()
    release_flow()


if __name__ == "__main__":
    asyncio.run(run_real_test())
