"""End-to-end integration test for RT-VTON video pipeline.

// intent: run full pipeline with mock VTON to validate all stages
// status: done
// confidence: high

Mocks the VTON inference step (fashn_vton not fully installed) and runs
everything else for real: frame extraction, pose tracking, keyframe
detection, RAFT optical flow, garment warping, temporal smoothing,
video assembly.

Produces an actual output video + per-stage timing report.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
#  Synthetic test video with a moving "person" shape
# ---------------------------------------------------------------------------

def create_test_video(path: str, num_frames: int = 45, fps: int = 15, w: int = 320, h: int = 480):
    """Create a test video with a moving torso-like shape.

    Simulates a person swaying left-right with a colored rectangle
    as the 'garment region' and a circle as the 'head'.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.full((h, w, 3), 220, dtype=np.uint8)  # Light gray background

        # Sway offset (sinusoidal motion)
        t = i / num_frames
        sway = int(30 * np.sin(2 * np.pi * t))

        cx = w // 2 + sway

        # Head (circle)
        cv2.circle(frame, (cx, 100), 35, (200, 170, 140), -1)

        # Torso / garment region (rectangle)
        torso_x1 = cx - 60
        torso_y1 = 140
        torso_x2 = cx + 60
        torso_y2 = 320
        cv2.rectangle(frame, (torso_x1, torso_y1), (torso_x2, torso_y2), (0, 100, 180), -1)

        # Arms
        cv2.line(frame, (torso_x1, 160), (torso_x1 - 40, 250), (200, 170, 140), 8)
        cv2.line(frame, (torso_x2, 160), (torso_x2 + 40, 250), (200, 170, 140), 8)

        # Legs
        cv2.line(frame, (cx - 20, torso_y2), (cx - 30, h - 20), (50, 50, 120), 10)
        cv2.line(frame, (cx + 20, torso_y2), (cx + 30, h - 20), (50, 50, 120), 10)

        writer.write(frame)

    writer.release()
    print(f"Created test video: {path} ({num_frames} frames, {fps} FPS, {w}x{h})")


def create_garment_image(w: int = 320, h: int = 480) -> Image.Image:
    """Create a fake garment image (red rectangle on white background)."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 80), (260, 350), (220, 40, 40), -1)  # Red garment
    # Add some texture
    for y in range(80, 350, 15):
        cv2.line(img, (60, y), (260, y), (200, 30, 30), 1)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
#  Mock VTON: overlay garment color onto detected torso region
# ---------------------------------------------------------------------------

def mock_vton_result(person_frame: Image.Image, garment_image: Image.Image) -> Image.Image:
    """Simulate VTON by tinting the torso region with garment color.

    Detects the darkest rectangular region (our synthetic torso) and
    replaces it with a blend of the garment's dominant color.
    """
    person_arr = np.array(person_frame, dtype=np.float32)
    garment_arr = np.array(garment_image, dtype=np.float32)

    # Garment dominant color (mean of non-white pixels)
    mask = np.mean(garment_arr, axis=2) < 240
    if np.sum(mask) > 100:
        garment_color = np.mean(garment_arr[mask], axis=0)
    else:
        garment_color = np.array([220, 40, 40], dtype=np.float32)

    # Find torso region (darkest area in person image)
    gray = np.mean(person_arr, axis=2)
    # Threshold for synthetic torso
    torso_mask = gray < 150

    result = person_arr.copy()
    # Blend garment color into torso region
    result[torso_mask] = person_arr[torso_mask] * 0.3 + garment_color * 0.7

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
#  Main E2E test
# ---------------------------------------------------------------------------

async def run_e2e_test():
    """Run full pipeline with mock VTON, report per-stage timings."""
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.chdir(Path(__file__).resolve().parent.parent)

    output_dir = Path("/tmp/fitview_e2e_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic test data
    video_path = str(output_dir / "input.mp4")
    create_test_video(video_path, num_frames=45, fps=15)
    garment_image = create_garment_image()
    garment_image.save(str(output_dir / "garment.png"))

    # 2. Run stages individually with timing
    stages_report = []

    # --- Stage 1: Frame extraction ---
    t0 = time.time()
    from services.video_pipeline.frame_extractor import extract_frames, get_video_metadata

    metadata = get_video_metadata(video_path)
    frames = list(extract_frames(video_path, target_fps=15, max_duration_sec=10))
    total_frames = len(frames)
    dt = time.time() - t0
    stages_report.append(("frame_extraction", dt, {
        "frames": total_frames,
        "source_fps": metadata.fps,
        "resolution": f"{metadata.width}x{metadata.height}",
    }))
    print(f"\n[1/7] Frame extraction: {total_frames} frames in {dt*1000:.0f}ms")

    # --- Stage 2: Pose tracking ---
    # MediaPipe can't detect poses on synthetic stick figures, so we
    # also generate synthetic landmarks that mimic the video's motion.
    # Real pose tracking still runs to measure FPS.
    t0 = time.time()
    from services.video_pipeline.pose_tracker import detect_pose, PoseResult

    real_poses: list[PoseResult] = []
    for _, frame, _ in frames:
        pose = detect_pose(frame)
        real_poses.append(pose)
    real_detected = sum(1 for p in real_poses if p.detected)
    dt = time.time() - t0
    stages_report.append(("pose_tracking", dt, {
        "real_detected": real_detected,
        "total": total_frames,
        "fps": f"{total_frames/dt:.1f}",
        "note": "synthetic landmarks injected for keyframe testing",
    }))
    print(f"[2/7] Pose tracking: {real_detected}/{total_frames} real detections in {dt*1000:.0f}ms ({total_frames/dt:.1f} FPS)")

    # Generate synthetic poses matching the video's sinusoidal sway
    poses: list[PoseResult] = []
    for i in range(total_frames):
        t_norm = i / total_frames
        sway = 0.09 * np.sin(2 * np.pi * t_norm)  # Normalized shoulder sway

        # 33 landmarks, (x, y, visibility)
        lm = np.full((33, 3), 0.5, dtype=np.float32)
        lm[:, 2] = 0.9  # visibility

        # Set garment-relevant landmarks with sway
        cx = 0.5 + sway
        lm[11] = [cx - 0.12, 0.30, 0.9]  # left shoulder
        lm[12] = [cx + 0.12, 0.30, 0.9]  # right shoulder
        lm[13] = [cx - 0.18, 0.45, 0.9]  # left elbow
        lm[14] = [cx + 0.18, 0.45, 0.9]  # right elbow
        lm[15] = [cx - 0.20, 0.55, 0.9]  # left wrist
        lm[16] = [cx + 0.20, 0.55, 0.9]  # right wrist
        lm[23] = [cx - 0.08, 0.65, 0.9]  # left hip
        lm[24] = [cx + 0.08, 0.65, 0.9]  # right hip

        from services.video_pipeline.pose_tracker import GARMENT_LANDMARK_IDS
        garment_lm = lm[GARMENT_LANDMARK_IDS]
        poses.append(PoseResult(landmarks=lm, garment_landmarks=garment_lm, detected=True))
    detected = total_frames
    print(f"       Injected {total_frames} synthetic pose landmarks for pipeline testing")

    # --- Stage 3: Keyframe detection ---
    t0 = time.time()
    from services.video_pipeline.keyframe_detector import KeyframeDetector

    detector = KeyframeDetector(
        pose_threshold=0.05,
        min_interval=5,
        max_interval=30,
    )
    keyframe_indices = []
    for i, pose in enumerate(poses):
        decision = detector.check(i, pose)
        if decision.is_keyframe:
            keyframe_indices.append(i)

    dt = time.time() - t0
    stages_report.append(("keyframe_detection", dt, {
        "keyframes": len(keyframe_indices),
        "indices": keyframe_indices,
        "ratio": f"{len(keyframe_indices)/total_frames*100:.0f}%",
    }))
    print(f"[3/7] Keyframe detection: {len(keyframe_indices)} keyframes {keyframe_indices} in {dt*1000:.0f}ms")

    # --- Stage 4: Mock VTON on keyframes ---
    t0 = time.time()
    from services.video_pipeline.async_vton_worker import KeyframeVTONResult

    vton_results: dict[int, KeyframeVTONResult] = {}
    for kf_idx in keyframe_indices:
        person_frame = frames[kf_idx][1]
        result_image = mock_vton_result(person_frame, garment_image)

        # Create a simple garment mask (where we changed pixels)
        diff = np.abs(
            np.array(result_image, dtype=np.float32) -
            np.array(person_frame, dtype=np.float32)
        ).mean(axis=2)
        mask = (diff > 20).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)

        vton_results[kf_idx] = KeyframeVTONResult(
            frame_idx=kf_idx,
            result_image=result_image,
            garment_mask=mask,
            processing_time_ms=int((time.time() - t0) * 1000),
        )

    dt = time.time() - t0
    stages_report.append(("mock_vton", dt, {
        "keyframes_processed": len(vton_results),
        "note": "mock — real VTON would take ~5s per keyframe",
    }))
    print(f"[4/7] Mock VTON: {len(vton_results)} keyframes in {dt*1000:.0f}ms (mock)")

    # Save a keyframe result for visual inspection
    if vton_results:
        first_kf = list(vton_results.values())[0]
        first_kf.result_image.save(str(output_dir / "keyframe_vton_result.png"))

    # --- Stage 5: Optical flow + garment warping ---
    t0 = time.time()
    from services.video_pipeline.optical_flow import estimate_flow
    from services.video_pipeline.garment_warper import warp_garment

    output_frames: list[Image.Image] = [None] * total_frames
    output_masks: list[np.ndarray | None] = [None] * total_frames

    # Place keyframe results
    for kf_idx, kf_result in vton_results.items():
        output_frames[kf_idx] = kf_result.result_image
        output_masks[kf_idx] = kf_result.garment_mask

    sorted_kf = sorted(keyframe_indices)
    warped_count = 0
    flow_times = []

    for frame_idx in range(total_frames):
        if output_frames[frame_idx] is not None:
            continue

        # Find nearest preceding keyframe
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
            frames[prev_kf_idx][1],
            frames[frame_idx][1],
            occlusion_threshold=2.0,
            device="cpu",
        )
        flow_times.append(time.time() - ft0)

        warped_frame, warped_mask = warp_garment(
            kf_result.result_image,
            kf_result.garment_mask,
            frames[frame_idx][1],
            flow_result,
        )

        output_frames[frame_idx] = warped_frame
        output_masks[frame_idx] = warped_mask
        warped_count += 1

    dt = time.time() - t0
    avg_flow_ms = np.mean(flow_times) * 1000 if flow_times else 0
    flow_fps = 1000 / avg_flow_ms if avg_flow_ms > 0 else 0
    stages_report.append(("flow_warping", dt, {
        "warped_frames": warped_count,
        "avg_flow_ms": f"{avg_flow_ms:.0f}",
        "flow_fps": f"{flow_fps:.1f}",
    }))
    print(f"[5/7] Flow warping: {warped_count} frames in {dt*1000:.0f}ms (avg flow: {avg_flow_ms:.0f}ms = {flow_fps:.1f} FPS)")

    # --- Stage 6: Temporal smoothing ---
    t0 = time.time()
    from services.video_pipeline.temporal_smoother import TemporalSmoother, blend_keyframe_transition

    smoother = TemporalSmoother(blend_window=3)
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
    print(f"[6/7] Temporal smoothing: {blended_count} frames blended in {dt*1000:.0f}ms")

    # --- Stage 7: Video assembly ---
    t0 = time.time()
    from services.video_pipeline.video_assembler import assemble_video

    valid_frames = [f for f in output_frames if f is not None]
    output_video_path = str(output_dir / "output.mp4")
    assemble_video(valid_frames, output_video_path, fps=15)

    dt = time.time() - t0
    stages_report.append(("video_assembly", dt, {"output_frames": len(valid_frames)}))
    print(f"[7/7] Video assembly: {len(valid_frames)} frames → {output_video_path} in {dt*1000:.0f}ms")

    # --- Quality metrics ---
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    # Temporal consistency: SSIM between consecutive output frames in garment region
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
    print(f"Temporal SSIM (consecutive frames): avg={avg_ssim:.4f}, min={min_ssim:.4f} (target: >0.85)")

    # Garment region coverage
    coverage = []
    for mask in output_masks:
        if mask is not None:
            coverage.append(float(np.mean(mask > 0.5)))
    avg_coverage = np.mean(coverage) if coverage else 0
    print(f"Garment mask coverage: avg={avg_coverage*100:.1f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("STAGE TIMING SUMMARY")
    print("=" * 60)
    total_time = sum(dt for _, dt, _ in stages_report)
    for name, dt, details in stages_report:
        pct = dt / total_time * 100
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"  {name:25s} {dt*1000:8.0f}ms ({pct:4.1f}%)  {detail_str}")
    print(f"  {'TOTAL':25s} {total_time*1000:8.0f}ms")
    print(f"  Effective FPS: {total_frames / total_time:.2f}")

    print(f"\nOutput files in: {output_dir}")
    print(f"  input.mp4             — synthetic input video")
    print(f"  garment.png           — garment image")
    print(f"  keyframe_vton_result.png — first keyframe VTON result")
    print(f"  output.mp4            — final try-on video")

    # Release resources
    from services.video_pipeline.pose_tracker import release as release_pose
    from services.video_pipeline.optical_flow import release as release_flow
    release_pose()
    release_flow()

    return {
        "total_frames": total_frames,
        "keyframes": len(keyframe_indices),
        "warped": warped_count,
        "blended": blended_count,
        "total_ms": int(total_time * 1000),
        "effective_fps": round(total_frames / total_time, 2),
        "avg_ssim": round(avg_ssim, 4),
        "output": output_video_path,
    }


if __name__ == "__main__":
    result = asyncio.run(run_e2e_test())
    print(f"\nResult: {result}")
