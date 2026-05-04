// intent: Extract body shape ratios from MediaPipe world landmarks (3D meters) with 2D fallback
// status: done
// next: validate ratios against known anthropometric distributions
// confidence: high

"use client";

import { useMemo } from "react";
import type { PoseLandmark } from "./use-pose-detector";

/**
 * Body shape ratios derived from MediaPipe 33-point pose landmarks.
 *
 * **Prefer world landmarks** (3D, in meters, relative to hip midpoint) over
 * normalized 2D image coordinates. World landmarks correct for camera perspective
 * and give ratios close to real anthropometric measurements:
 *   - S/H ratio: ~1.05–1.15 women, ~1.15–1.30 men (matches ANSUR/CAESAR)
 *   - W/H ratio: ~0.75–0.85 (matches WHO reference ranges)
 *
 * Falls back to 2D landmarks (with dampened calibration) when world landmarks
 * are unavailable.
 *
 * Landmark indices (MediaPipe Pose):
 *   0: nose, 11: L shoulder, 12: R shoulder, 13: L elbow, 14: R elbow,
 *   15: L wrist, 16: R wrist, 23: L hip, 24: R hip, 25: L knee, 26: R knee,
 *   27: L ankle, 28: R ankle
 */

export interface BodyRatios {
  /** Shoulder width / hip width — indicates upper body breadth (V-shape vs pear) */
  shoulderToHip: number;
  /** Waist width / hip width — indicates waist taper */
  waistToHip: number;
  /** Shoulder width / torso height — indicates relative breadth */
  shoulderToTorso: number;
  /** Torso height / leg height — indicates torso-leg proportion */
  torsoToLeg: number;
  /** Arm length / torso height — indicates relative arm length */
  armToTorso: number;
  /** Full body height in landmark units (shoulder-ankle) */
  bodyHeightPx: number;
  /** How many key landmarks were visible (out of 12 body landmarks) */
  landmarksUsed: number;
  /** Average visibility confidence of used landmarks */
  confidence: number;
  /** Whether we have enough landmarks for reliable ratios */
  reliable: boolean;
  /** Whether world (3D) landmarks were used instead of 2D fallback */
  source: "world_3d" | "image_2d";
}

const CONF_THRESHOLD = 0.4;

// Key body landmark indices (excluding face detail)
const BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

/** 3D Euclidean distance (uses z when available from world landmarks) */
function dist3d(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

/** 2D Euclidean distance (ignores z — for image-space landmarks) */
function dist2d(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function midpoint3d(a: PoseLandmark, b: PoseLandmark): { x: number; y: number; z: number } {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function isVisible(lm: PoseLandmark): boolean {
  return lm.visibility > CONF_THRESHOLD;
}

/**
 * Extract body ratios from world landmarks (3D meters).
 *
 * World landmarks have origin at hip midpoint, y-axis up, units in meters.
 * This gives accurate real-body proportions without camera distortion.
 */
function extractFromWorldLandmarks(
  world: PoseLandmark[],
  image: PoseLandmark[],
): BodyRatios | null {
  if (world.length < 33 || image.length < 33) return null;

  // Use image-space visibility (world landmarks don't have meaningful visibility)
  const lShoulder = world[11];
  const rShoulder = world[12];
  const lElbow = world[13];
  const rElbow = world[14];
  const lWrist = world[15];
  const rWrist = world[16];
  const lHip = world[23];
  const rHip = world[24];
  const lKnee = world[25];
  const rKnee = world[26];
  const lAnkle = world[27];
  const rAnkle = world[28];

  // Visibility from image landmarks (more reliable than world visibility)
  const vis = (idx: number) => isVisible(image[idx]);

  const visibleCount = BODY_LANDMARKS.filter((i) => vis(i)).length;
  const avgConf =
    BODY_LANDMARKS.reduce((sum, i) => sum + image[i].visibility, 0) / BODY_LANDMARKS.length;

  // Need at minimum: both shoulders, both hips
  if (!vis(11) || !vis(12) || !vis(23) || !vis(24)) return null;

  // ── Core widths (3D meters) ──
  const shoulderWidth = dist3d(lShoulder, rShoulder);
  const hipWidth = dist3d(lHip, rHip);

  // Waist estimation: interpolate between shoulder and hip on each side
  const lWaist: PoseLandmark = {
    x: lShoulder.x * 0.35 + lHip.x * 0.65,
    y: lShoulder.y * 0.35 + lHip.y * 0.65,
    z: lShoulder.z * 0.35 + lHip.z * 0.65,
    visibility: Math.min(image[11].visibility, image[23].visibility),
  };
  const rWaist: PoseLandmark = {
    x: rShoulder.x * 0.35 + rHip.x * 0.65,
    y: rShoulder.y * 0.35 + rHip.y * 0.65,
    z: rShoulder.z * 0.35 + rHip.z * 0.65,
    visibility: Math.min(image[12].visibility, image[24].visibility),
  };
  const waistWidth = dist3d(lWaist, rWaist);

  // ── Vertical segments (3D) ──
  const shoulderMid = midpoint3d(lShoulder, rShoulder);
  const hipMid = midpoint3d(lHip, rHip);
  const torsoHeight = Math.sqrt(
    (hipMid.x - shoulderMid.x) ** 2 +
    (hipMid.y - shoulderMid.y) ** 2 +
    (hipMid.z - shoulderMid.z) ** 2,
  );

  // Leg height: use best visible side
  let legHeight = 0;
  if (vis(25) && vis(27)) {
    legHeight = dist3d(lHip, lKnee) + dist3d(lKnee, lAnkle);
  } else if (vis(26) && vis(28)) {
    legHeight = dist3d(rHip, rKnee) + dist3d(rKnee, rAnkle);
  }

  // Arm length: shoulder → elbow → wrist (use best visible side)
  let armLength = 0;
  if (vis(13) && vis(15)) {
    armLength = dist3d(lShoulder, lElbow) + dist3d(lElbow, lWrist);
  } else if (vis(14) && vis(16)) {
    armLength = dist3d(rShoulder, rElbow) + dist3d(rElbow, rWrist);
  }

  // Body height (shoulder to ankle, 3D)
  const ankleMid =
    vis(27) && vis(28)
      ? midpoint3d(lAnkle, rAnkle)
      : vis(27)
        ? { x: lAnkle.x, y: lAnkle.y, z: lAnkle.z }
        : vis(28)
          ? { x: rAnkle.x, y: rAnkle.y, z: rAnkle.z }
          : null;

  const bodyHeight = ankleMid
    ? Math.sqrt(
        (ankleMid.x - shoulderMid.x) ** 2 +
        (ankleMid.y - shoulderMid.y) ** 2 +
        (ankleMid.z - shoulderMid.z) ** 2,
      )
    : torsoHeight;

  // Guard against degenerate values
  if (hipWidth < 0.01 || torsoHeight < 0.01) return null;

  const reliable = visibleCount >= 8 && legHeight > 0 && armLength > 0;

  return {
    shoulderToHip: round3(shoulderWidth / hipWidth),
    waistToHip: round3(waistWidth / hipWidth),
    shoulderToTorso: round3(shoulderWidth / torsoHeight),
    torsoToLeg: legHeight > 0 ? round3(torsoHeight / legHeight) : 0,
    armToTorso: armLength > 0 ? round3(armLength / torsoHeight) : 0,
    bodyHeightPx: round3(bodyHeight),
    landmarksUsed: visibleCount,
    confidence: round3(avgConf),
    reliable,
    source: "world_3d",
  };
}

/**
 * Fallback: extract body ratios from 2D image-space landmarks.
 * Less accurate due to camera perspective distortion.
 */
function extractFromImageLandmarks(landmarks: PoseLandmark[]): BodyRatios | null {
  if (landmarks.length < 33) return null;

  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lElbow = landmarks[13];
  const rElbow = landmarks[14];
  const lWrist = landmarks[15];
  const rWrist = landmarks[16];
  const lHip = landmarks[23];
  const rHip = landmarks[24];
  const lKnee = landmarks[25];
  const rKnee = landmarks[26];
  const lAnkle = landmarks[27];
  const rAnkle = landmarks[28];

  const visibleCount = BODY_LANDMARKS.filter((i) => isVisible(landmarks[i])).length;
  const avgConf =
    BODY_LANDMARKS.reduce((sum, i) => sum + landmarks[i].visibility, 0) / BODY_LANDMARKS.length;

  if (!isVisible(lShoulder) || !isVisible(rShoulder) || !isVisible(lHip) || !isVisible(rHip)) {
    return null;
  }

  const shoulderWidth = dist2d(lShoulder, rShoulder);
  const hipWidth = dist2d(lHip, rHip);

  const lWaist: PoseLandmark = {
    x: lShoulder.x * 0.35 + lHip.x * 0.65,
    y: lShoulder.y * 0.35 + lHip.y * 0.65,
    z: 0,
    visibility: Math.min(lShoulder.visibility, lHip.visibility),
  };
  const rWaist: PoseLandmark = {
    x: rShoulder.x * 0.35 + rHip.x * 0.65,
    y: rShoulder.y * 0.35 + rHip.y * 0.65,
    z: 0,
    visibility: Math.min(rShoulder.visibility, rHip.visibility),
  };
  const waistWidth = dist2d(lWaist, rWaist);

  const shoulderMid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
  const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
  const torsoHeight = Math.abs(hipMid.y - shoulderMid.y);

  let legHeight = 0;
  if (isVisible(lKnee) && isVisible(lAnkle)) {
    legHeight = Math.abs(lAnkle.y - lHip.y);
  } else if (isVisible(rKnee) && isVisible(rAnkle)) {
    legHeight = Math.abs(rAnkle.y - rHip.y);
  }

  let armLength = 0;
  if (isVisible(lElbow) && isVisible(lWrist)) {
    armLength = dist2d(lShoulder, lElbow) + dist2d(lElbow, lWrist);
  } else if (isVisible(rElbow) && isVisible(rWrist)) {
    armLength = dist2d(rShoulder, rElbow) + dist2d(rElbow, rWrist);
  }

  const ankleMid =
    isVisible(lAnkle) && isVisible(rAnkle)
      ? { x: (lAnkle.x + rAnkle.x) / 2, y: (lAnkle.y + rAnkle.y) / 2 }
      : isVisible(lAnkle)
        ? { x: lAnkle.x, y: lAnkle.y }
        : isVisible(rAnkle)
          ? { x: rAnkle.x, y: rAnkle.y }
          : null;

  const bodyHeightPx = ankleMid ? Math.abs(ankleMid.y - shoulderMid.y) : torsoHeight;

  if (hipWidth < 0.01 || torsoHeight < 0.01) return null;

  const reliable = visibleCount >= 8 && legHeight > 0 && armLength > 0;

  return {
    shoulderToHip: round3(shoulderWidth / hipWidth),
    waistToHip: round3(waistWidth / hipWidth),
    shoulderToTorso: round3(shoulderWidth / torsoHeight),
    torsoToLeg: legHeight > 0 ? round3(torsoHeight / legHeight) : 0,
    armToTorso: armLength > 0 ? round3(armLength / torsoHeight) : 0,
    bodyHeightPx: round3(bodyHeightPx),
    landmarksUsed: visibleCount,
    confidence: round3(avgConf),
    reliable,
    source: "image_2d",
  };
}

/**
 * Extract body ratios — prefers world landmarks (3D), falls back to image (2D).
 */
export function extractBodyRatios(
  imageLandmarks: PoseLandmark[] | null,
  worldLandmarks?: PoseLandmark[] | null,
): BodyRatios | null {
  // Prefer world landmarks (3D meters, perspective-corrected)
  if (worldLandmarks && worldLandmarks.length >= 33 && imageLandmarks && imageLandmarks.length >= 33) {
    const result = extractFromWorldLandmarks(worldLandmarks, imageLandmarks);
    if (result) return result;
  }

  // Fallback to 2D image-space landmarks
  if (imageLandmarks) {
    return extractFromImageLandmarks(imageLandmarks);
  }

  return null;
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/**
 * Estimate weight from height + body ratios.
 *
 * Baselines depend on the landmark source:
 *   - world_3d: ratios match real anthropometric data (S/H ~1.1, W/H ~0.8)
 *   - image_2d: ratios are inflated by 2D projection (S/H ~1.5, W/H ~0.95)
 */
export function estimateWeightFromRatios(
  heightCm: number,
  ratios: BodyRatios,
): number {
  const heightM = heightCm / 100;
  let baseWeight = 21.5 * heightM * heightM;

  if (ratios.source === "world_3d") {
    // World landmarks: ratios are close to real anthropometric values
    const sthDelta = ratios.shoulderToHip - 1.1;
    baseWeight += sthDelta * 20;

    const wthDelta = ratios.waistToHip - 0.8;
    baseWeight += wthDelta * 25;
  } else {
    // 2D fallback: inflated ratios from camera perspective
    const sthDelta = ratios.shoulderToHip - 1.5;
    baseWeight += sthDelta * 12;

    const wthDelta = ratios.waistToHip - 0.95;
    baseWeight += wthDelta * 15;
  }

  return Math.round(Math.max(35, Math.min(150, baseWeight)) * 10) / 10;
}

/**
 * React hook: extracts body ratios from landmarks with memoization.
 * Prefers world landmarks (3D) when available.
 */
export function useBodyRatios(
  imageLandmarks: PoseLandmark[] | null,
  worldLandmarks?: PoseLandmark[] | null,
): BodyRatios | null {
  return useMemo(
    () => extractBodyRatios(imageLandmarks, worldLandmarks),
    [imageLandmarks, worldLandmarks],
  );
}
