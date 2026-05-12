// intent: One-shot pose detection on a static image via MediaPipe PoseLandmarker (IMAGE mode)
// status: done
// next: validate upper-body skeleton overlay against uploaded photo aspect ratios
// confidence: high

"use client";

import type { PoseLandmark } from "./use-pose-detector";

export interface ImagePoseResult {
  landmarks: PoseLandmark[];
  worldLandmarks: PoseLandmark[];
}

let cachedDetector: unknown = null;
let detectorPromise: Promise<unknown> | null = null;

/**
 * Get or create a PoseLandmarker in IMAGE mode.
 * Cached as singleton — reused across calls.
 */
async function getDetector(): Promise<unknown> {
  if (cachedDetector) return cachedDetector;
  if (detectorPromise) return detectorPromise;

  detectorPromise = (async () => {
    const vision = await import("@mediapipe/tasks-vision");
    const { PoseLandmarker, FilesetResolver } = vision;

    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    );

    const detector = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numPoses: 1,
    });

    cachedDetector = detector;
    return detector;
  })();

  return detectorPromise;
}

/**
 * Run pose detection on a static HTMLImageElement.
 * Returns landmarks + worldLandmarks, or null if no pose found.
 */
export async function detectPoseFromImage(
  image: HTMLImageElement,
): Promise<ImagePoseResult | null> {
  const detector = (await getDetector()) as {
    detect: (image: HTMLImageElement) => {
      landmarks: PoseLandmark[][];
      worldLandmarks: PoseLandmark[][];
    };
  };

  const result = detector.detect(image);

  if (result.landmarks.length === 0) return null;

  return {
    landmarks: result.landmarks[0],
    worldLandmarks: result.worldLandmarks[0],
  };
}

/**
 * Draw skeleton on a canvas from static image landmarks.
 * Reuses the same connection/color logic as the video overlay.
 */
export function drawImageSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmark[],
  width: number,
  height: number,
): void {
  const CONF = 0.3;

  const POSE_CONNECTIONS: [number, number][] = [
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
  ];

  const REGION_COLORS: Record<string, [number, number][]> = {
    "#00ff88": [[11, 12], [11, 23], [12, 24], [23, 24]],
    "#00aaff": [[11, 13], [13, 15]],
    "#ffaa00": [[12, 14], [14, 16]],
  };

  const colorMap = new Map<string, string>();
  for (const [color, conns] of Object.entries(REGION_COLORS)) {
    for (const [a, b] of conns) colorMap.set(`${a}-${b}`, color);
  }

  const toX = (x: number) => x * width;
  const toY = (y: number) => y * height;

  // Connections
  for (const [a, b] of POSE_CONNECTIONS) {
    if (a >= landmarks.length || b >= landmarks.length) continue;
    const la = landmarks[a];
    const lb = landmarks[b];
    if (la.visibility < CONF || lb.visibility < CONF) continue;

    ctx.strokeStyle = colorMap.get(`${a}-${b}`) ?? "#00ff88";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(toX(la.x), toY(la.y));
    ctx.lineTo(toX(lb.x), toY(lb.y));
    ctx.stroke();
  }

  // Key upper-body points only. Full 33-point overlays add noise for top sizing.
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24]) {
    const l = landmarks[i];
    if (!l) continue;
    if (l.visibility < CONF) continue;
    const r = 4;

    ctx.fillStyle = l.visibility > 0.7 ? "#00ff88" : l.visibility > 0.5 ? "#ffaa00" : "#ff4444";
    ctx.beginPath();
    ctx.arc(toX(l.x), toY(l.y), r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}
