// intent: Client-side MediaPipe PoseLandmarker for real-time skeleton overlay
// status: done
// next: wire into live-size page overlay canvas
// confidence: high

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ── Types ──

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface PoseDebugInfo {
  fps: number;
  landmarkCount: number;
  avgConfidence: number;
  detectMs: number;
  modelLoaded: boolean;
  error: string | null;
}

export interface UsePoseDetectorReturn {
  landmarks: PoseLandmark[] | null;
  worldLandmarks: PoseLandmark[] | null;
  debug: PoseDebugInfo;
  drawSkeleton: (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    options?: DrawOptions
  ) => void;
}

export interface DrawOptions {
  showSkeleton?: boolean;
  showPoints?: boolean;
  showFaceMesh?: boolean;
  showBoundingBox?: boolean;
  showLabels?: boolean;
  showConfidence?: boolean;
  pointRadius?: number;
  lineWidth?: number;
  skeletonColor?: string;
  pointColor?: string;
  lowConfColor?: string;
  confThreshold?: number;
  mirror?: boolean;
}

const DEFAULT_DRAW: Required<DrawOptions> = {
  showSkeleton: true,
  showPoints: true,
  showFaceMesh: true,
  showBoundingBox: true,
  showLabels: false,
  showConfidence: true,
  pointRadius: 4,
  lineWidth: 2.5,
  skeletonColor: "#00ff88",
  pointColor: "#00ff88",
  lowConfColor: "#ff4444",
  confThreshold: 0.3,
  mirror: true,
};

// ── 33 MediaPipe Pose Landmark connections (skeleton) ──
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

const POSE_CONNECTIONS: [number, number][] = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left hand (wrist → fingers)
  [15, 17], [15, 19], [15, 21], [17, 19],
  // Right hand
  [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27],
  // Right leg
  [24, 26], [26, 28],
  // Left foot
  [27, 29], [27, 31], [29, 31],
  // Right foot
  [28, 30], [28, 32], [30, 32],
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
];

// Landmark names for label overlay
const LANDMARK_NAMES: Record<number, string> = {
  0: "nose",
  11: "L shoulder",
  12: "R shoulder",
  13: "L elbow",
  14: "R elbow",
  15: "L wrist",
  16: "R wrist",
  23: "L hip",
  24: "R hip",
  25: "L knee",
  26: "R knee",
  27: "L ankle",
  28: "R ankle",
};

// Body region coloring for visual clarity
const REGION_COLORS: Record<string, [number, number][]> = {
  "#00ff88": [[11, 12], [11, 23], [12, 24], [23, 24]], // torso — green
  "#00aaff": [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19]], // left arm — blue
  "#ffaa00": [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20]], // right arm — orange
  "#aa66ff": [[23, 25], [25, 27], [27, 29], [27, 31], [29, 31]], // left leg — purple
  "#ff66aa": [[24, 26], [26, 28], [28, 30], [28, 32], [30, 32]], // right leg — pink
  "#ffffff80": [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]], // face — faint white
};

// Build a quick lookup: connection → color
const CONNECTION_COLOR_MAP = new Map<string, string>();
for (const [color, connections] of Object.entries(REGION_COLORS)) {
  for (const [a, b] of connections) {
    CONNECTION_COLOR_MAP.set(`${a}-${b}`, color);
  }
}

export function usePoseDetector(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  enabled: boolean
): UsePoseDetectorReturn {
  const landmarksRef = useRef<PoseLandmark[] | null>(null);
  const worldLandmarksRef = useRef<PoseLandmark[] | null>(null);
  const [landmarks, setLandmarks] = useState<PoseLandmark[] | null>(null);
  const [worldLandmarks, setWorldLandmarks] = useState<PoseLandmark[] | null>(null);
  const [debug, setDebug] = useState<PoseDebugInfo>({
    fps: 0,
    landmarkCount: 0,
    avgConfidence: 0,
    detectMs: 0,
    modelLoaded: false,
    error: null,
  });

  const detectorRef = useRef<unknown>(null);
  const rafIdRef = useRef<number>(0);
  const frameTimesRef = useRef<number[]>([]);
  const lastStateUpdateRef = useRef<number>(0);

  // Initialize PoseLandmarker
  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function init() {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const { PoseLandmarker, FilesetResolver } = vision;

        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const detector = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (cancelled) {
          detector.close();
          return;
        }

        detectorRef.current = detector;
        setDebug((prev) => ({ ...prev, modelLoaded: true, error: null }));
      } catch (err) {
        if (!cancelled) {
          setDebug((prev) => ({
            ...prev,
            error: err instanceof Error ? err.message : "Failed to load PoseLandmarker",
          }));
        }
      }
    }

    void init();

    return () => {
      cancelled = true;
      if (detectorRef.current) {
        (detectorRef.current as { close: () => void }).close();
        detectorRef.current = null;
      }
    };
  }, [enabled]);

  // Detection loop
  useEffect(() => {
    if (!enabled || !detectorRef.current) return;

    const video = videoRef.current;
    if (!video) return;

    let running = true;

    function detectFrame() {
      if (!running || !detectorRef.current || !video) return;

      if (video.readyState >= 2 && video.videoWidth > 0) {
        const t0 = performance.now();

        try {
          const detector = detectorRef.current as {
            detectForVideo: (
              video: HTMLVideoElement,
              timestamp: number
            ) => {
              landmarks: PoseLandmark[][];
              worldLandmarks: PoseLandmark[][];
            };
          };

          const result = detector.detectForVideo(video, performance.now());

          if (result.landmarks.length > 0) {
            landmarksRef.current = result.landmarks[0];
            worldLandmarksRef.current = result.worldLandmarks[0];
          } else {
            landmarksRef.current = null;
            worldLandmarksRef.current = null;
          }

          const detectMs = performance.now() - t0;
          const now = performance.now();

          // Track FPS
          frameTimesRef.current.push(now);
          const cutoff = now - 1000;
          frameTimesRef.current = frameTimesRef.current.filter((t) => t > cutoff);

          // Update React state at ~10Hz to avoid render thrash
          if (now - lastStateUpdateRef.current > 100) {
            lastStateUpdateRef.current = now;
            const lm = landmarksRef.current;

            const visibleCount = lm
              ? lm.filter((l) => l.visibility > 0.3).length
              : 0;
            const avgConf = lm
              ? lm.reduce((sum, l) => sum + l.visibility, 0) / lm.length
              : 0;

            setLandmarks(lm ? [...lm] : null);
            setWorldLandmarks(worldLandmarksRef.current ? [...worldLandmarksRef.current] : null);
            setDebug((prev) => ({
              ...prev,
              fps: frameTimesRef.current.length,
              landmarkCount: visibleCount,
              avgConfidence: avgConf,
              detectMs: Math.round(detectMs * 10) / 10,
            }));
          }
        } catch {
          // Detection can throw on invalid frames — skip silently
        }
      }

      rafIdRef.current = requestAnimationFrame(detectFrame);
    }

    rafIdRef.current = requestAnimationFrame(detectFrame);

    return () => {
      running = false;
      cancelAnimationFrame(rafIdRef.current);
    };
  }, [enabled, videoRef, debug.modelLoaded]);

  // Draw function — called externally per animation frame
  const drawSkeleton = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      options?: DrawOptions
    ) => {
      const opts = { ...DEFAULT_DRAW, ...options };
      const lm = landmarksRef.current;

      ctx.clearRect(0, 0, width, height);

      if (!lm || lm.length === 0) return;

      const toX = (x: number) => (opts.mirror ? 1 - x : x) * width;
      const toY = (y: number) => y * height;

      // Draw skeleton connections (colored by body region)
      if (opts.showSkeleton) {
        for (const [a, b] of POSE_CONNECTIONS) {
          if (a >= lm.length || b >= lm.length) continue;
          const la = lm[a];
          const lb = lm[b];
          if (la.visibility < opts.confThreshold || lb.visibility < opts.confThreshold) continue;

          const color = CONNECTION_COLOR_MAP.get(`${a}-${b}`) ?? opts.skeletonColor;
          ctx.strokeStyle = color;
          ctx.lineWidth = opts.lineWidth;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(toX(la.x), toY(la.y));
          ctx.lineTo(toX(lb.x), toY(lb.y));
          ctx.stroke();
        }
      }

      // Draw landmark points
      if (opts.showPoints) {
        for (let i = 0; i < lm.length; i++) {
          const l = lm[i];
          if (l.visibility < opts.confThreshold) continue;

          // Face landmarks (0-10) get smaller dots
          const isFace = i <= 10;
          const radius = isFace && !opts.showFaceMesh
            ? 0
            : isFace
              ? opts.pointRadius * 0.6
              : opts.pointRadius;

          if (radius === 0) continue;

          const confColor = l.visibility > 0.7
            ? opts.pointColor
            : l.visibility > 0.5
              ? "#ffaa00"
              : opts.lowConfColor;

          ctx.fillStyle = confColor;
          ctx.beginPath();
          ctx.arc(toX(l.x), toY(l.y), radius, 0, Math.PI * 2);
          ctx.fill();

          // White border for visibility
          ctx.strokeStyle = "rgba(255,255,255,0.6)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Draw confidence values next to key joints
      if (opts.showConfidence) {
        ctx.font = "10px monospace";
        ctx.textBaseline = "bottom";
        for (const [idx, name] of Object.entries(LANDMARK_NAMES)) {
          const i = Number(idx);
          if (i >= lm.length) continue;
          const l = lm[i];
          if (l.visibility < opts.confThreshold) continue;

          const conf = Math.round(l.visibility * 100);
          ctx.fillStyle = l.visibility > 0.7 ? "#00ff88" : "#ffaa00";
          ctx.fillText(`${conf}%`, toX(l.x) + 6, toY(l.y) - 2);
        }
      }

      // Draw labels
      if (opts.showLabels) {
        ctx.font = "11px monospace";
        ctx.textBaseline = "top";
        for (const [idx, name] of Object.entries(LANDMARK_NAMES)) {
          const i = Number(idx);
          if (i >= lm.length) continue;
          const l = lm[i];
          if (l.visibility < opts.confThreshold) continue;

          ctx.fillStyle = "rgba(255,255,255,0.9)";
          ctx.fillText(name, toX(l.x) + 8, toY(l.y) + 4);
        }
      }

      // Draw bounding box around detected pose
      if (opts.showBoundingBox) {
        const visible = lm.filter((l) => l.visibility > opts.confThreshold);
        if (visible.length > 4) {
          const xs = visible.map((l) => toX(l.x));
          const ys = visible.map((l) => toY(l.y));
          const pad = 12;
          const minX = Math.min(...xs) - pad;
          const maxX = Math.max(...xs) + pad;
          const minY = Math.min(...ys) - pad;
          const maxY = Math.max(...ys) + pad;

          ctx.strokeStyle = "rgba(0, 255, 136, 0.4)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
          ctx.setLineDash([]);
        }
      }
    },
    []
  );

  return { landmarks, worldLandmarks, debug, drawSkeleton };
}
