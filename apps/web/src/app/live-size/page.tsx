"use client";

import type { InputHTMLAttributes } from "react";
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
import NextImage from "next/image";
import Link from "next/link";
import {
  Bug,
  Camera,
  Eye,
  ImageIcon,
  Loader2,
  Play,
  RotateCcw,
  Ruler,
  Sparkles,
  Square,
  Upload,
  User,
  Video,
  X,
} from "lucide-react";
import {
  getUpperStyleRecommendation,
  getVisualSizeRecommendation,
  type PopulationKey,
  type UpperStyleOccasion,
  type UpperStyleRecommendationResult,
  type UpperStyleSize,
  type VisualSizeResult,
} from "@/lib/api";
import { GARMENTS } from "@/data/garments";
import { usePoseDetector, type DrawOptions } from "@/hooks/use-pose-detector";
import { useBodyRatios, extractBodyRatios, type BodyRatios } from "@/hooks/use-body-ratios";
import { detectPoseFromImage, drawImageSkeleton } from "@/hooks/use-image-pose";
import type { Garment } from "@/types";

type Population = PopulationKey;
type InputMode = "webcam" | "upload";
type StyleSliderAxis = "bold" | "loose" | "warm" | "cover";
type StyleSliders = Record<StyleSliderAxis, number>;
type StyleInput = {
  result: VisualSizeResult;
  ratios: BodyRatios;
  height: number;
};

const POPULATIONS: Array<{ value: Population; label: string }> = [
  { value: "us_women", label: "US women" },
  { value: "vietnamese", label: "Vietnamese" },
  { value: "universal", label: "Universal (legacy)" },
];

const OCCASIONS: Array<{ value: UpperStyleOccasion; label: string }> = [
  { value: "casual", label: "Casual" },
  { value: "work", label: "Work" },
  { value: "date", label: "Date" },
  { value: "formal", label: "Formal" },
  { value: "party", label: "Party" },
];

const STYLE_SLIDER_DEFAULTS: StyleSliders = {
  bold: 0,
  loose: 0,
  warm: 0,
  cover: 0,
};

const STYLE_SLIDERS: Array<{
  axis: StyleSliderAxis;
  label: string;
  minLabel: string;
  maxLabel: string;
}> = [
  { axis: "bold", label: "Statement", minLabel: "Quiet", maxLabel: "Bold" },
  { axis: "loose", label: "Fit vibe", minLabel: "Body-hugging", maxLabel: "Oversized" },
  { axis: "warm", label: "Color mood", minLabel: "Cool", maxLabel: "Warm" },
  { axis: "cover", label: "Coverage", minLabel: "Open", maxLabel: "Covered" },
];

const STYLE_LIMIT_OPTIONS = [
  { value: "5", label: "5 tops" },
  { value: "8", label: "8 tops" },
  { value: "10", label: "10 tops" },
  { value: "15", label: "15 tops" },
];

// intent: use a stable upper-body model category instead of the noisy generic RTR "top" bucket
// status: done
// next: expose garment category as a user choice if we support more top archetypes
// blockers: RTR generic "top" predictions are less stable than shirt/t-shirt for the US anchor
// confidence: medium
const UPPER_BODY_CATEGORY = "shirt";

const USD_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

const VND_FORMATTER = new Intl.NumberFormat("vi-VN", {
  style: "currency",
  currency: "VND",
  maximumFractionDigits: 0,
});

const UPPER_STYLE_SIZES: UpperStyleSize[] = ["XS", "S", "M", "L", "XL", "XXL"];

function toUpperStyleSize(size: string | null | undefined): UpperStyleSize | null {
  if (size === "XXS") return "XS";
  if (size === "XXXL") return "XXL";

  if (
    size === "XS" ||
    size === "S" ||
    size === "M" ||
    size === "L" ||
    size === "XL" ||
    size === "XXL"
  ) {
    return size;
  }
  return null;
}

function getUpperStyleSizeCandidates(size: UpperStyleSize): UpperStyleSize[] {
  const sizeIndex = UPPER_STYLE_SIZES.indexOf(size);
  if (sizeIndex === -1) return [size];

  return [...UPPER_STYLE_SIZES].sort(
    (a, b) => Math.abs(UPPER_STYLE_SIZES.indexOf(a) - sizeIndex) - Math.abs(UPPER_STYLE_SIZES.indexOf(b) - sizeIndex)
  );
}

function formatUsd(value: number): string {
  return USD_FORMATTER.format(value);
}

function formatVnd(value: number | null | undefined): string | null {
  return typeof value === "number" ? VND_FORMATTER.format(value) : null;
}

function scorePct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function getGarmentImage(garment: Garment): string {
  return garment.images.find((image) => image.type === "display")?.url ?? garment.images[0]?.url ?? "";
}

function getRealGarmentForStyleItem(itemId: string): Garment | undefined {
  return GARMENTS.find((garment) => garment.id === itemId);
}

const LIVE_INTERVAL_MS = 3000;

export default function LiveSizePage() {
  const [inputMode, setInputMode] = useState<InputMode>("webcam");
  const [heightCm, setHeightCm] = useState("165");
  const [population, setPopulation] = useState<Population>("us_women");
  const [occasion, setOccasion] = useState<UpperStyleOccasion>("casual");
  const [styleSliders, setStyleSliders] = useState<StyleSliders>(STYLE_SLIDER_DEFAULTS);
  const [styleLimit, setStyleLimit] = useState(5);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveRunning, setLiveRunning] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [visualResult, setVisualResult] = useState<VisualSizeResult | null>(null);
  const [styleResult, setStyleResult] = useState<UpperStyleRecommendationResult | null>(null);
  const [styleLoading, setStyleLoading] = useState(false);
  const [styleError, setStyleError] = useState<string | null>(null);
  const [lastBodyRatios, setLastBodyRatios] = useState<BodyRatios | null>(null);
  const [lastStyleInput, setLastStyleInput] = useState<StyleInput | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);

  // Upload mode state
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadRatios, setUploadRatios] = useState<BodyRatios | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const uploadCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Overlay & debug state
  const [showDebug, setShowDebug] = useState(false);
  const [drawOpts, setDrawOpts] = useState<DrawOptions>({
    showSkeleton: true,
    showPoints: false,
    showFaceMesh: false,
    showBoundingBox: false,
    showLabels: false,
    showConfidence: false,
    mirror: true,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const styleRequestIdRef = useRef(0);
  const firstReliableFrameQueuedRef = useRef(false);

  const { landmarks, worldLandmarks, debug: poseDebug, drawSkeleton } = usePoseDetector(
    videoRef,
    cameraActive
  );

  // Extract body shape ratios — prefers world landmarks (3D meters) over 2D image coords
  const webcamRatios = useBodyRatios(landmarks, worldLandmarks);
  // Use upload ratios when in upload mode, webcam ratios otherwise
  const bodyRatios = inputMode === "upload" ? uploadRatios : webcamRatios;
  const displayBodyRatios = bodyRatios?.reliable ? bodyRatios : (lastBodyRatios ?? bodyRatios);
  const isShowingSavedRatios = Boolean(displayBodyRatios && displayBodyRatios === lastBodyRatios && bodyRatios !== lastBodyRatios);

  // Overlay draw loop — runs at display refresh rate
  // Accounts for object-cover cropping so skeleton aligns pixel-perfectly with the video
  useEffect(() => {
    if (!cameraActive) return;

    let running = true;

    function drawFrame() {
      if (!running) return;

      const overlay = overlayRef.current;
      const video = videoRef.current;
      if (overlay && video && video.videoWidth > 0) {
        const rect = video.getBoundingClientRect();
        const dw = rect.width;
        const dh = rect.height;

        if (overlay.width !== dw || overlay.height !== dh) {
          overlay.width = dw;
          overlay.height = dh;
        }

        const ctx = overlay.getContext("2d");
        if (ctx) {
          // Compute object-cover offset: video may be cropped
          const videoAspect = video.videoWidth / video.videoHeight;
          const displayAspect = dw / dh;

          let renderW = dw;
          let renderH = dh;
          let offsetX = 0;
          let offsetY = 0;

          if (videoAspect > displayAspect) {
            // Video wider than container — horizontal crop
            renderW = dh * videoAspect;
            offsetX = (dw - renderW) / 2;
          } else {
            // Video taller than container — vertical crop
            renderH = dw / videoAspect;
            offsetY = (dh - renderH) / 2;
          }

          ctx.clearRect(0, 0, dw, dh);
          ctx.save();
          ctx.translate(offsetX, offsetY);
          drawSkeleton(ctx, renderW, renderH, drawOpts);
          ctx.restore();
        }
      }

      requestAnimationFrame(drawFrame);
    }

    requestAnimationFrame(drawFrame);

    return () => {
      running = false;
    };
  }, [cameraActive, drawSkeleton, drawOpts]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const currentModel = visualResult?.research_model.model ?? "population-selected GBM";
  const confidenceScore = visualResult?.research_model.confidence_score;
  const recommendedSize = visualResult?.research_model.predicted_size;
  const stylePredictedSize =
    styleResult?.user_state.predicted_size ??
    toUpperStyleSize(visualResult?.research_model.predicted_size) ??
    toUpperStyleSize(visualResult?.rule_based.predicted_size);
  const confidenceLabel = visualResult?.research_model.confidence;
  const estimatedWeight = visualResult?.weight_estimation?.estimated_weight_kg;
  const bodyShape = visualResult?.body_shape?.type;

  const liveStatus = useMemo(() => {
    if (processing) return "Đang phân tích frame mới";
    if (liveRunning) return `Live refresh mỗi ${LIVE_INTERVAL_MS / 1000} giây`;
    if (cameraActive) return "Camera sẵn sàng";
    return "Chưa bật camera";
  }, [cameraActive, liveRunning, processing]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    firstReliableFrameQueuedRef.current = false;
    setLiveRunning(false);
    setCameraActive(false);
  }, []);

  // intent: bridge last valid upper-body size output into richer style recommendations
  // status: done
  // next: persist style preferences in the real profile store when profiles are writable
  // blockers: style service accepts only XS-XXL, while visual model may emit XXS
  // confidence: high
  const runStyleRecommendation = useCallback(
    async (
      input: StyleInput,
      overrides: {
        occasion?: UpperStyleOccasion;
        sliders?: StyleSliders;
        topK?: number;
      } = {}
    ) => {
      const { result, ratios, height } = input;
      const predicted =
        toUpperStyleSize(result.research_model.predicted_size) ??
        toUpperStyleSize(result.rule_based.predicted_size);

      if (!predicted) {
        setStyleLoading(false);
        setStyleError("Style recommendations need a supported size from XS to XXL.");
        return;
      }

      const nextOccasion = overrides.occasion ?? occasion;
      const nextSliders = overrides.sliders ?? styleSliders;
      const nextTopK = overrides.topK ?? styleLimit;
      const requestId = ++styleRequestIdRef.current;
      setStyleLoading(true);
      setStyleError(null);

      try {
        let nextStyleResult: UpperStyleRecommendationResult | null = null;
        for (const candidateSize of getUpperStyleSizeCandidates(predicted)) {
          const candidateResult = await getUpperStyleRecommendation({
            height_cm: height,
            predicted_size: candidateSize,
            population,
            ratios: {
              sh: ratios.shoulderToHip,
              wh: ratios.waistToHip,
              st: ratios.shoulderToTorso,
              tl: ratios.torsoToLeg,
              at: ratios.armToTorso,
            },
            context: {
              occasion: nextOccasion,
              sliders: nextSliders,
            },
            top_k: nextTopK,
          });

          nextStyleResult ??= candidateResult;
          if (candidateResult.items.length > 0) {
            nextStyleResult = candidateResult;
            break;
          }
        }

        if (nextStyleResult && styleRequestIdRef.current === requestId) {
          startTransition(() => {
            setStyleResult(nextStyleResult);
          });
        }
      } catch (err) {
        if (styleRequestIdRef.current === requestId) {
          setStyleError(err instanceof Error ? err.message : "Style recommendation failed");
        }
      } finally {
        if (styleRequestIdRef.current === requestId) {
          setStyleLoading(false);
        }
      }
    },
    [occasion, population, styleLimit, styleSliders]
  );

  const handleOccasionChange = useCallback(
    (value: string) => {
      const nextOccasion = value as UpperStyleOccasion;
      setOccasion(nextOccasion);
      setStyleError(null);
      if (lastStyleInput) {
        void runStyleRecommendation(lastStyleInput, { occasion: nextOccasion });
      }
    },
    [lastStyleInput, runStyleRecommendation]
  );

  const handleStyleSliderChange = useCallback(
    (axis: StyleSliderAxis, value: number) => {
      const nextSliders = { ...styleSliders, [axis]: value };
      setStyleSliders(nextSliders);
      setStyleError(null);
      if (lastStyleInput) {
        void runStyleRecommendation(lastStyleInput, { sliders: nextSliders });
      }
    },
    [lastStyleInput, runStyleRecommendation, styleSliders]
  );

  const handleStyleLimitChange = useCallback(
    (value: string) => {
      const nextTopK = Number.parseInt(value, 10);
      if (!Number.isFinite(nextTopK)) return;

      setStyleLimit(nextTopK);
      setStyleError(null);
      if (lastStyleInput) {
        void runStyleRecommendation(lastStyleInput, { topK: nextTopK });
      }
    },
    [lastStyleInput, runStyleRecommendation]
  );

  const handleResetStyleSliders = useCallback(() => {
    const nextSliders = { ...STYLE_SLIDER_DEFAULTS };
    setStyleSliders(nextSliders);
    setStyleError(null);
    if (lastStyleInput) {
      void runStyleRecommendation(lastStyleInput, { sliders: nextSliders });
    }
  }, [lastStyleInput, runStyleRecommendation]);

  const handleRefreshStyleRecommendations = useCallback(() => {
    if (lastStyleInput) {
      void runStyleRecommendation(lastStyleInput);
    }
  }, [lastStyleInput, runStyleRecommendation]);

  const hasActiveStyleSliders = useMemo(
    () => Object.values(styleSliders).some((value) => Math.abs(value) > 0.05),
    [styleSliders]
  );

  // ── Image upload handlers ──

  const processUploadedImage = useCallback(async (file: File) => {
    setError(null);
    setProcessing(true);
    setUploadRatios(null);

    const url = URL.createObjectURL(file);
    setUploadedImage(url);

    try {
      // Load into an HTMLImageElement for MediaPipe
      const img = new Image();
      img.src = url;
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load image"));
      });

      // Run pose detection (IMAGE mode)
      const pose = await detectPoseFromImage(img);
      if (!pose) {
        setError("No pose detected — make sure your shoulders and upper torso are visible.");
        setProcessing(false);
        return;
      }

      // Draw skeleton overlay on the upload canvas
      const canvas = uploadCanvasRef.current;
      if (canvas) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(img, 0, 0);
          drawImageSkeleton(ctx, pose.landmarks, img.naturalWidth, img.naturalHeight);
        }
      }

      // Extract body ratios (prefer world landmarks)
      const ratios = extractBodyRatios(pose.landmarks, pose.worldLandmarks);
      setUploadRatios(ratios);

      if (!ratios || !ratios.reliable) {
        setError("Need both shoulders visible to estimate an upper-body size.");
        setProcessing(false);
        return;
      }
      setLastBodyRatios(ratios);

      // Run size inference
      const height = Number.parseFloat(heightCm);
      if (!Number.isFinite(height) || height < 100 || height > 250) {
        setError("Set height (100–250 cm) then try again.");
        setProcessing(false);
        return;
      }

      const startedAt = performance.now();
      const result = await getVisualSizeRecommendation({
        height_cm: height,
        shoulder_to_hip: ratios.shoulderToHip,
        waist_to_hip: ratios.waistToHip,
        shoulder_to_torso: ratios.shoulderToTorso,
        torso_to_leg: ratios.torsoToLeg,
        arm_to_torso: ratios.armToTorso,
        age: 30,
        category: UPPER_BODY_CATEGORY,
        population,
        confidence: ratios.confidence,
        source: ratios.source,
      });

      const styleInput = { result, ratios, height };
      setVisualResult(result);
      setLastStyleInput(styleInput);
      setRoundTripMs(Math.round(performance.now() - startedAt));
      setLastUpdatedAt(new Date().toLocaleTimeString());
      void runStyleRecommendation(styleInput);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image processing failed");
    } finally {
      setProcessing(false);
    }
  }, [heightCm, population, runStyleRecommendation]);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      void processUploadedImage(file);
    }
  }, [processUploadedImage]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void processUploadedImage(file);
    e.target.value = "";
  }, [processUploadedImage]);

  const clearUpload = useCallback(() => {
    if (uploadedImage) URL.revokeObjectURL(uploadedImage);
    setUploadedImage(null);
    setUploadRatios(null);
    setError(null);
  }, [uploadedImage]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setError(null);
    firstReliableFrameQueuedRef.current = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setLiveRunning(false);
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : "Failed to access camera");
    }
  }, []);

  const runInference = useCallback(async () => {
    if (busyRef.current) return;

    const height = Number.parseFloat(heightCm);

    if (!Number.isFinite(height) || height < 100 || height > 250) {
      setError("Height must be between 100 and 250 cm.");
      return;
    }

    if (!bodyRatios || !bodyRatios.reliable) {
      setError("Need both shoulders visible — center your upper body in frame.");
      return;
    }

    setError(null);
    setLastBodyRatios(bodyRatios);
    busyRef.current = true;
    setProcessing(true);
    const startedAt = performance.now();

    try {
      const result = await getVisualSizeRecommendation({
        height_cm: height,
        shoulder_to_hip: bodyRatios.shoulderToHip,
        waist_to_hip: bodyRatios.waistToHip,
        shoulder_to_torso: bodyRatios.shoulderToTorso,
        torso_to_leg: bodyRatios.torsoToLeg,
        arm_to_torso: bodyRatios.armToTorso,
        age: 30,
        category: UPPER_BODY_CATEGORY,
        population,
        confidence: bodyRatios.confidence,
        source: bodyRatios.source,
      });

      const styleInput = { result, ratios: bodyRatios, height };
      setVisualResult(result);
      setLastStyleInput(styleInput);
      setRoundTripMs(Math.round(performance.now() - startedAt));
      setLastUpdatedAt(new Date().toLocaleTimeString());
      void runStyleRecommendation(styleInput);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Visual inference failed");
    } finally {
      busyRef.current = false;
      setProcessing(false);
    }
  }, [heightCm, population, bodyRatios, runStyleRecommendation]);

  const runInferenceRef = useRef(runInference);

  useEffect(() => {
    runInferenceRef.current = runInference;
  }, [runInference]);

  useEffect(() => {
    if (!cameraActive || !bodyRatios?.reliable || firstReliableFrameQueuedRef.current) return;

    firstReliableFrameQueuedRef.current = true;
    void runInference();
  }, [bodyRatios, cameraActive, runInference]);

  useEffect(() => {
    if (!cameraActive || !liveRunning) return;

    const initialRunId = window.setTimeout(() => {
      void runInferenceRef.current();
    }, 750);
    const intervalId = window.setInterval(() => {
      void runInferenceRef.current();
    }, LIVE_INTERVAL_MS);

    return () => {
      window.clearTimeout(initialRunId);
      window.clearInterval(intervalId);
    };
  }, [cameraActive, liveRunning]);

  return (
    <div className="relative isolate mx-auto flex min-h-dvh max-w-[100rem] flex-col bg-background px-3 py-3 md:px-4 lg:h-[calc(100dvh-4rem)] lg:min-h-0 lg:overflow-hidden">
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-4 top-3 -z-10 h-28 rounded-full bg-foreground/[0.03] blur-3xl" />
      <div className="mb-3 flex shrink-0 flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-muted-foreground">
              Upper-Body Size Console
            </p>
            <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Scan ready
            </span>
          </div>
          <h1 className="mt-1 text-2xl font-black tracking-tight md:text-4xl">
            Live top sizing
          </h1>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground md:text-sm">
            One-shot sizing by default. Turn on live refresh only when you want repeated predictions.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode toggle */}
          <div className="inline-flex rounded-xl border border-foreground/10 bg-background/80 p-1 shadow-lg shadow-black/5 backdrop-blur">
            <button
              onClick={() => setInputMode("webcam")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
                inputMode === "webcam"
                  ? "bg-foreground text-background shadow-md shadow-black/15"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <Camera className="h-3.5 w-3.5" />
              Webcam
            </button>
            <button
              onClick={() => setInputMode("upload")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
                inputMode === "upload"
                  ? "bg-foreground text-background shadow-md shadow-black/15"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" />
              Upload
            </button>
          </div>
          <Link
            href="/measure"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-semibold shadow-sm transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <Ruler className="h-4 w-4" />
            Manual
          </Link>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[minmax(27rem,0.95fr)_minmax(31rem,1.05fr)]">
        <section className="flex min-h-0 flex-col rounded-3xl border border-border bg-card p-3 shadow-sm md:p-4">
          <div className="grid shrink-0 gap-3 md:grid-cols-3">
            <Field
              label="Height (cm)"
              value={heightCm}
              onChange={setHeightCm}
              inputProps={{ type: "number", min: 100, max: 250 }}
            />
            <SelectField
              label="Population"
              value={population}
              onChange={(value) => setPopulation(value as Population)}
              options={POPULATIONS}
            />
            <div className="rounded-2xl border border-border bg-muted/40 px-3 py-2.5">
              <span className="mb-1 block text-xs font-medium text-muted-foreground">Garment zone</span>
              <p className="text-sm font-black">Tops / upper body only</p>
            </div>
          </div>
          {displayBodyRatios && (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                {isShowingSavedRatios
                  ? "Showing last valid ratios"
                  : displayBodyRatios.reliable
                    ? "Upper body detected"
                    : `Partial (${displayBodyRatios.landmarksUsed}/12)`}
              </span>
              <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                {displayBodyRatios.source === "world_3d" ? "3D world" : "2D image"}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                S/H {displayBodyRatios.shoulderToHip.toFixed(2)}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                W/H {displayBodyRatios.waistToHip.toFixed(2)}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                T/L {displayBodyRatios.torsoToLeg.toFixed(2)}
              </span>
              {estimatedWeight && (
                <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                  ~{estimatedWeight}kg
                </span>
              )}
              {bodyShape && (
                <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                  {bodyShape}
                </span>
              )}
            </div>
          )}

          {/* ── Webcam mode ── */}
          {inputMode === "webcam" && (
            <>
              <div className="mt-3 flex shrink-0 flex-wrap gap-2">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="inline-flex items-center gap-2 rounded-xl bg-foreground px-4 py-2.5 text-sm font-semibold text-background shadow-sm transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <Camera className="h-4 w-4" />
                    Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      onClick={() => setLiveRunning(true)}
                      disabled={liveRunning}
                      className="inline-flex items-center gap-2 rounded-xl bg-foreground px-4 py-2.5 text-sm font-semibold text-background shadow-sm transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Play className="h-4 w-4" />
                      Live refresh
                    </button>
                    <button
                      onClick={() => setLiveRunning(false)}
                      disabled={!liveRunning}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-semibold shadow-sm transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Square className="h-4 w-4" />
                      Stop
                    </button>
                    <button
                      onClick={() => void runInference()}
                      disabled={processing}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-semibold shadow-sm transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {processing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Video className="h-4 w-4" />}
                      Analyze once
                    </button>
                    <button
                      onClick={stopCamera}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-semibold shadow-sm transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </button>
                  </>
                )}
              </div>

              <div className="mt-3 min-h-0 flex-1 overflow-hidden rounded-3xl border border-border bg-black shadow-sm">
                <div className="relative h-full min-h-[21rem] lg:min-h-0">
                  <video
                    ref={videoRef}
                    className="h-full w-full object-cover"
                    style={{ transform: "scaleX(-1)" }}
                    playsInline
                    muted
                  />
                  <canvas
                    ref={overlayRef}
                    className="pointer-events-none absolute inset-0 h-full w-full"
                  />
                  <div aria-hidden="true" className="pointer-events-none absolute inset-3 rounded-[1.5rem] border border-white/10" />
                  {!cameraActive && !cameraError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
                      <User className="h-14 w-14" />
                      <p className="text-sm">Bật camera để bắt đầu.</p>
                    </div>
                  )}
                  {cameraError && (
                    <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-destructive">
                      {cameraError}
                    </div>
                  )}
                  {cameraActive && !landmarks && (
                    <div className="pointer-events-none absolute inset-0">
                      <div className="absolute inset-x-[18%] bottom-[8%] top-[8%] rounded-[28px] border-2 border-dashed border-white/45" />
                      <p className="absolute bottom-4 left-4 right-4 rounded-full bg-black/70 px-3 py-2 text-center text-xs font-semibold text-white backdrop-blur">
                        Keep both shoulders in frame. First reliable reading runs once; live refresh is manual.
                      </p>
                    </div>
                  )}
                  {cameraActive && showDebug && (
                    <div className="pointer-events-none absolute top-3 left-3 rounded-xl bg-black/70 px-3 py-2 font-mono text-[11px] leading-relaxed text-white/80 backdrop-blur-sm">
                      <p>FPS: {poseDebug.fps}</p>
                      <p>Detect: {poseDebug.detectMs}ms</p>
                      <p>Landmarks: {poseDebug.landmarkCount}/33</p>
                      <p>Confidence: {(poseDebug.avgConfidence * 100).toFixed(0)}%</p>
                      <p>Model: {poseDebug.modelLoaded ? "loaded" : "loading..."}</p>
                      {poseDebug.error && <p>Error: {poseDebug.error}</p>}
                      {roundTripMs !== null && <p>API RTT: {roundTripMs}ms</p>}
                    </div>
                  )}
                </div>
              </div>

              {cameraActive && (
                <div className="mt-3 flex shrink-0 flex-wrap gap-2">
                  <ToggleButton active={showDebug} onClick={() => setShowDebug((v) => !v)} icon={<Bug className="h-3.5 w-3.5" />} label="Debug HUD" />
                  <ToggleButton active={drawOpts.showSkeleton ?? true} onClick={() => setDrawOpts((o) => ({ ...o, showSkeleton: !o.showSkeleton }))} icon={<Eye className="h-3.5 w-3.5" />} label="Skeleton" />
                  {showDebug && (
                    <>
                      <ToggleButton active={drawOpts.showPoints ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showPoints: !o.showPoints }))} label="Points" />
                      <ToggleButton active={drawOpts.showFaceMesh ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showFaceMesh: !o.showFaceMesh }))} label="Face" />
                      <ToggleButton active={drawOpts.showBoundingBox ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showBoundingBox: !o.showBoundingBox }))} label="BBox" />
                      <ToggleButton active={drawOpts.showLabels ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showLabels: !o.showLabels }))} label="Labels" />
                      <ToggleButton active={drawOpts.showConfidence ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showConfidence: !o.showConfidence }))} label="Conf %" />
                    </>
                  )}
                </div>
              )}
            </>
          )}

          {/* ── Upload mode ── */}
          {inputMode === "upload" && (
            <>
              <div
                className={`mt-6 overflow-hidden rounded-3xl border-2 border-dashed transition-colors ${
                  isDragging
                    ? "border-primary bg-primary/5"
                    : uploadedImage
                      ? "border-border bg-muted/40"
                      : "border-border bg-muted/40 hover:border-muted-foreground/40"
                }`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleFileDrop}
              >
                {uploadedImage ? (
                  <div className="relative">
                    <canvas
                      ref={uploadCanvasRef}
                      className="h-auto w-full"
                    />
                    <button
                      onClick={clearUpload}
                      className="absolute top-3 right-3 rounded-full bg-black/60 p-2 text-white backdrop-blur-sm transition-colors hover:bg-black/80"
                    >
                      <X className="h-4 w-4" />
                    </button>
                    {processing && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-sm">
                        <Loader2 className="h-8 w-8 animate-spin text-white" />
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    className="flex aspect-[4/3] cursor-pointer flex-col items-center justify-center gap-4 p-8"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-12 w-12 text-muted-foreground/60" />
                    <div className="text-center">
                      <p className="text-sm font-medium">Drop an upper-body photo here</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        or click to browse — JPG, PNG, WebP
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileSelect}
              />
              {uploadedImage && !processing && (
                <div className="mt-4 flex gap-3">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-3 text-sm font-semibold shadow-sm transition-colors hover:bg-accent"
                  >
                    <ImageIcon className="h-4 w-4" />
                    Upload another
                  </button>
                </div>
              )}
            </>
          )}

          {/* ── Status bar (shared) ── */}
          <div className="mt-3 flex shrink-0 flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {inputMode === "webcam" && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1">
                {liveStatus}
              </span>
            )}
            {inputMode === "upload" && uploadRatios && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1">
                Pose detected ({uploadRatios.landmarksUsed}/12 landmarks)
              </span>
            )}
            {poseDebug.modelLoaded && inputMode === "webcam" && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1">
                Pose model loaded
              </span>
            )}
            {roundTripMs !== null && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1">
                Round trip {roundTripMs} ms
              </span>
            )}
            {lastUpdatedAt && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1">
                Cập nhật lúc {lastUpdatedAt}
              </span>
            )}
          </div>

          {error && (
            <div className="mt-4 rounded-2xl border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}
        </section>

        <section className="flex min-h-0 flex-col gap-3 overflow-hidden">
          {/* Size prediction hero */}
          <div className="relative overflow-hidden rounded-3xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Predicted top size</p>
                <div className="mt-2 flex items-end gap-3">
                  <span className="text-5xl font-black tracking-tight md:text-6xl">
                    {recommendedSize ?? "--"}
                  </span>
                  <span
                    className={`mb-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
                      confidenceLabel === "high"
                        ? "bg-foreground text-background"
                        : confidenceLabel === "medium"
                          ? "bg-muted text-foreground"
                          : "bg-zinc-900 text-white dark:bg-zinc-200 dark:text-zinc-950"
                    }`}
                  >
                    {confidenceLabel ?? "waiting"}
                  </span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1.5">
                {processing && <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />}
                {displayBodyRatios && (
                  <span className="rounded-full bg-muted px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {isShowingSavedRatios
                      ? "last valid"
                      : displayBodyRatios.source === "world_3d"
                        ? "3D world"
                        : "2D fallback"}
                  </span>
                )}
              </div>
            </div>

            {/* Probability distribution bar */}
            {visualResult?.research_model?.probabilities && (
              <div className="mt-4">
                <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Size probability distribution
                </p>
                <div className="flex h-8 overflow-hidden rounded-xl border border-border bg-background">
                  {["XXS", "XS", "S", "M", "L", "XL", "XXL"].map((size) => {
                    const prob = visualResult.research_model.probabilities?.[size] ?? 0;
                    if (prob < 0.005) return null;
                    const isTop = size === recommendedSize;
                    return (
                      <div
                        key={size}
                        className={`flex items-center justify-center text-[10px] font-bold transition-all ${
                          isTop
                            ? "bg-foreground text-background"
                            : "bg-muted text-muted-foreground"
                        }`}
                        style={{ width: `${Math.max(prob * 100, 3)}%` }}
                        title={`${size}: ${(prob * 100).toFixed(1)}%`}
                      >
                        {prob > 0.06 ? `${size} ${(prob * 100).toFixed(0)}%` : size}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Key metrics row */}
            <div className="mt-4 grid gap-2 sm:grid-cols-4">
              <MetricCard
                label="Est. weight"
                value={estimatedWeight ? `${estimatedWeight} kg` : "--"}
                helper="From body ratios"
              />
              <MetricCard
                label="Est. BMI"
                value={
                  visualResult?.weight_estimation?.estimated_bmi
                    ? visualResult.weight_estimation.estimated_bmi.toFixed(1)
                    : "--"
                }
                helper="Derived"
              />
              <MetricCard
                label="Body shape"
                value={bodyShape ?? "--"}
                helper="S/H + W/H classification"
              />
              <MetricCard
                label="Confidence"
                value={typeof confidenceScore === "number" ? `${(confidenceScore * 100).toFixed(1)}%` : "--"}
                helper={currentModel.replace("gbm_copula_tempered_a075", "GBM copula")}
              />
            </div>

            {/* Estimated body measurements from rule-based */}
            {visualResult?.rule_based && (
              <div className="mt-2 hidden gap-2 xl:grid xl:grid-cols-3">
                <MetricCard
                  label="Est. chest"
                  value={visualResult.rule_based.estimated_chest_cm ? `${visualResult.rule_based.estimated_chest_cm} cm` : "--"}
                  helper="Rule-based regression"
                />
                <MetricCard
                  label="Est. waist"
                  value={visualResult.rule_based.estimated_waist_cm ? `${visualResult.rule_based.estimated_waist_cm} cm` : "--"}
                  helper="Rule-based regression"
                />
                <MetricCard
                  label="Est. hip"
                  value={visualResult.rule_based.estimated_hip_cm ? `${visualResult.rule_based.estimated_hip_cm} cm` : "--"}
                  helper="Rule-based regression"
                />
              </div>
            )}
          </div>

          <StyleRecommendationsCard
            result={styleResult}
            loading={styleLoading}
            error={styleError}
            occasion={occasion}
            predictedSize={stylePredictedSize}
            styleSliders={styleSliders}
            styleLimit={styleLimit}
            canRefresh={Boolean(lastStyleInput)}
            hasActiveStyleSliders={hasActiveStyleSliders}
            onOccasionChange={handleOccasionChange}
            onStyleSliderChange={handleStyleSliderChange}
            onStyleLimitChange={handleStyleLimitChange}
            onResetStyleSliders={handleResetStyleSliders}
            onRefresh={handleRefreshStyleRecommendations}
          />

          <CompactDiagnosticsCard
            ratios={displayBodyRatios}
            visualResult={visualResult}
            poseLandmarkCount={poseDebug.landmarkCount}
            poseDetectMs={poseDebug.detectMs}
            roundTripMs={roundTripMs}
            model={currentModel}
            recommendedSize={recommendedSize}
            confidenceScore={confidenceScore}
            isShowingSavedRatios={isShowingSavedRatios}
          />
        </section>
      </div>

    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  inputProps,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  inputProps: InputHTMLAttributes<HTMLInputElement>;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted-foreground">{label}</span>
      <input
        {...inputProps}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-2xl border border-border bg-background px-3 py-2.5 text-sm outline-none transition-colors focus:border-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted-foreground">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-2xl border border-border bg-background px-3 py-2.5 text-sm outline-none transition-colors focus:border-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ToggleButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon?: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "border border-foreground bg-foreground text-background"
          : "border border-border bg-muted text-muted-foreground hover:bg-accent"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function CompactDiagnosticsCard({
  ratios,
  visualResult,
  poseLandmarkCount,
  poseDetectMs,
  roundTripMs,
  model,
  recommendedSize,
  confidenceScore,
  isShowingSavedRatios,
}: {
  ratios: BodyRatios | null | undefined;
  visualResult: VisualSizeResult | null;
  poseLandmarkCount: number;
  poseDetectMs: number;
  roundTripMs: number | null;
  model: string;
  recommendedSize: string | null | undefined;
  confidenceScore: number | undefined;
  isShowingSavedRatios: boolean;
}) {
  const modelLabel = model.replace("gbm_copula_tempered_a075", "GBM copula a=.75");
  const weight = visualResult?.weight_estimation?.estimated_weight_kg;
  const bmi = visualResult?.weight_estimation?.estimated_bmi;

  return (
    <div className="hidden shrink-0 rounded-3xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-[0.18em] text-muted-foreground">Diagnostics</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {isShowingSavedRatios
              ? "Holding the last valid reading while the camera updates."
              : "Client ratios, population weight estimate, and served GBM trace."}
          </p>
        </div>
        <span className="rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-[10px] text-muted-foreground">
          {roundTripMs !== null ? `${roundTripMs}ms` : "idle"}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        <MiniStat label="Pose" value={`${poseLandmarkCount}/33`} helper={`${poseDetectMs || 0}ms`} />
        <MiniStat label="Source" value={ratios?.source === "world_3d" ? "3D" : ratios ? "2D" : "--"} helper={ratios?.reliable ? "reliable" : "waiting"} />
        <MiniStat label="Weight" value={typeof weight === "number" ? `${weight}kg` : "--"} helper={typeof bmi === "number" ? `BMI ${bmi.toFixed(1)}` : "estimated"} />
        <MiniStat label="Model" value={recommendedSize ?? "--"} helper={typeof confidenceScore === "number" ? `${Math.round(confidenceScore * 100)}% ${modelLabel}` : modelLabel} />
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-3">
        <MiniRatioBar label="S/H" value={ratios?.shoulderToHip ?? 0} min={0.7} max={1.6} baseline={ratios?.source === "world_3d" ? 1.1 : 1.5} />
        <MiniRatioBar label="W/H" value={ratios?.waistToHip ?? 0} min={0.5} max={1.2} baseline={ratios?.source === "world_3d" ? 0.8 : 0.95} />
        <MiniRatioBar label="S/T" value={ratios?.shoulderToTorso ?? 0} min={0.3} max={1.5} baseline={0.7} />
      </div>
    </div>
  );
}

function MiniStat({ label, value, helper }: { label: string; value: string; helper: string }) {
  return (
    <div className="rounded-2xl border border-border bg-muted/40 px-3 py-2">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 truncate font-mono text-sm font-bold">{value}</p>
      <p className="mt-0.5 truncate text-[10px] text-muted-foreground">{helper}</p>
    </div>
  );
}

function MiniRatioBar({
  label,
  value,
  min,
  max,
  baseline,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  baseline: number;
}) {
  const range = max - min;
  const pct = range > 0 ? Math.max(0, Math.min(100, ((value - min) / range) * 100)) : 0;
  const baselinePct = range > 0 ? Math.max(0, Math.min(100, ((baseline - min) / range) * 100)) : 50;

  return (
    <div className="rounded-2xl border border-border bg-muted/40 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
        <span className="font-mono text-xs font-bold">{value > 0 ? value.toFixed(2) : "--"}</span>
      </div>
      <div className="relative mt-2 h-1.5 rounded-full bg-muted">
        <div className="absolute top-0 h-full w-px bg-muted-foreground/50" style={{ left: `${baselinePct}%` }} />
        {value > 0 && (
          <div className="absolute left-0 top-0 h-full rounded-full bg-foreground/75" style={{ width: `${pct}%` }} />
        )}
      </div>
    </div>
  );
}

function StyleRecommendationsCard({
  result,
  loading,
  error,
  occasion,
  predictedSize,
  styleSliders,
  styleLimit,
  canRefresh,
  hasActiveStyleSliders,
  onOccasionChange,
  onStyleSliderChange,
  onStyleLimitChange,
  onResetStyleSliders,
  onRefresh,
}: {
  result: UpperStyleRecommendationResult | null;
  loading: boolean;
  error: string | null;
  occasion: UpperStyleOccasion;
  predictedSize: UpperStyleSize | null;
  styleSliders: StyleSliders;
  styleLimit: number;
  canRefresh: boolean;
  hasActiveStyleSliders: boolean;
  onOccasionChange: (value: string) => void;
  onStyleSliderChange: (axis: StyleSliderAxis, value: number) => void;
  onStyleLimitChange: (value: string) => void;
  onResetStyleSliders: () => void;
  onRefresh: () => void;
}) {
  const bodyShape = result?.user_state.body_shape;
  const bodyType = result?.user_state.body_type_vn;
  const items = result?.items ?? [];

  // intent: keep recommendations usable first, with minimal neutral styling
  // status: done
  // next: verify scroll behavior against real desktop and mobile viewports
  // blockers: none
  // confidence: high
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-3xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-xl font-bold tracking-tight">Recommended tops</h2>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Fit-Flatter-Match ranking from the recommendation service.
          </p>
        </div>
        {loading ? (
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        ) : (
          <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            {predictedSize ? `Size ${predictedSize}` : "waiting"}
          </span>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
          Occasion: {occasion}
        </span>
        <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
          Showing up to {styleLimit}
        </span>
        {(bodyShape || bodyType) && (
          <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
            {bodyShape
              ? `${bodyShape.display_label} (${scorePct(bodyShape.confidence)})`
              : `${bodyType?.label_vi} (${scorePct(bodyType?.confidence ?? 0)})`}
          </span>
        )}
      </div>

      <details className="mt-3 shrink-0 rounded-2xl border border-border bg-muted/30 p-3">
        <summary className="cursor-pointer list-none text-sm font-semibold focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 [&::-webkit-details-marker]:hidden">
          <span className="flex items-center justify-between gap-3">
            <span>Style controls</span>
            <span className="text-xs font-normal text-muted-foreground">Adjust ranking</span>
          </span>
        </summary>

        <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-muted-foreground">
            Steer the ranking without clearing the last recommendations.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={onRefresh}
              disabled={!canRefresh || loading}
              className="rounded-xl border border-border bg-background px-3 py-2 text-xs font-semibold transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Refresh
            </button>
            <button
              type="button"
              onClick={onResetStyleSliders}
              disabled={!hasActiveStyleSliders || loading}
              className="rounded-xl border border-border bg-background px-3 py-2 text-xs font-semibold transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Neutral style
            </button>
          </div>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <SelectField
            label="Occasion"
            value={occasion}
            onChange={onOccasionChange}
            options={OCCASIONS}
          />
          <SelectField
            label="Result count"
            value={String(styleLimit)}
            onChange={onStyleLimitChange}
            options={STYLE_LIMIT_OPTIONS}
          />
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-4">
          {STYLE_SLIDERS.map((slider) => (
            <StylePreferenceSlider
              key={slider.axis}
              slider={slider}
              value={styleSliders[slider.axis]}
              onChange={onStyleSliderChange}
            />
          ))}
        </div>
      </details>

      {error && (
        <div className="mt-4 rounded-2xl border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      )}

      {!error && loading && items.length === 0 && (
        <div className="mt-4 space-y-3">
          {[1, 2, 3].map((row) => (
            <div key={row} className="h-24 animate-pulse rounded-2xl bg-muted" />
          ))}
        </div>
      )}

      {!error && !loading && items.length === 0 && (
        <div className="mt-4 rounded-2xl border border-dashed border-border bg-muted/40 p-5 text-sm text-muted-foreground">
          {result
            ? "No tops matched this body profile, occasion, and fit threshold. Try another occasion or run a new prediction."
            : "Run a visual size prediction to get style recommendations."}
        </div>
      )}

      {items.length > 0 && (
        <div className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain pr-1">
          {items.map((item) => {
            const garment = getRealGarmentForStyleItem(item.item_id);
            const imageUrl = garment ? getGarmentImage(garment) : item.image_url;
            const title = garment?.name ?? item.title;
            const metadata = [
              garment?.brand ?? item.brand,
              garment?.type ?? item.category,
              garment ? formatUsd(garment.price) : formatVnd(item.price_vnd),
            ].filter(Boolean).join(" - ");
            const recommendedSize = result?.user_state.predicted_size;
            const sizeAvailable = recommendedSize ? garment?.sizes.includes(recommendedSize) ?? false : false;

            return (
              <div key={item.item_id} className="rounded-2xl border border-border bg-muted/30 p-3 transition-colors hover:bg-muted/50">
                <div className="flex items-start gap-3">
                  <div className="relative h-20 w-16 shrink-0 overflow-hidden rounded-2xl border border-border bg-background">
                    {imageUrl && (
                      <NextImage
                        src={imageUrl}
                        alt={title}
                        fill
                        sizes="64px"
                        className="object-cover"
                      />
                    )}
                    <span className="absolute left-1.5 top-1.5 rounded-full bg-background/90 px-2 py-1 text-[10px] font-semibold text-foreground backdrop-blur">
                      #{item.rank}
                    </span>
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        {garment ? (
                          <Link
                            href={`/product/${garment.id}`}
                            className="line-clamp-2 text-sm font-semibold transition-colors hover:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          >
                            {title}
                          </Link>
                        ) : (
                          <span className="line-clamp-2 text-sm font-semibold">{title}</span>
                        )}
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          {metadata}
                        </p>
                      </div>
                      <span className="shrink-0 rounded-full bg-foreground px-2.5 py-1 text-xs font-semibold text-background">
                        {scorePct(item.total_score)}
                      </span>
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2">
                      <StyleScorePill label="Fit" value={item.explanation.fit.score} />
                      <StyleScorePill label="Flatter" value={item.explanation.flatter.score} />
                      <StyleScorePill label="Match" value={item.explanation.match.score} />
                      {recommendedSize && garment && (
                        <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${
                          sizeAvailable
                            ? "border-border bg-background text-foreground"
                            : "border-border bg-muted text-muted-foreground"
                        }`}>
                          {sizeAvailable ? `Has ${recommendedSize}` : `Nearest to ${recommendedSize}`}
                        </span>
                      )}
                    </div>

                    <p className="mt-2 line-clamp-1 text-xs leading-relaxed text-muted-foreground">
                      {garment?.description ??
                        "Ranked from the real catalog by fit, body-shape flattery, occasion, and style preferences."}
                    </p>
                    <p className="mt-1 line-clamp-1 text-xs leading-relaxed text-muted-foreground">
                      {item.explanation.fit.overall_verdict}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StylePreferenceSlider({
  slider,
  value,
  onChange,
}: {
  slider: (typeof STYLE_SLIDERS)[number];
  value: number;
  onChange: (axis: StyleSliderAxis, value: number) => void;
}) {
  return (
    <label className="rounded-2xl border border-border bg-background p-2.5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-medium">{slider.label}</span>
        <span className="rounded-full bg-muted px-2 py-1 font-mono text-[10px] text-muted-foreground">
          {Math.abs(value) < 0.05 ? "Neutral" : value > 0 ? `+${value.toFixed(2)}` : value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min="-1"
        max="1"
        step="0.25"
        value={value}
        onChange={(event) => onChange(slider.axis, Number.parseFloat(event.target.value))}
        className="mt-2 w-full accent-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        aria-label={`${slider.label} preference`}
      />
      <div className="mt-0.5 flex justify-between gap-3 text-[10px] text-muted-foreground">
        <span>{slider.minLabel}</span>
        <span>{slider.maxLabel}</span>
      </div>
    </label>
  );
}

function StyleScorePill({ label, value }: { label: string; value: number }) {
  return (
    <span className="rounded-full border border-border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
      {label} {scorePct(value)}
    </span>
  );
}

function MetricCard({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-muted/40 p-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1.5 truncate text-lg font-semibold tracking-tight">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
    </div>
  );
}
