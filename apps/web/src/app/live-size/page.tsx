"use client";

import type { InputHTMLAttributes } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  Square,
  Upload,
  User,
  Video,
  X,
} from "lucide-react";
import {
  getVisualSizeRecommendation,
  type VisualSizeResult,
} from "@/lib/api";
import { usePoseDetector, type DrawOptions } from "@/hooks/use-pose-detector";
import { useBodyRatios, extractBodyRatios, estimateWeightFromRatios, type BodyRatios } from "@/hooks/use-body-ratios";
import { detectPoseFromImage, drawImageSkeleton } from "@/hooks/use-image-pose";

type Population = "universal" | "vietnamese";
type InputMode = "webcam" | "upload";

const POPULATIONS: Array<{ value: Population; label: string }> = [
  { value: "vietnamese", label: "Vietnamese" },
  { value: "universal", label: "Universal (US)" },
];

const CATEGORIES = [
  { value: "dress", label: "Dress" },
  { value: "top", label: "Top" },
  { value: "shirt", label: "Shirt" },
  { value: "outerwear", label: "Outerwear" },
  { value: "bottom", label: "Bottom" },
];

const LIVE_INTERVAL_MS = 3000;

export default function LiveSizePage() {
  const [inputMode, setInputMode] = useState<InputMode>("webcam");
  const [heightCm, setHeightCm] = useState("165");
  const [population, setPopulation] = useState<Population>("vietnamese");
  const [category, setCategory] = useState("dress");

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveRunning, setLiveRunning] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [visualResult, setVisualResult] = useState<VisualSizeResult | null>(null);
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
    showPoints: true,
    showFaceMesh: true,
    showBoundingBox: true,
    showLabels: false,
    showConfidence: false,
    mirror: true,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);

  const { landmarks, worldLandmarks, debug: poseDebug, drawSkeleton } = usePoseDetector(
    videoRef,
    cameraActive
  );

  // Extract body shape ratios — prefers world landmarks (3D meters) over 2D image coords
  const webcamRatios = useBodyRatios(landmarks, worldLandmarks);
  // Use upload ratios when in upload mode, webcam ratios otherwise
  const bodyRatios = inputMode === "upload" ? uploadRatios : webcamRatios;

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

  const currentModel = visualResult?.research_model.model ?? "gbm_copula_tempered_a075";
  const confidenceScore = visualResult?.research_model.confidence_score;
  const recommendedSize = visualResult?.research_model.predicted_size;
  const confidenceLabel = visualResult?.research_model.confidence;
  const estimatedWeight = visualResult?.weight_estimation?.estimated_weight_kg;
  const bodyShape = visualResult?.body_shape?.type;

  const liveStatus = useMemo(() => {
    if (processing) return "Đang phân tích frame mới";
    if (liveRunning) return `Đang cập nhật mỗi ${LIVE_INTERVAL_MS / 1000} giây`;
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
    setLiveRunning(false);
    setCameraActive(false);
  }, []);

  // ── Image upload handlers ──

  const processUploadedImage = useCallback(async (file: File) => {
    setError(null);
    setProcessing(true);
    setUploadRatios(null);
    setVisualResult(null);

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
        setError("No pose detected — make sure the full body is visible in the photo.");
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
        setError("Partial body detected — need full body (shoulders, hips, ankles) visible.");
        setProcessing(false);
        return;
      }

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
        category,
        population,
        confidence: ratios.confidence,
        source: ratios.source,
      });

      setVisualResult(result);
      setRoundTripMs(Math.round(performance.now() - startedAt));
      setLastUpdatedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image processing failed");
    } finally {
      setProcessing(false);
    }
  }, [heightCm, category, population]);

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
    setVisualResult(null);
    setError(null);
  }, [uploadedImage]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setError(null);

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
      setError("Need full body visible — step back so shoulders, hips, and ankles are in frame.");
      return;
    }

    setError(null);
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
        category,
        population,
        confidence: bodyRatios.confidence,
        source: bodyRatios.source,
      });

      setVisualResult(result);
      setRoundTripMs(Math.round(performance.now() - startedAt));
      setLastUpdatedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Visual inference failed");
    } finally {
      busyRef.current = false;
      setProcessing(false);
    }
  }, [category, heightCm, population, bodyRatios]);

  useEffect(() => {
    if (!cameraActive || !liveRunning) return;

    void runInference();
    const intervalId = window.setInterval(() => {
      void runInference();
    }, LIVE_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [cameraActive, liveRunning, runInference]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 md:px-6">
      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-muted-foreground">
            Visual Size Estimation
          </p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-5xl">
            Size Recommendation
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-6 text-muted-foreground md:text-base">
            Dùng webcam hoặc upload ảnh để trích xuất body skeleton, tính tỷ lệ cơ thể,
            rồi gợi ý size phù hợp. Chỉ cần chiều cao — cân nặng được ước lượng từ tỷ lệ.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode toggle */}
          <div className="inline-flex rounded-xl border border-border bg-muted p-1">
            <button
              onClick={() => setInputMode("webcam")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
                inputMode === "webcam"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Camera className="h-3.5 w-3.5" />
              Webcam
            </button>
            <button
              onClick={() => setInputMode("upload")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
                inputMode === "upload"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" />
              Upload
            </button>
          </div>
          <Link
            href="/measure"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent"
          >
            <Ruler className="h-4 w-4" />
            Manual
          </Link>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-3xl border border-border bg-card p-5 shadow-sm md:p-6">
          <div className="grid gap-4 md:grid-cols-3">
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
            <SelectField
              label="Category"
              value={category}
              onChange={setCategory}
              options={CATEGORIES}
            />
          </div>
          {bodyRatios && (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className={`rounded-full px-2.5 py-1 font-mono ${bodyRatios.reliable ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" : "bg-amber-500/15 text-amber-600 dark:text-amber-400"}`}>
                {bodyRatios.reliable ? "Full body detected" : `Partial (${bodyRatios.landmarksUsed}/12)`}
              </span>
              <span className={`rounded-full px-2.5 py-1 font-mono ${bodyRatios.source === "world_3d" ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" : "bg-amber-500/15 text-amber-600 dark:text-amber-400"}`}>
                {bodyRatios.source === "world_3d" ? "3D world" : "2D image"}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                S/H {bodyRatios.shoulderToHip.toFixed(2)}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                W/H {bodyRatios.waistToHip.toFixed(2)}
              </span>
              <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-muted-foreground">
                T/L {bodyRatios.torsoToLeg.toFixed(2)}
              </span>
              {estimatedWeight && (
                <span className="rounded-full bg-blue-500/15 px-2.5 py-1 font-mono text-blue-600 dark:text-blue-400">
                  ~{estimatedWeight}kg
                </span>
              )}
              {bodyShape && (
                <span className="rounded-full bg-purple-500/15 px-2.5 py-1 font-mono text-purple-600 dark:text-purple-400">
                  {bodyShape}
                </span>
              )}
            </div>
          )}

          {/* ── Webcam mode ── */}
          {inputMode === "webcam" && (
            <>
              <div className="mt-6 flex flex-wrap gap-3">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground shadow-sm transition-opacity hover:opacity-90"
                  >
                    <Camera className="h-4 w-4" />
                    Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      onClick={() => setLiveRunning(true)}
                      disabled={liveRunning}
                      className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Play className="h-4 w-4" />
                      Start Live
                    </button>
                    <button
                      onClick={() => setLiveRunning(false)}
                      disabled={!liveRunning}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-3 text-sm font-semibold shadow-sm transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Square className="h-4 w-4" />
                      Stop
                    </button>
                    <button
                      onClick={() => void runInference()}
                      disabled={processing}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-3 text-sm font-semibold shadow-sm transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {processing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Video className="h-4 w-4" />}
                      Once
                    </button>
                    <button
                      onClick={stopCamera}
                      className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-3 text-sm font-semibold shadow-sm transition-colors hover:bg-accent"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </button>
                  </>
                )}
              </div>

              <div className="mt-6 overflow-hidden rounded-3xl border border-border bg-muted/40">
                <div className="relative aspect-[4/3]">
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
                      <div className="absolute inset-x-[18%] top-[8%] bottom-[8%] rounded-[28px] border-2 border-dashed border-white/45" />
                      <p className="absolute bottom-4 left-4 right-4 rounded-full bg-black/50 px-3 py-2 text-center text-xs font-medium text-white">
                        Đứng chính diện, thấy rõ toàn thân, giữ máy ổn định 2–3 giây.
                      </p>
                    </div>
                  )}
                  {cameraActive && showDebug && (
                    <div className="pointer-events-none absolute top-3 left-3 rounded-xl bg-black/70 px-3 py-2 font-mono text-[11px] leading-relaxed text-green-400 backdrop-blur-sm">
                      <p>FPS: {poseDebug.fps}</p>
                      <p>Detect: {poseDebug.detectMs}ms</p>
                      <p>Landmarks: {poseDebug.landmarkCount}/33</p>
                      <p>Confidence: {(poseDebug.avgConfidence * 100).toFixed(0)}%</p>
                      <p>Model: {poseDebug.modelLoaded ? "loaded" : "loading..."}</p>
                      {poseDebug.error && <p className="text-red-400">Error: {poseDebug.error}</p>}
                      {roundTripMs !== null && <p className="text-blue-400">API RTT: {roundTripMs}ms</p>}
                    </div>
                  )}
                </div>
              </div>

              {cameraActive && (
                <div className="mt-4 flex flex-wrap gap-2">
                  <ToggleButton active={showDebug} onClick={() => setShowDebug((v) => !v)} icon={<Bug className="h-3.5 w-3.5" />} label="Debug HUD" />
                  <ToggleButton active={drawOpts.showSkeleton ?? true} onClick={() => setDrawOpts((o) => ({ ...o, showSkeleton: !o.showSkeleton }))} icon={<Eye className="h-3.5 w-3.5" />} label="Skeleton" />
                  <ToggleButton active={drawOpts.showPoints ?? true} onClick={() => setDrawOpts((o) => ({ ...o, showPoints: !o.showPoints }))} label="Points" />
                  <ToggleButton active={drawOpts.showFaceMesh ?? true} onClick={() => setDrawOpts((o) => ({ ...o, showFaceMesh: !o.showFaceMesh }))} label="Face" />
                  <ToggleButton active={drawOpts.showBoundingBox ?? true} onClick={() => setDrawOpts((o) => ({ ...o, showBoundingBox: !o.showBoundingBox }))} label="BBox" />
                  <ToggleButton active={drawOpts.showLabels ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showLabels: !o.showLabels }))} label="Labels" />
                  <ToggleButton active={drawOpts.showConfidence ?? false} onClick={() => setDrawOpts((o) => ({ ...o, showConfidence: !o.showConfidence }))} label="Conf %" />
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
                      <p className="text-sm font-medium">Drop a full-body photo here</p>
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
          <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            {inputMode === "webcam" && (
              <span className="rounded-full border border-border bg-muted px-3 py-1.5">
                {liveStatus}
              </span>
            )}
            {inputMode === "upload" && uploadRatios && (
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-emerald-600 dark:text-emerald-400">
                Pose detected ({uploadRatios.landmarksUsed}/12 landmarks)
              </span>
            )}
            {poseDebug.modelLoaded && inputMode === "webcam" && (
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-emerald-600 dark:text-emerald-400">
                Pose model loaded
              </span>
            )}
            {roundTripMs !== null && (
              <span className="rounded-full border border-border bg-muted px-3 py-1.5">
                Round trip {roundTripMs} ms
              </span>
            )}
            {lastUpdatedAt && (
              <span className="rounded-full border border-border bg-muted px-3 py-1.5">
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

        <section className="space-y-6">
          {/* Size prediction hero */}
          <div className="rounded-3xl border border-border bg-card p-6 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Predicted size</p>
                <div className="mt-2 flex items-end gap-3">
                  <span className="text-6xl font-black tracking-tight">
                    {recommendedSize ?? "--"}
                  </span>
                  <span
                    className={`mb-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
                      confidenceLabel === "high"
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                        : confidenceLabel === "medium"
                          ? "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
                          : "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
                    }`}
                  >
                    {confidenceLabel ?? "waiting"}
                  </span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1.5">
                {processing && <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />}
                {bodyRatios && (
                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${
                    bodyRatios.source === "world_3d"
                      ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                      : "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                  }`}>
                    {bodyRatios.source === "world_3d" ? "3D world" : "2D fallback"}
                  </span>
                )}
              </div>
            </div>

            {/* Probability distribution bar */}
            {visualResult?.research_model?.probabilities && (
              <div className="mt-5">
                <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Size probability distribution
                </p>
                <div className="flex h-8 overflow-hidden rounded-xl">
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
            <div className="mt-5 grid gap-3 sm:grid-cols-4">
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
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
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

          {/* Body ratios from skeleton */}
          <div className="rounded-3xl border border-border bg-card p-6 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold tracking-tight">Body ratios</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {bodyRatios?.source === "world_3d"
                    ? "3D world landmarks (meters) — perspective-corrected, matches real anthropometric data"
                    : "2D image landmarks — affected by camera perspective, dampened calibration applied"}
                </p>
              </div>
              <div className="flex flex-col items-end gap-1 text-right">
                {bodyRatios && (
                  <>
                    <span className="font-mono text-xs text-muted-foreground">
                      {bodyRatios.landmarksUsed}/12 landmarks
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {(bodyRatios.confidence * 100).toFixed(0)}% avg conf
                    </span>
                  </>
                )}
              </div>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <RatioBar
                label="Shoulder / Hip"
                value={bodyRatios?.shoulderToHip ?? 0}
                min={0.7}
                max={1.6}
                baseline={bodyRatios?.source === "world_3d" ? 1.1 : 1.5}
                helper="V-shape vs pear"
              />
              <RatioBar
                label="Waist / Hip"
                value={bodyRatios?.waistToHip ?? 0}
                min={0.5}
                max={1.2}
                baseline={bodyRatios?.source === "world_3d" ? 0.8 : 0.95}
                helper="Waist taper"
              />
              <RatioBar
                label="Shoulder / Torso"
                value={bodyRatios?.shoulderToTorso ?? 0}
                min={0.3}
                max={1.5}
                baseline={0.7}
                helper="Relative breadth"
              />
              <RatioBar
                label="Torso / Leg"
                value={bodyRatios?.torsoToLeg ?? 0}
                min={0.2}
                max={0.8}
                baseline={0.45}
                helper="Proportion"
              />
              <RatioBar
                label="Arm / Torso"
                value={bodyRatios?.armToTorso ?? 0}
                min={0.5}
                max={2.0}
                baseline={1.1}
                helper="Relative arm length"
              />
              <MetricCard
                label="Reliable"
                value={bodyRatios?.reliable ? "Yes" : "No"}
                helper={bodyRatios?.reliable ? "Full body visible" : "Step back for full body"}
              />
            </div>
          </div>

          {/* Data pipeline summary */}
          {visualResult && (
            <div className="rounded-3xl border border-border bg-card p-6 shadow-sm">
              <h2 className="text-xl font-bold tracking-tight">Pipeline data</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Full input/output trace for this prediction
              </p>
              <div className="mt-4 space-y-3">
                <PipelineRow
                  step="1"
                  label="Pose detection"
                  detail={`MediaPipe PoseLandmarker (${bodyRatios?.source === "world_3d" ? "world 3D" : "image 2D"})`}
                  value={`${poseDebug.landmarkCount}/33 landmarks, ${poseDebug.detectMs}ms`}
                />
                <PipelineRow
                  step="2"
                  label="Ratio extraction"
                  detail="Client-side, no server call"
                  value={bodyRatios ? `S/H=${bodyRatios.shoulderToHip.toFixed(2)}, W/H=${bodyRatios.waistToHip.toFixed(2)}` : "--"}
                />
                <PipelineRow
                  step="3"
                  label="Weight estimation"
                  detail={`BMI baseline 21.5, ${bodyRatios?.source === "world_3d" ? "anthropometric" : "2D-calibrated"} coefficients`}
                  value={`${estimatedWeight ?? "--"}kg (BMI ${visualResult.weight_estimation?.estimated_bmi?.toFixed(1) ?? "--"})`}
                />
                <PipelineRow
                  step="4"
                  label="GBM prediction"
                  detail={currentModel}
                  value={`${recommendedSize ?? "--"} (${typeof confidenceScore === "number" ? (confidenceScore * 100).toFixed(1) + "%" : "--"})`}
                />
                {roundTripMs !== null && (
                  <PipelineRow
                    step="5"
                    label="Round trip"
                    detail="Frontend → API gateway → recommendation service"
                    value={`${roundTripMs}ms`}
                  />
                )}
              </div>
            </div>
          )}
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
      <span className="mb-1 block text-sm font-medium text-muted-foreground">{label}</span>
      <input
        {...inputProps}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm outline-none transition-colors focus:border-foreground"
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
      <span className="mb-1 block text-sm font-medium text-muted-foreground">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm outline-none transition-colors focus:border-foreground"
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
          ? "border border-emerald-500/40 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
          : "border border-border bg-muted text-muted-foreground hover:bg-accent"
      }`}
    >
      {icon}
      {label}
    </button>
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
    <div className="rounded-2xl border border-border bg-muted/40 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-2 text-xl font-semibold tracking-tight">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
    </div>
  );
}

function RatioBar({
  label,
  value,
  min,
  max,
  baseline,
  helper,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  baseline: number;
  helper: string;
}) {
  const range = max - min;
  const pct = range > 0 ? Math.max(0, Math.min(100, ((value - min) / range) * 100)) : 0;
  const baselinePct = range > 0 ? Math.max(0, Math.min(100, ((baseline - min) / range) * 100)) : 50;

  return (
    <div className="rounded-2xl border border-border bg-muted/40 p-4">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <p className="font-mono text-sm font-semibold">{value > 0 ? value.toFixed(3) : "--"}</p>
      </div>
      <div className="relative mt-2.5 h-2 rounded-full bg-muted">
        {/* Baseline marker */}
        <div
          className="absolute top-0 h-full w-px bg-muted-foreground/40"
          style={{ left: `${baselinePct}%` }}
        />
        {/* Value fill */}
        {value > 0 && (
          <div
            className="absolute top-0 left-0 h-full rounded-full bg-foreground/70 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <p className="mt-1.5 text-[10px] text-muted-foreground">{helper}</p>
    </div>
  );
}

function PipelineRow({
  step,
  label,
  detail,
  value,
}: {
  step: string;
  label: string;
  detail: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-border bg-muted/40 p-3.5">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-foreground/10 text-[10px] font-bold text-muted-foreground">
        {step}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <p className="text-sm font-medium">{label}</p>
          <p className="shrink-0 font-mono text-xs font-medium">{value}</p>
        </div>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}
