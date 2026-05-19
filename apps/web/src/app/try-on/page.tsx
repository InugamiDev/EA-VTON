"use client";

import { useState, useRef, useCallback, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import {
  Camera,
  Upload,
  Check,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Download,
  Shirt,
  Sparkles,
  AlertTriangle,
  Loader2,
  X,
  ShieldCheck,
  CheckCircle,
  XCircle,
  AlertCircle,
  Cpu,
  Clock,
  Gauge,
} from "lucide-react";
import { cn, formatPrice } from "@/lib/utils";
import { GARMENTS } from "@/data/garments";
import type { Garment } from "@/types";
import {
  uploadPhoto,
  createTryOnJob,
  pollTryOnJob,
  getResultImageUrl,
  getUploadedPhotoUrl,
  checkHealth,
  type QualityAssessment,
  type TryOnJob,
} from "@/lib/api";

const STEPS = ["Photo", "Garment", "Result"] as const;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const POLL_INTERVAL = 2000;

/* ------------------------------------------------------------------ */
/*  Quality indicator component                                        */
/* ------------------------------------------------------------------ */

function QualityIndicator({ quality }: { quality: QualityAssessment }) {
  const items = [
    { label: "Brightness", check: quality.brightness },
    { label: "Sharpness", check: quality.blur },
    { label: "Resolution", check: quality.dimensions },
  ];

  const Icon = {
    pass: CheckCircle,
    warn: AlertCircle,
    fail: XCircle,
  };

  const color = {
    pass: "text-emerald-600 dark:text-emerald-400",
    warn: "text-amber-600 dark:text-amber-400",
    fail: "text-red-600 dark:text-red-400",
  };

  return (
    <div className="mx-auto max-w-sm rounded-2xl border border-border bg-card p-5 shadow-sm">
      <p className="mb-3 text-sm font-semibold">Photo Quality Assessment</p>
      <div className="space-y-2">
        {items.map((item) => {
          const StatusIcon = Icon[item.check.status];
          return (
            <div key={item.label} className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{item.label}</span>
              <span className={cn("flex items-center gap-1.5 font-medium", color[item.check.status])}>
                <StatusIcon className="h-4 w-4" />
                {item.check.status === "pass" ? "Good" : item.check.status === "warn" ? "Fair" : "Poor"}
              </span>
            </div>
          );
        })}
        <div className="mt-2 border-t border-border pt-2">
          <div className="flex items-center justify-between text-sm font-semibold">
            <span>Overall</span>
            <span className={cn(color[quality.overall])}>
              {quality.overall === "pass" ? "Ready" : quality.overall === "warn" ? "Acceptable" : "Poor — retake recommended"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main wizard                                                        */
/* ------------------------------------------------------------------ */

function TryOnWizard() {
  const searchParams = useSearchParams();
  const preselectedId = searchParams.get("garment");

  // Wizard state
  const [step, setStep] = useState(0);
  const [selectedGarment, setSelectedGarment] = useState<Garment | null>(
    preselectedId ? GARMENTS.find((g) => g.id === preselectedId) ?? null : null,
  );
  const [consentChecked, setConsentChecked] = useState(false);

  // Photo state
  const [userPhotoPreview, setUserPhotoPreview] = useState<string | null>(null);
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [photoFilename, setPhotoFilename] = useState<string | null>(null);
  const [quality, setQuality] = useState<QualityAssessment | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // Try-on state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<TryOnJob | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Backend info
  const [inferenceMode, setInferenceMode] = useState<string | null>(null);

  // Camera state
  const [cameraActive, setCameraActive] = useState(false);
  const [faceStatus, setFaceStatus] = useState<
    "loading" | "no-face" | "too-small" | "too-close" | "off-center" | "multiple" | "ready"
  >("loading");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const faceDetectorRef = useRef<unknown>(null);
  const rafRef = useRef<number>(0);

  // Check backend on mount
  useEffect(() => {
    checkHealth()
      .then((h) => setInferenceMode(h.inference_mode))
      .catch(() => setInferenceMode("unknown"));
  }, []);

  // Camera helpers
  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    faceDetectorRef.current = null;
    setCameraActive(false);
    setFaceStatus("loading");
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch {
      alert("Could not access camera. Please allow camera permissions or use the upload option.");
    }
  }, []);

  // Attach stream + start lightweight face detection once video mounts
  useEffect(() => {
    if (!cameraActive || !videoRef.current || !streamRef.current) return;
    const video = videoRef.current;
    video.srcObject = streamRef.current;

    let cancelled = false;
    type Detection = { boundingBox: { originX: number; originY: number; width: number; height: number }; keypoints: Array<{ x: number; y: number }> };
    let detector: { detectForVideo: (v: HTMLVideoElement, t: number) => { detections: Detection[] } } | null = null;

    async function initFaceDetection() {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const { FaceDetector, FilesetResolver } = vision;
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const fd = await FaceDetector.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.5,
        });
        if (!cancelled) {
          detector = fd as unknown as typeof detector;
          faceDetectorRef.current = fd;
        }
      } catch {
        // Face detection unavailable — degrade gracefully
      }
    }

    function drawRoundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, radius: number) {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function detectLoop() {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (canvas && video.readyState >= 2) {
        const ctx = canvas.getContext("2d");
        const container = canvas.parentElement!;
        const dispW = container.clientWidth;
        const dispH = container.clientHeight;
        canvas.width = dispW;
        canvas.height = dispH;

        // object-cover transform
        const vidW = video.videoWidth;
        const vidH = video.videoHeight;
        const scale = Math.max(dispW / vidW, dispH / vidH);
        const scaledW = vidW * scale;
        const scaledH = vidH * scale;
        const offsetX = (dispW - scaledW) / 2;
        const offsetY = (dispH - scaledH) / 2;

        if (ctx) {
          ctx.clearRect(0, 0, dispW, dispH);

          if (!detector) {
            // Still loading model — draw subtle pulsing guide
            setFaceStatus("loading");
          } else {
            try {
              const result = detector.detectForVideo(video, performance.now());
              const dets = result.detections;

              if (dets.length === 0) {
                // No face found
                setFaceStatus("no-face");

                // Draw a subtle dashed guide oval in center
                ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([8, 6]);
                ctx.beginPath();
                ctx.ellipse(dispW / 2, dispH * 0.35, dispW * 0.22, dispH * 0.18, 0, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
              } else if (dets.length > 1) {
                // Multiple faces — highlight all with amber, use the largest
                setFaceStatus("multiple");
                for (const det of dets) {
                  const bb = det.boundingBox;
                  const x = offsetX + bb.originX * scale;
                  const y = offsetY + bb.originY * scale;
                  const w = bb.width * scale;
                  const h = bb.height * scale;
                  ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
                  ctx.lineWidth = 1.5;
                  drawRoundedRect(ctx, x, y, w, h, Math.min(w, h) * 0.2);
                  ctx.stroke();
                }
              } else {
                // Single face — run quality checks
                const det = dets[0];
                const bb = det.boundingBox;

                const x = offsetX + bb.originX * scale;
                const y = offsetY + bb.originY * scale;
                const w = bb.width * scale;
                const h = bb.height * scale;

                // Face size relative to display
                const faceArea = (w * h) / (dispW * dispH);
                const faceCenterX = x + w / 2;
                const faceCenterY = y + h / 2;
                const offCenterX = Math.abs(faceCenterX - dispW / 2) / dispW;
                const offCenterY = Math.abs(faceCenterY - dispH * 0.4) / dispH;

                let status: typeof faceStatus;
                let boxColor: string;

                if (faceArea < 0.02) {
                  // Face too small (< 2% of frame) — too far away
                  status = "too-small";
                  boxColor = "rgba(245, 158, 11, 0.7)";
                } else if (faceArea > 0.35) {
                  // Face too large (> 35% of frame) — too close
                  status = "too-close";
                  boxColor = "rgba(245, 158, 11, 0.7)";
                } else if (offCenterX > 0.25 || offCenterY > 0.3) {
                  // Face too far from center
                  status = "off-center";
                  boxColor = "rgba(245, 158, 11, 0.7)";
                } else {
                  status = "ready";
                  boxColor = "rgba(34, 197, 94, 0.8)";
                }

                setFaceStatus(status);

                // Draw face bounding box with padding
                const pad = w * 0.12;
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = status === "ready" ? 2.5 : 1.5;
                const rx = x - pad;
                const ry = y - pad;
                const rw = w + pad * 2;
                const rh = h + pad * 2;
                drawRoundedRect(ctx, rx, ry, rw, rh, Math.min(rw, rh) * 0.25);
                ctx.stroke();

                // Draw keypoints
                if (det.keypoints) {
                  ctx.fillStyle = boxColor;
                  for (const kp of det.keypoints) {
                    const kx = offsetX + kp.x * scaledW;
                    const ky = offsetY + kp.y * scaledH;
                    ctx.beginPath();
                    ctx.arc(kx, ky, status === "ready" ? 3 : 2, 0, Math.PI * 2);
                    ctx.fill();
                  }
                }

                // Corner accents when ready
                if (status === "ready") {
                  const accentLen = Math.min(rw, rh) * 0.15;
                  ctx.strokeStyle = "rgba(34, 197, 94, 0.9)";
                  ctx.lineWidth = 3;
                  // Top-left
                  ctx.beginPath(); ctx.moveTo(rx, ry + accentLen); ctx.lineTo(rx, ry); ctx.lineTo(rx + accentLen, ry); ctx.stroke();
                  // Top-right
                  ctx.beginPath(); ctx.moveTo(rx + rw - accentLen, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + accentLen); ctx.stroke();
                  // Bottom-left
                  ctx.beginPath(); ctx.moveTo(rx, ry + rh - accentLen); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx + accentLen, ry + rh); ctx.stroke();
                  // Bottom-right
                  ctx.beginPath(); ctx.moveTo(rx + rw - accentLen, ry + rh); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw, ry + rh - accentLen); ctx.stroke();
                }
              }
            } catch {
              // Skip frame on detection error
            }
          }
        }
      }
      rafRef.current = requestAnimationFrame(detectLoop);
    }

    initFaceDetection();
    video.addEventListener("playing", () => {
      if (!cancelled) detectLoop();
    }, { once: true });

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [cameraActive]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  /* ---- Upload photo to backend ---- */
  const handleUploadToBackend = useCallback(async (blob: Blob, previewUrl: string) => {
    setUploading(true);
    setUploadError(null);
    setUserPhotoPreview(previewUrl);

    try {
      const result = await uploadPhoto(blob instanceof File ? blob : new File([blob], "capture.jpg", { type: "image/jpeg" }));
      setPhotoId(result.photo_id);
      setPhotoFilename(result.filename);
      setQuality(result.quality);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setUploadError(msg);
      setUserPhotoPreview(null);
    } finally {
      setUploading(false);
    }
  }, []);

  /* ---- Camera capture ---- */
  const capturePhoto = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    stopCamera();

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);

    // Convert to blob for upload
    const resp = await fetch(dataUrl);
    const blob = await resp.blob();
    await handleUploadToBackend(blob, dataUrl);
  }, [stopCamera, handleUploadToBackend]);

  /* ---- File upload ---- */
  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_SIZE) {
      setUploadError("File too large. Please choose an image under 10 MB.");
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    await handleUploadToBackend(file, previewUrl);
  }, [handleUploadToBackend]);

  /* ---- Retake ---- */
  const retakePhoto = () => {
    setUserPhotoPreview(null);
    setPhotoId(null);
    setPhotoFilename(null);
    setQuality(null);
    setUploadError(null);
  };

  /* ---- Start try-on processing ---- */
  const startProcessing = useCallback(async () => {
    if (!photoId || !selectedGarment) return;

    setProcessing(true);
    setProcessingError(null);
    setElapsedSeconds(0);
    setStep(2);

    // Elapsed time counter
    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    try {
      // Create job
      const job = await createTryOnJob(photoId, selectedGarment.id);
      setJobId(job.job_id);

      // Poll for completion
      pollingRef.current = setInterval(async () => {
        try {
          const status = await pollTryOnJob(job.job_id);

          if (status.status === "completed") {
            if (pollingRef.current) clearInterval(pollingRef.current);
            if (timerRef.current) clearInterval(timerRef.current);
            setJobResult(status);
            setProcessing(false);
          } else if (status.status === "failed") {
            if (pollingRef.current) clearInterval(pollingRef.current);
            if (timerRef.current) clearInterval(timerRef.current);
            setProcessingError(status.error || "Try-on generation failed");
            setProcessing(false);
          }
        } catch (err: unknown) {
          // Keep polling on transient errors
          console.error("Poll error:", err);
        }
      }, POLL_INTERVAL);
    } catch (err: unknown) {
      if (timerRef.current) clearInterval(timerRef.current);
      const msg = err instanceof Error ? err.message : "Failed to start try-on";
      setProcessingError(msg);
      setProcessing(false);
    }
  }, [photoId, selectedGarment]);

  /* ---- Navigation ---- */
  const canAdvanceFromPhoto = !!photoId && !uploading;
  const canAdvanceFromGarment = !!selectedGarment && !!photoId && consentChecked;

  const goNext = () => {
    if (step === 0 && canAdvanceFromPhoto) setStep(1);
    if (step === 1 && canAdvanceFromGarment) startProcessing();
  };

  const goBack = () => {
    if (processing) return;
    if (step === 2) {
      setJobResult(null);
      setJobId(null);
      setProcessingError(null);
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    }
    setStep((s) => Math.max(0, s - 1));
  };

  const resetWizard = () => {
    retakePhoto();
    setSelectedGarment(null);
    setConsentChecked(false);
    setJobResult(null);
    setJobId(null);
    setProcessingError(null);
    setProcessing(false);
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    setStep(0);
  };

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <section className="py-16">
      <div className="mx-auto max-w-4xl px-6">
        {/* Backend status */}
        {inferenceMode && (
          <div className="mb-6 flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Cpu className="h-3.5 w-3.5" />
            Inference: <span className="font-medium text-foreground">{inferenceMode === "replicate" ? "Replicate IDM-VTON (real AI)" : "Local composite (demo)"}</span>
          </div>
        )}

        {/* Stepper */}
        <nav aria-label="Wizard steps" className="mb-12">
          <ol className="flex items-center justify-center gap-2 sm:gap-4">
            {STEPS.map((label, i) => {
              const isCurrent = i === step;
              const isDone = i < step || (i === 2 && !!jobResult);
              return (
                <li key={label} className="flex items-center gap-2 sm:gap-4">
                  {i > 0 && (
                    <div className={cn("hidden h-px w-8 sm:block sm:w-16", isDone ? "bg-primary" : "bg-border")} />
                  )}
                  <button
                    type="button"
                    disabled
                    className={cn(
                      "flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all duration-200",
                      isCurrent && "bg-primary text-primary-foreground shadow-sm",
                      isDone && !isCurrent && "bg-primary/10 text-primary",
                      !isCurrent && !isDone && "bg-muted text-muted-foreground",
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
                        isCurrent && "bg-primary-foreground text-primary",
                        isDone && !isCurrent && "bg-primary text-primary-foreground",
                        !isCurrent && !isDone && "bg-border text-muted-foreground",
                      )}
                    >
                      {isDone && !isCurrent ? <Check className="h-3.5 w-3.5" /> : i + 1}
                    </span>
                    <span className="hidden sm:inline">{label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>

        {/* ============================================================ */}
        {/*  Step 1: Photo                                                */}
        {/* ============================================================ */}
        {step === 0 && (
          <div className="space-y-8">
            <div className="text-center">
              <h1 className="text-3xl font-bold tracking-tight">Upload Your Photo</h1>
              <p className="mt-2 text-muted-foreground">
                Take a photo or upload one. We&apos;ll check quality before processing.
              </p>
            </div>

            {/* Choose method */}
            {!userPhotoPreview && !cameraActive && (
              <div className="grid gap-6 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={startCamera}
                  className="group flex flex-col items-center gap-4 rounded-2xl border border-border bg-card p-8 shadow-sm transition-all duration-200 hover:border-primary hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent transition-all duration-200 group-hover:bg-primary group-hover:text-primary-foreground">
                    <Camera className="h-7 w-7" />
                  </div>
                  <span className="text-lg font-semibold">Use Camera</span>
                  <span className="text-sm text-muted-foreground">Take a photo with your webcam</span>
                </button>

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="group flex flex-col items-center gap-4 rounded-2xl border border-border bg-card p-8 shadow-sm transition-all duration-200 hover:border-primary hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent transition-all duration-200 group-hover:bg-primary group-hover:text-primary-foreground">
                    <Upload className="h-7 w-7" />
                  </div>
                  <span className="text-lg font-semibold">Upload Photo</span>
                  <span className="text-sm text-muted-foreground">JPEG or PNG, max 10 MB</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </div>
            )}

            {/* Camera viewfinder */}
            {cameraActive && (
              <div className="space-y-4">
                <div className="relative mx-auto aspect-[3/4] max-w-sm overflow-hidden rounded-2xl border border-border shadow-sm">
                  <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" />
                  {/* Face detection canvas overlay */}
                  <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                  {/* Status hint */}
                  <div className="pointer-events-none absolute bottom-4 left-0 right-0 text-center">
                    <span className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium backdrop-blur-sm transition-colors duration-300",
                      faceStatus === "ready"
                        ? "bg-green-600/80 text-white"
                        : faceStatus === "loading"
                          ? "bg-black/40 text-white/70"
                          : "bg-amber-600/80 text-white"
                    )}>
                      {faceStatus === "loading" && "Initializing face detection..."}
                      {faceStatus === "no-face" && "No face detected — position your face in frame"}
                      {faceStatus === "too-small" && "Too far away — move closer to the camera"}
                      {faceStatus === "too-close" && "Too close — move back a little"}
                      {faceStatus === "off-center" && "Center your face in the frame"}
                      {faceStatus === "multiple" && "Multiple faces detected — only one person please"}
                      {faceStatus === "ready" && "Face detected — ready to capture"}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-center gap-4">
                  <button
                    type="button"
                    onClick={capturePhoto}
                    disabled={faceStatus !== "ready"}
                    className={cn(
                      "flex min-h-[2.625rem] items-center gap-2 rounded-lg px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                      faceStatus === "ready"
                        ? "bg-primary text-primary-foreground hover:opacity-90"
                        : "cursor-not-allowed bg-muted text-muted-foreground opacity-60"
                    )}
                  >
                    <Camera className="h-4 w-4" />
                    {faceStatus === "ready" ? "Capture" : "Waiting for face..."}
                  </button>
                  <button
                    type="button"
                    onClick={stopCamera}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <X className="h-4 w-4" />
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Uploading state */}
            {uploading && (
              <div className="flex flex-col items-center gap-4 py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="text-sm font-medium text-muted-foreground">
                  Uploading and analyzing photo quality...
                </p>
              </div>
            )}

            {/* Upload error */}
            {uploadError && (
              <div className="mx-auto flex max-w-sm items-start gap-3 rounded-2xl border border-red-300 bg-red-50 p-5 text-sm text-red-800 dark:border-red-700 dark:bg-red-950/30 dark:text-red-200">
                <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <p className="font-medium">Upload failed</p>
                  <p className="mt-1">{uploadError}</p>
                  <button
                    onClick={retakePhoto}
                    className="mt-3 text-sm font-medium underline transition-colors hover:text-red-900 dark:hover:text-red-100"
                  >
                    Try again
                  </button>
                </div>
              </div>
            )}

            {/* Photo preview + quality results */}
            {userPhotoPreview && !uploading && !uploadError && (
              <div className="space-y-6">
                <div className="relative mx-auto aspect-[3/4] max-w-sm overflow-hidden rounded-2xl border border-border shadow-sm">
                  <Image src={userPhotoPreview} alt="Your photo" fill className="object-cover" unoptimized />
                </div>

                {/* Quality assessment from backend */}
                {quality && <QualityIndicator quality={quality} />}

                {quality?.overall === "fail" && (
                  <div className="mx-auto flex max-w-sm items-start gap-3 rounded-2xl border border-amber-300 bg-amber-50 p-5 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-200">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <div>
                      <p className="font-medium">Low quality photo detected</p>
                      <p className="mt-1">The try-on result may be poor. Consider retaking with better lighting and a clearer camera.</p>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-center gap-4">
                  <button
                    type="button"
                    onClick={goNext}
                    disabled={!canAdvanceFromPhoto}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    Use This Photo
                    <ChevronRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={retakePhoto}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Retake
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/*  Step 2: Garment selection                                    */}
        {/* ============================================================ */}
        {step === 1 && (
          <div className="space-y-8">
            <div className="text-center">
              <h1 className="text-3xl font-bold tracking-tight">Choose a Garment</h1>
              <p className="mt-2 text-muted-foreground">Select the item you want to try on virtually.</p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {GARMENTS.map((g) => {
                const display = g.images.find((i) => i.type === "display");
                const isSelected = selectedGarment?.id === g.id;
                return (
                  <button
                    key={g.id}
                    type="button"
                    onClick={() => setSelectedGarment(g)}
                    className={cn(
                      "group relative overflow-hidden rounded-2xl border bg-card p-4 text-left shadow-sm transition-all duration-200 hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                      isSelected ? "border-primary ring-2 ring-primary" : "border-border hover:border-primary/50",
                    )}
                  >
                    {isSelected && (
                      <div className="absolute right-3 top-3 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Check className="h-3.5 w-3.5" />
                      </div>
                    )}
                    <div className="relative mb-3 aspect-[3/4] overflow-hidden rounded-lg">
                      <Image
                        src={display?.url ?? ""}
                        alt={g.name}
                        fill
                        className="object-cover transition-transform duration-200 group-hover:scale-105"
                        unoptimized
                      />
                    </div>
                    <p className="text-xs font-medium text-muted-foreground">{g.brand}</p>
                    <p className="mt-0.5 text-sm font-semibold leading-snug">{g.name}</p>
                    <p className="mt-1 text-sm font-semibold">{formatPrice(g.price)}</p>
                  </button>
                );
              })}
            </div>

            {/* Consent */}
            <label className="mx-auto flex max-w-xl cursor-pointer items-start gap-3 rounded-2xl border border-border bg-card p-6 shadow-sm transition-all duration-200 hover:bg-accent/50">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => setConsentChecked(e.target.checked)}
                className="mt-1 h-4 w-4 rounded border-border accent-primary"
              />
              <span className="text-sm leading-relaxed text-muted-foreground">
                <ShieldCheck className="mr-1 inline-block h-4 w-4 text-primary" />
                I consent to having my photo processed by AI for virtual try-on purposes.
                Photos are deleted after processing.
              </span>
            </label>

            {/* Nav */}
            <div className="flex items-center justify-between">
              <button
                type="button"
                onClick={goBack}
                className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <ChevronLeft className="h-4 w-4" />
                Back
              </button>
              <button
                type="button"
                onClick={goNext}
                disabled={!canAdvanceFromGarment}
                className={cn(
                  "flex min-h-[2.625rem] items-center gap-2 rounded-lg px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  canAdvanceFromGarment
                    ? "bg-primary text-primary-foreground hover:opacity-90"
                    : "cursor-not-allowed bg-muted text-muted-foreground",
                )}
              >
                Generate Try-On
                <Sparkles className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/*  Step 3: Processing & Result                                  */}
        {/* ============================================================ */}
        {step === 2 && (
          <div className="space-y-8">
            {/* Processing state */}
            {processing && (
              <div className="flex flex-col items-center gap-6 py-16">
                <div className="relative">
                  <Loader2 className="h-12 w-12 animate-spin text-primary" />
                </div>
                <div className="text-center">
                  <h2 className="text-2xl font-bold">Generating your try-on</h2>
                  <p className="mt-2 text-muted-foreground">
                    {inferenceMode === "replicate"
                      ? "Running IDM-VTON model on Replicate..."
                      : "Processing image composite..."}
                  </p>
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  {elapsedSeconds}s elapsed
                </div>
                {selectedGarment && (
                  <div className="flex items-center gap-3 rounded-2xl border border-border bg-card p-4 shadow-sm">
                    <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-lg">
                      <Image
                        src={selectedGarment.images[0]?.url ?? ""}
                        alt={selectedGarment.name}
                        fill
                        className="object-cover"
                        unoptimized
                      />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">{selectedGarment.brand}</p>
                      <p className="text-sm font-semibold">{selectedGarment.name}</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error state */}
            {processingError && !processing && (
              <div className="flex flex-col items-center gap-6 py-16">
                <XCircle className="h-12 w-12 text-destructive" />
                <div className="text-center">
                  <h2 className="text-2xl font-bold">Generation failed</h2>
                  <p className="mt-2 text-muted-foreground">{processingError}</p>
                </div>
                <div className="flex gap-4">
                  <button
                    onClick={() => {
                      setProcessingError(null);
                      startProcessing();
                    }}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Retry
                  </button>
                  <button
                    onClick={resetWizard}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    Start Over
                  </button>
                </div>
              </div>
            )}

            {/* Result */}
            {jobResult && !processing && !processingError && selectedGarment && (
              <div className="space-y-8">
                <div className="text-center">
                  <h1 className="text-3xl font-bold tracking-tight">Your Virtual Try-On</h1>
                  <p className="mt-2 text-muted-foreground">
                    Here&apos;s how <span className="font-medium text-foreground">{selectedGarment.name}</span> looks on you.
                  </p>
                </div>

                {/* Confidence + method badge */}
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold",
                      jobResult.confidence_label === "high"
                        ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                        : jobResult.confidence_label === "medium"
                          ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
                          : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
                    )}
                  >
                    <Gauge className="h-3.5 w-3.5" />
                    {jobResult.confidence_label === "high" ? "High" : jobResult.confidence_label === "medium" ? "Medium" : "Low"} Confidence
                    {jobResult.confidence_score !== null && ` (${Math.round(jobResult.confidence_score * 100)}%)`}
                  </span>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1.5 text-xs font-medium text-muted-foreground">
                    <Cpu className="h-3.5 w-3.5" />
                    {jobResult.method === "replicate" ? "IDM-VTON" : "Local Composite"}
                  </span>
                  {jobResult.processing_time_ms !== null && (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1.5 text-xs font-medium text-muted-foreground">
                      <Clock className="h-3.5 w-3.5" />
                      {(jobResult.processing_time_ms / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>

                {/* Pipeline stages */}
                {jobResult.stages && jobResult.stages.length > 0 && (
                  <div className="mx-auto max-w-lg rounded-2xl border border-border bg-card p-6 shadow-sm">
                    <p className="mb-4 text-sm font-semibold">Pipeline Stages</p>
                    <div className="space-y-3">
                      {jobResult.stages.map((stage) => {
                        const stageNames: Record<string, string> = {
                          preprocess_person: "Person Preprocessing",
                          preprocess_garment: "Garment Preprocessing",
                          inference: "Try-On Inference",
                          postprocess: "Post-Processing & Scoring",
                        };
                        const stageIcons: Record<string, string> = {
                          completed: "text-emerald-600 dark:text-emerald-400",
                          failed: "text-red-600 dark:text-red-400",
                          skipped: "text-muted-foreground",
                        };
                        return (
                          <div key={stage.name} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              {stage.status === "completed" ? (
                                <CheckCircle className={cn("h-4 w-4", stageIcons[stage.status])} />
                              ) : stage.status === "failed" ? (
                                <XCircle className={cn("h-4 w-4", stageIcons[stage.status])} />
                              ) : (
                                <AlertCircle className={cn("h-4 w-4", stageIcons[stage.status] || "text-muted-foreground")} />
                              )}
                              <span className="text-muted-foreground">{stageNames[stage.name] || stage.name}</span>
                            </div>
                            <span className="font-mono text-xs text-muted-foreground">
                              {stage.duration_ms}ms
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Before / After */}
                <div className="grid gap-6 sm:grid-cols-2">
                  <div className="space-y-2">
                    <p className="text-center text-sm font-medium text-muted-foreground">Your Photo</p>
                    <div className="relative aspect-[3/4] overflow-hidden rounded-2xl border border-border shadow-sm">
                      {photoFilename && (
                        <Image
                          src={getUploadedPhotoUrl(photoFilename)}
                          alt="Your photo"
                          fill
                          className="object-cover"
                          unoptimized
                        />
                      )}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <p className="text-center text-sm font-medium text-muted-foreground">AI-Generated Result</p>
                    <div className="relative aspect-[3/4] overflow-hidden rounded-2xl border border-border shadow-sm">
                      {jobResult.result_url && jobId && (
                        <Image
                          src={getResultImageUrl(jobId)}
                          alt="Try-on result"
                          fill
                          className="object-cover"
                          unoptimized
                        />
                      )}
                      <div className="absolute bottom-3 left-3 rounded-md bg-black/70 px-2.5 py-1 text-xs font-medium text-white backdrop-blur">
                        {jobResult.method === "replicate" ? "AI Generated" : "Composite Preview"}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Garment info */}
                <div className="mx-auto max-w-md rounded-2xl border border-border bg-card p-6 shadow-sm">
                  <div className="flex items-center gap-4">
                    <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-lg">
                      <Image
                        src={selectedGarment.images[0]?.url ?? ""}
                        alt={selectedGarment.name}
                        fill
                        className="object-cover"
                        unoptimized
                      />
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-muted-foreground">{selectedGarment.brand}</p>
                      <p className="truncate font-semibold">{selectedGarment.name}</p>
                      <p className="text-sm font-semibold">{formatPrice(selectedGarment.price)}</p>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex flex-wrap items-center justify-center gap-4">
                  <button
                    type="button"
                    onClick={resetWizard}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <Shirt className="h-4 w-4" />
                    Try Another Garment
                  </button>
                  <Link
                    href={`/product/${selectedGarment.id}`}
                    className="flex min-h-[2.625rem] items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    Get Size Recommendation
                    <ChevronRight className="h-4 w-4" />
                  </Link>
                  {jobId && jobResult.result_url && (
                    <a
                      href={getResultImageUrl(jobId)}
                      download={`fitview-tryon-${selectedGarment.id}.jpg`}
                      className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    >
                      <Download className="h-4 w-4" />
                      Download
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/*  Page export                                                        */
/* ------------------------------------------------------------------ */

export default function TryOnPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-[60vh] items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <TryOnWizard />
    </Suspense>
  );
}
