"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Camera, Upload, Loader2, RotateCcw, Ruler, User } from "lucide-react";
import {
  estimateBodyMeasurements,
  quickBodyMeasurement,
  type BodyMeasurementResult,
} from "@/lib/api";

// intent: webcam capture page for photo-based body measurement estimation
// status: done
// confidence: high

type InputMode = "webcam" | "upload";
type Step = "input" | "capture" | "result";

const POPULATIONS = [
  { value: "universal", label: "Universal (US)" },
  { value: "vietnamese", label: "Vietnamese" },
];

const FIT_OPTIONS = [
  { value: "slim", label: "Slim fit" },
  { value: "regular", label: "Regular fit" },
  { value: "relaxed", label: "Relaxed fit" },
];

export default function MeasurePage() {
  const [step, setStep] = useState<Step>("input");
  const [inputMode, setInputMode] = useState<InputMode>("webcam");

  // User inputs
  const [heightCm, setHeightCm] = useState<string>("165");
  const [weightKg, setWeightKg] = useState<string>("55");
  const [population, setPopulation] = useState("universal");
  const [fitPreference, setFitPreference] = useState("regular");

  // Capture state
  const [photoBlob, setPhotoBlob] = useState<Blob | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Result state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BodyMeasurementResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
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
      setCameraError(
        err instanceof Error ? err.message : "Failed to access camera"
      );
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Mirror the capture (webcam is mirrored)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          setPhotoBlob(blob);
          setPhotoPreview(URL.createObjectURL(blob));
          stopCamera();
          setStep("capture");
        }
      },
      "image/jpeg",
      0.92
    );
  }, [stopCamera]);

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setPhotoBlob(file);
      setPhotoPreview(URL.createObjectURL(file));
      setStep("capture");
      e.target.value = "";
    },
    []
  );

  const retakePhoto = useCallback(() => {
    if (photoPreview) URL.revokeObjectURL(photoPreview);
    setPhotoBlob(null);
    setPhotoPreview(null);
    setResult(null);
    setError(null);
    setStep("input");
  }, [photoPreview]);

  const submitMeasurement = useCallback(async () => {
    const h = parseFloat(heightCm);
    const w = parseFloat(weightKg);
    if (isNaN(h) || isNaN(w) || h < 100 || h > 250 || w < 25 || w > 200) {
      setError("Please enter valid height (100-250 cm) and weight (25-200 kg)");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let res: BodyMeasurementResult;
      if (photoBlob) {
        res = await estimateBodyMeasurements(photoBlob, h, w, population, fitPreference);
      } else {
        res = await quickBodyMeasurement({
          height_cm: h,
          weight_kg: w,
          population,
          fit_preference: fitPreference,
        });
      }
      setResult(res);
      setStep("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Estimation failed");
    } finally {
      setLoading(false);
    }
  }, [heightCm, weightKg, photoBlob, population, fitPreference]);

  const resetAll = useCallback(() => {
    retakePhoto();
    setStep("input");
  }, [retakePhoto]);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-bold mb-2">Body Measurement</h1>
      <p className="text-gray-500 mb-8">
        Enter your height and weight, then take a front-facing photo for accurate measurements.
      </p>

      {/* ── Step 1: Input height/weight + photo ── */}
      {step === "input" && (
        <div className="space-y-6">
          {/* Height / Weight inputs */}
          <div className="grid grid-cols-2 gap-4 max-w-md">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Height (cm)
              </label>
              <input
                type="number"
                value={heightCm}
                onChange={(e) => setHeightCm(e.target.value)}
                min={100}
                max={250}
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Weight (kg)
              </label>
              <input
                type="number"
                value={weightKg}
                onChange={(e) => setWeightKg(e.target.value)}
                min={25}
                max={200}
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
              />
            </div>
          </div>

          {/* Population + fit preference */}
          <div className="flex flex-wrap gap-4 max-w-md">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Population
              </label>
              <select
                value={population}
                onChange={(e) => setPopulation(e.target.value)}
                className="rounded-md border border-gray-300 px-3 py-2 text-sm bg-white dark:bg-gray-900 dark:border-gray-700"
              >
                {POPULATIONS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Fit preference
              </label>
              <select
                value={fitPreference}
                onChange={(e) => setFitPreference(e.target.value)}
                className="rounded-md border border-gray-300 px-3 py-2 text-sm bg-white dark:bg-gray-900 dark:border-gray-700"
              >
                {FIT_OPTIONS.map((f) => (
                  <option key={f.value} value={f.value}>
                    {f.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Photo capture mode toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setInputMode("webcam")}
              className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium border transition-colors ${
                inputMode === "webcam"
                  ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50 dark:bg-gray-900 dark:text-gray-300 dark:border-gray-700"
              }`}
            >
              <Camera className="h-4 w-4" />
              Webcam
            </button>
            <button
              onClick={() => {
                setInputMode("upload");
                stopCamera();
              }}
              className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium border transition-colors ${
                inputMode === "upload"
                  ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50 dark:bg-gray-900 dark:text-gray-300 dark:border-gray-700"
              }`}
            >
              <Upload className="h-4 w-4" />
              Upload
            </button>
          </div>

          {/* Webcam view */}
          {inputMode === "webcam" && (
            <div className="space-y-3">
              <div className="relative aspect-[3/4] max-w-md rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-900">
                <video
                  ref={videoRef}
                  className="h-full w-full object-cover"
                  style={{ transform: "scaleX(-1)" }}
                  playsInline
                  muted
                />
                {!cameraActive && !cameraError && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <User className="h-16 w-16 text-gray-300 mb-4" />
                    <p className="text-sm text-gray-500">Camera not active</p>
                  </div>
                )}
                {cameraError && (
                  <div className="absolute inset-0 flex items-center justify-center p-4">
                    <p className="text-sm text-red-500 text-center">{cameraError}</p>
                  </div>
                )}
                {/* Pose guide overlay */}
                {cameraActive && (
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute inset-x-[20%] top-[5%] bottom-[5%] border-2 border-dashed border-white/30 rounded-lg" />
                    <p className="absolute bottom-3 left-0 right-0 text-center text-xs text-white/70 bg-black/30 py-1">
                      Stand in the frame, arms slightly away from body
                    </p>
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="flex items-center gap-2 rounded-md bg-black text-white px-4 py-2 text-sm font-medium hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                  >
                    <Camera className="h-4 w-4" />
                    Start Camera
                  </button>
                ) : (
                  <button
                    onClick={capturePhoto}
                    className="flex items-center gap-2 rounded-md bg-emerald-600 text-white px-6 py-2 text-sm font-medium hover:bg-emerald-700"
                  >
                    <Camera className="h-4 w-4" />
                    Capture Photo
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Upload mode */}
          {inputMode === "upload" && (
            <div className="space-y-3">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-8 aspect-[3/4] max-w-md cursor-pointer hover:border-gray-400 transition-colors dark:border-gray-600 dark:hover:border-gray-500"
              >
                <Upload className="h-8 w-8 text-gray-400 mb-3" />
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Upload front-facing photo
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Full body, arms slightly apart
                </p>
                <p className="text-xs text-gray-400 mt-2">Click to browse</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>
          )}

          {/* Skip photo — quick estimate */}
          <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={submitMeasurement}
              disabled={loading}
              className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 underline"
            >
              Skip photo — estimate from height/weight only
            </button>
          </div>
        </div>
      )}

      {/* ── Step 2: Review captured photo ── */}
      {step === "capture" && photoPreview && (
        <div className="space-y-4">
          <div className="relative aspect-[3/4] max-w-md rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={photoPreview}
              alt="Captured"
              className="h-full w-full object-cover"
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={submitMeasurement}
              disabled={loading}
              className="flex items-center gap-2 rounded-md bg-black text-white px-6 py-2 text-sm font-medium hover:bg-gray-800 disabled:opacity-40 dark:bg-white dark:text-black dark:hover:bg-gray-200"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Estimating...
                </>
              ) : (
                <>
                  <Ruler className="h-4 w-4" />
                  Estimate Measurements
                </>
              )}
            </button>
            <button
              onClick={retakePhoto}
              disabled={loading}
              className="flex items-center gap-2 rounded-md border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
            >
              <RotateCcw className="h-4 w-4" />
              Retake
            </button>
          </div>
        </div>
      )}

      {/* ── Step 3: Results ── */}
      {step === "result" && result && (
        <div className="space-y-6">
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950 p-3">
              {result.warnings.map((w, i) => (
                <p key={i} className="text-sm text-amber-700 dark:text-amber-400">
                  {w}
                </p>
              ))}
            </div>
          )}

          {/* Size recommendation — hero card */}
          <div className="rounded-xl border-2 border-black dark:border-white p-6">
            <div className="flex items-baseline gap-3 mb-2">
              <span className="text-4xl font-bold">{result.recommended_size}</span>
              <span
                className={`text-sm px-2 py-0.5 rounded ${
                  result.size_confidence === "high"
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
                    : result.size_confidence === "medium"
                    ? "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300"
                    : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                }`}
              >
                {result.size_confidence} confidence
              </span>
            </div>
            <p className="text-sm text-gray-500">
              Recommended size based on {result.method === "mediapipe_landmark_ellipse" ? "photo analysis" : "height/weight regression"}
              {result.landmark_confidence > 0 && ` (landmark confidence: ${(result.landmark_confidence * 100).toFixed(0)}%)`}
            </p>
            {result.alternatives.length > 0 && (
              <div className="mt-3 space-y-1">
                {result.alternatives.map((alt, i) => (
                  <p key={i} className="text-xs text-gray-500">
                    <span className="font-medium">{alt.size}</span> — {alt.note}
                  </p>
                ))}
              </div>
            )}
          </div>

          {/* Measurements grid */}
          <div>
            <h2 className="text-lg font-semibold mb-3">Estimated Measurements</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <MeasurementCard label="Chest" value={result.chest_circumference} unit="cm" />
              <MeasurementCard label="Waist" value={result.waist_circumference} unit="cm" />
              <MeasurementCard label="Hip" value={result.hip_circumference} unit="cm" />
              <MeasurementCard label="Shoulder" value={result.shoulder_width} unit="cm" />
              <MeasurementCard label="Arm length" value={result.arm_length} unit="cm" />
              <MeasurementCard label="Leg length" value={result.leg_length} unit="cm" />
              <MeasurementCard label="Torso" value={result.torso_length} unit="cm" />
              <MeasurementCard label="BMI" value={result.bmi} unit="" />
            </div>
          </div>

          {/* Population comparison */}
          {result.population_comparison && (
            <div>
              <h2 className="text-lg font-semibold mb-3">Population Comparison</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ComparisonCard
                  title="US (Universal)"
                  chest={result.population_comparison.baseline.estimated_chest_cm}
                  waist={result.population_comparison.baseline.estimated_waist_cm}
                  hip={result.population_comparison.baseline.estimated_hip_cm}
                  size={result.population_comparison.baseline.recommended_size}
                />
                <ComparisonCard
                  title="Vietnamese"
                  chest={result.population_comparison.eavton.estimated_chest_cm}
                  waist={result.population_comparison.eavton.estimated_waist_cm}
                  hip={result.population_comparison.eavton.estimated_hip_cm}
                  size={result.population_comparison.eavton.recommended_size}
                  bodyType={result.population_comparison.eavton.nearest_body_type}
                />
              </div>
              {result.population_comparison.comparison.size_changed && (
                <p className="mt-3 text-sm text-amber-600 dark:text-amber-400">
                  Size differs between models — the Vietnamese model recommends a different size
                  (chest difference: {result.population_comparison.comparison.chest_difference_cm} cm)
                </p>
              )}
            </div>
          )}

          {/* Photo + result side by side */}
          {photoPreview && (
            <div className="max-w-md">
              <h2 className="text-lg font-semibold mb-3">Your Photo</h2>
              <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={photoPreview}
                  alt="Your photo"
                  className="h-full w-full object-cover"
                />
              </div>
            </div>
          )}

          {/* Reset */}
          <button
            onClick={resetAll}
            className="flex items-center gap-2 rounded-md border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            <RotateCcw className="h-4 w-4" />
            Start Over
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950 p-4 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

/* ── Sub-components ── */

function MeasurementCard({
  label,
  value,
  unit,
}: {
  label: string;
  value: number;
  unit: string;
}) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-semibold">
        {value}
        {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
      </p>
    </div>
  );
}

function ComparisonCard({
  title,
  chest,
  waist,
  hip,
  size,
  bodyType,
}: {
  title: string;
  chest: number;
  waist: number;
  hip: number;
  size: string;
  bodyType?: string | null;
}) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-medium mb-2">{title}</h3>
      <div className="space-y-1 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">Chest</span>
          <span>{chest} cm</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Waist</span>
          <span>{waist} cm</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Hip</span>
          <span>{hip} cm</span>
        </div>
        <div className="flex justify-between pt-2 border-t border-gray-100 dark:border-gray-800">
          <span className="text-gray-500">Size</span>
          <span className="font-semibold">{size}</span>
        </div>
        {bodyType && (
          <div className="flex justify-between">
            <span className="text-gray-500">Body type</span>
            <span className="text-xs">{bodyType}</span>
          </div>
        )}
      </div>
    </div>
  );
}
