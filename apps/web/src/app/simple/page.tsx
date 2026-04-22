"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Image from "next/image";
import { Upload, Loader2, Download, X, Clock } from "lucide-react";
import {
  createSimpleTryOn,
  pollSimpleTryOn,
  getSimpleResultImageUrl,
  type TryOnJob,
  type PipelineStage,
} from "@/lib/api";

const POLL_INTERVAL = 2000;
const GARMENT_TYPES = [
  { value: "upper_body", label: "Top (shirt, jacket, etc.)" },
  { value: "lower_body", label: "Bottom (pants, skirt)" },
  { value: "full_body", label: "Full body (dress, jumpsuit)" },
];

export default function SimpleTryOnPage() {
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [garmentFile, setGarmentFile] = useState<File | null>(null);
  const [personPreview, setPersonPreview] = useState<string | null>(null);
  const [garmentPreview, setGarmentPreview] = useState<string | null>(null);
  const [garmentType, setGarmentType] = useState("upper_body");

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<TryOnJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const personInputRef = useRef<HTMLInputElement>(null);
  const garmentInputRef = useRef<HTMLInputElement>(null);

  // intent: handle file selection for person/garment with preview
  // status: done
  // confidence: high
  const handleFile = useCallback(
    (type: "person" | "garment", file: File | null) => {
      if (!file) return;
      const url = URL.createObjectURL(file);
      if (type === "person") {
        setPersonFile(file);
        setPersonPreview(url);
      } else {
        setGarmentFile(file);
        setGarmentPreview(url);
      }
    },
    []
  );

  const clearFile = useCallback((type: "person" | "garment") => {
    if (type === "person") {
      setPersonFile(null);
      if (personPreview) URL.revokeObjectURL(personPreview);
      setPersonPreview(null);
    } else {
      setGarmentFile(null);
      if (garmentPreview) URL.revokeObjectURL(garmentPreview);
      setGarmentPreview(null);
    }
  }, [personPreview, garmentPreview]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startTryOn = useCallback(async () => {
    if (!personFile || !garmentFile) return;

    setError(null);
    setJob(null);
    setElapsed(0);
    startTimeRef.current = Date.now();

    // Start elapsed timer
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    try {
      const { job_id } = await createSimpleTryOn(personFile, garmentFile, garmentType);
      setJobId(job_id);

      // Start polling
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollSimpleTryOn(job_id);
          setJob(status);

          if (status.status === "completed" || status.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            if (timerRef.current) clearInterval(timerRef.current);
            pollRef.current = null;
            timerRef.current = null;

            if (status.status === "failed") {
              setError(status.error || "Try-on failed");
            }
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : "Polling failed");
          if (pollRef.current) clearInterval(pollRef.current);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      }, POLL_INTERVAL);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start try-on");
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, [personFile, garmentFile, garmentType]);

  const reset = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    setJobId(null);
    setJob(null);
    setError(null);
    setElapsed(0);
  }, []);

  const isProcessing = jobId !== null && job?.status !== "completed" && job?.status !== "failed";
  const isComplete = job?.status === "completed";

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-bold mb-2">Simple Try-On</h1>
      <p className="text-gray-500 mb-8">
        Upload a person photo and a garment image. Get a virtual try-on result.
      </p>

      {/* Upload zone */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <DropZone
          label="Person Photo"
          hint="Full or half-body photo"
          file={personFile}
          preview={personPreview}
          inputRef={personInputRef}
          onFile={(f) => handleFile("person", f)}
          onClear={() => clearFile("person")}
          disabled={isProcessing}
        />
        <DropZone
          label="Garment Image"
          hint="Flat-lay or product photo"
          file={garmentFile}
          preview={garmentPreview}
          inputRef={garmentInputRef}
          onFile={(f) => handleFile("garment", f)}
          onClear={() => clearFile("garment")}
          disabled={isProcessing}
        />
      </div>

      {/* Garment type + submit */}
      <div className="flex flex-wrap items-center gap-4 mb-8">
        <select
          value={garmentType}
          onChange={(e) => setGarmentType(e.target.value)}
          disabled={isProcessing}
          className="rounded-md border border-gray-300 px-3 py-2 text-sm bg-white dark:bg-gray-900 dark:border-gray-700"
        >
          {GARMENT_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>

        <button
          onClick={startTryOn}
          disabled={!personFile || !garmentFile || isProcessing}
          className="rounded-md bg-black text-white px-6 py-2 text-sm font-medium hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed dark:bg-white dark:text-black dark:hover:bg-gray-200"
        >
          {isProcessing ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Processing...
            </span>
          ) : (
            "Run Try-On"
          )}
        </button>

        {(isComplete || error) && (
          <button
            onClick={reset}
            className="rounded-md border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            Try Again
          </button>
        )}
      </div>

      {/* Status / progress */}
      {isProcessing && (
        <div className="mb-8 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
            <Clock className="h-4 w-4" />
            <span>Elapsed: {formatTime(elapsed)}</span>
            {job?.method && (
              <span className="ml-auto text-xs bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded">
                {job.method}
              </span>
            )}
          </div>
          {job?.stages && job.stages.length > 0 && (
            <div className="space-y-1">
              {job.stages.map((s: PipelineStage) => (
                <StageRow key={s.name} stage={s} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-8 rounded-lg border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950 p-4 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {isComplete && jobId && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">Result</h2>
            {job.confidence_label && (
              <span
                className={`text-xs px-2 py-0.5 rounded ${
                  job.confidence_label === "high"
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
                    : job.confidence_label === "medium"
                    ? "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300"
                    : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                }`}
              >
                {job.confidence_label} confidence
              </span>
            )}
            {job.processing_time_ms && (
              <span className="text-xs text-gray-500">
                {(job.processing_time_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Person input */}
            {personPreview && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Person</p>
                <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                  <Image src={personPreview} alt="Person" fill className="object-cover" />
                </div>
              </div>
            )}
            {/* Garment input */}
            {garmentPreview && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Garment</p>
                <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                  <Image src={garmentPreview} alt="Garment" fill className="object-cover" />
                </div>
              </div>
            )}
            {/* Result */}
            <div>
              <p className="text-xs text-gray-500 mb-1">Try-On Result</p>
              <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={getSimpleResultImageUrl(jobId)}
                  alt="Try-on result"
                  className="h-full w-full object-cover"
                />
              </div>
              <a
                href={getSimpleResultImageUrl(jobId)}
                download="tryon-result.png"
                className="mt-2 inline-flex items-center gap-1 text-xs text-gray-600 hover:text-black dark:text-gray-400 dark:hover:text-white"
              >
                <Download className="h-3 w-3" /> Download
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function DropZone({
  label,
  hint,
  file,
  preview,
  inputRef,
  onFile,
  onClear,
  disabled,
}: {
  label: string;
  hint: string;
  file: File | null;
  preview: string | null;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onFile: (f: File) => void;
  onClear: () => void;
  disabled: boolean;
}) {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) onFile(f);
    },
    [onFile]
  );

  if (preview && file) {
    return (
      <div className="relative rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="relative aspect-[3/4]">
          <Image src={preview} alt={label} fill className="object-cover" />
        </div>
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/60 to-transparent p-3">
          <p className="text-white text-sm font-medium">{label}</p>
          <p className="text-white/70 text-xs truncate">{file.name}</p>
        </div>
        {!disabled && (
          <button
            onClick={onClear}
            className="absolute top-2 right-2 rounded-full bg-black/50 p-1 text-white hover:bg-black/70"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={disabled ? undefined : handleDrop}
      className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 aspect-[3/4] transition-colors ${
        disabled
          ? "border-gray-200 bg-gray-50 cursor-not-allowed dark:border-gray-800 dark:bg-gray-900"
          : "border-gray-300 hover:border-gray-400 cursor-pointer dark:border-gray-600 dark:hover:border-gray-500"
      }`}
    >
      <Upload className="h-8 w-8 text-gray-400 mb-3" />
      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</p>
      <p className="text-xs text-gray-500 mt-1">{hint}</p>
      <p className="text-xs text-gray-400 mt-2">Click or drag & drop</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function StageRow({ stage }: { stage: PipelineStage }) {
  const icon =
    stage.status === "completed"
      ? "text-emerald-500"
      : stage.status === "running"
      ? "text-blue-500"
      : stage.status === "failed"
      ? "text-red-500"
      : "text-gray-300";

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${icon} bg-current`} />
      <span className="text-gray-600 dark:text-gray-400 flex-1">
        {stage.name.replace(/_/g, " ")}
      </span>
      {stage.duration_ms > 0 && (
        <span className="text-gray-400">{(stage.duration_ms / 1000).toFixed(1)}s</span>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
