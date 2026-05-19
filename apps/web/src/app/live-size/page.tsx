"use client";

// intent: v2 layout — mobile-first guided journey per apps/web/DESIGN.md
// status: scaffolded; uses image-upload as the simplest entry path
// next: optionally fold in the live camera feed from the original page if v2 lands well
// blockers: live-camera state graph in the original is complex; v2 starts with image upload
// confidence: high

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Camera,
  Ruler,
  Shirt,
  Sparkles,
  Upload,
} from "lucide-react";

import { detectPoseFromImage } from "@/hooks/use-image-pose";
import {
  estimateWeightFromRatios,
  extractBodyRatios,
  type BodyRatios,
} from "@/hooks/use-body-ratios";
import {
  getUpperStyleRecommendation,
  getVisualSizeRecommendation,
  type UpperStyleRecommendationRequest,
  type UpperStyleRecommendationResult,
  type UpperStyleSize,
  type VisualSizeResult,
} from "@/lib/api";

import {
  StepIndicator,
  type JourneyStep as TStep,
} from "@/components/shared/StepIndicator";
import { JourneyStep } from "@/components/shared/JourneyStep";
import { CalibrationPanel } from "@/components/shared/CalibrationPanel";

const STEPS: TStep[] = [
  { id: "capture", label: "Capture" },
  { id: "size", label: "Size" },
  { id: "style", label: "Style" },
  { id: "personalize", label: "Personalize" },
];

type PageState = {
  capturedImageURL: string | null;
  ratios: BodyRatios | null;
  height_cm: number;
  visualResult: VisualSizeResult | null;
  styleResult: UpperStyleRecommendationResult | null;
};

const INITIAL: PageState = {
  capturedImageURL: null,
  ratios: null,
  height_cm: 158,
  visualResult: null,
  styleResult: null,
};

function toUpperSize(s: string | null | undefined): UpperStyleSize | null {
  const u = (s ?? "").toUpperCase();
  if (["XS", "S", "M", "L", "XL", "XXL"].includes(u)) {
    return u as UpperStyleSize;
  }
  return null;
}

export default function LiveSizeV2Page() {
  const [step, setStep] = useState(0);
  const [pageState, setPageState] = useState<PageState>(INITIAL);
  const [busy, setBusy] = useState<null | "pose" | "size" | "style">(null);
  const [error, setError] = useState<string | null>(null);

  // ── Step 1: upload + extract ratios ──
  const onUpload = useCallback(async (file: File) => {
    setError(null);
    setBusy("pose");
    try {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.src = url;
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
      });
      const result = await detectPoseFromImage(img);
      if (!result || !result.landmarks) {
        throw new Error(
          "Could not detect pose in this image. Try a full-body photo with good lighting.",
        );
      }
      const ratios = extractBodyRatios(result.landmarks, result.worldLandmarks);
      if (!ratios) {
        throw new Error(
          "Detected a pose but couldn't compute body ratios. Make sure shoulders + hips are in frame.",
        );
      }
      setPageState((p) => ({ ...p, capturedImageURL: url, ratios }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to process image");
    } finally {
      setBusy(null);
    }
  }, []);

  // ── Step 2: predict size ──
  const predictSize = useCallback(async () => {
    if (!pageState.ratios) return;
    setError(null);
    setBusy("size");
    try {
      // weight estimate is informational only; the visual size API derives weight internally
      const _weightGuess = estimateWeightFromRatios(
        pageState.height_cm,
        pageState.ratios,
      );
      void _weightGuess;
      const res = await getVisualSizeRecommendation({
        height_cm: pageState.height_cm,
        shoulder_to_hip: pageState.ratios.shoulderToHip,
        waist_to_hip: pageState.ratios.waistToHip,
        shoulder_to_torso: pageState.ratios.shoulderToTorso,
        torso_to_leg: pageState.ratios.torsoToLeg,
        arm_to_torso: pageState.ratios.armToTorso,
        population: "vietnamese",
        source: pageState.ratios.source,
      });
      setPageState((p) => ({ ...p, visualResult: res }));
      setStep(2);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Size prediction failed");
    } finally {
      setBusy(null);
    }
  }, [pageState.ratios, pageState.height_cm]);

  // ── Step 3: get style recommendations ──
  const fetchStyle = useCallback(async () => {
    if (!pageState.visualResult || !pageState.ratios) return;
    const size = toUpperSize(
      pageState.visualResult.research_model.predicted_size,
    );
    if (!size) {
      setError("Predicted size is not in the supported XS–XXL range.");
      return;
    }
    setError(null);
    setBusy("style");
    try {
      const res = await getUpperStyleRecommendation({
        height_cm: pageState.height_cm,
        predicted_size: size,
        population: "vietnamese",
        ratios: {
          sh: pageState.ratios.shoulderToHip,
          wh: pageState.ratios.waistToHip,
          st: pageState.ratios.shoulderToTorso,
          tl: pageState.ratios.torsoToLeg,
          at: pageState.ratios.armToTorso,
        },
        context: { occasion: "casual", sliders: {} },
        top_k: 8,
      });
      setPageState((p) => ({ ...p, styleResult: res }));
      setStep(3);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Style recommendation failed");
    } finally {
      setBusy(null);
    }
  }, [pageState.visualResult, pageState.ratios, pageState.height_cm]);

  // ── Calibration request memo ──
  const calibrationBaseRequest =
    useMemo<UpperStyleRecommendationRequest | null>(() => {
      if (!pageState.ratios || !pageState.visualResult) return null;
      const size = toUpperSize(
        pageState.visualResult.research_model.predicted_size,
      );
      if (!size) return null;
      return {
        height_cm: pageState.height_cm,
        predicted_size: size,
        population: "vietnamese",
        ratios: {
          sh: pageState.ratios.shoulderToHip,
          wh: pageState.ratios.waistToHip,
          st: pageState.ratios.shoulderToTorso,
          tl: pageState.ratios.torsoToLeg,
          at: pageState.ratios.armToTorso,
        },
        context: { occasion: "casual", sliders: {} },
        top_k: 8,
      };
    }, [pageState.ratios, pageState.visualResult, pageState.height_cm]);

  const stepState = useCallback(
    (i: number): "active" | "complete" | "upcoming" => {
      if (i < step) return "complete";
      if (i === step) return "active";
      return "upcoming";
    },
    [step],
  );

  const predictedSize =
    pageState.visualResult?.research_model.predicted_size ?? null;
  const confidence =
    pageState.visualResult?.research_model.confidence_score ?? null;

  return (
    <main className="mx-auto flex min-h-[100dvh] w-full max-w-3xl flex-col gap-4 px-4 py-4 md:gap-6 md:py-6">
      {/* Top chrome: link to advanced (live camera + diagnostics) */}
      <div className="flex items-center justify-between">
        <Link
          href="/"
          className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-3 py-1.5 text-[11px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <ArrowLeft className="h-3 w-3" />
          Home
        </Link>
        <Link
          href="/live-size/advanced"
          className="rounded-full bg-muted px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          Advanced (live camera)
        </Link>
      </div>

      {/* Step indicator */}
      <StepIndicator
        steps={STEPS}
        current={step}
        onJump={(i) => i < step && setStep(i)}
      />

      {/* Page-level error */}
      {error && (
        <div
          className="rounded-2xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* Steps */}
      <div className="flex flex-col gap-3 md:gap-4">
        {/* Step 1: Capture */}
        <JourneyStep
          state={stepState(0)}
          eyebrow="Step 1"
          title="Stand in frame, full body"
          description="Upload a clear full-body photo. We extract body proportions only — face is blurred before any storage."
          collapsedSummary={
            pageState.ratios
              ? `${(pageState.ratios.shoulderToHip ?? 0).toFixed(2)} shoulder/hip • ${(pageState.ratios.waistToHip ?? 0).toFixed(2)} waist/hip`
              : undefined
          }
          onExpand={() => setStep(0)}
          primaryAction={
            pageState.ratios ? (
              <button
                type="button"
                onClick={() => setStep(1)}
                className="inline-flex h-11 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background"
              >
                Continue
                <Ruler className="h-4 w-4" />
              </button>
            ) : null
          }
        >
          <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-border bg-background/50 px-6 py-12 text-center transition-colors hover:border-foreground/30 hover:bg-accent/30">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <Upload className="h-5 w-5" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold">
                {busy === "pose" ? "Processing image…" : "Tap to upload"}
              </p>
              <p className="text-xs text-muted-foreground">
                JPG / PNG. Full-body photo. Good lighting.
              </p>
            </div>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void onUpload(f);
              }}
              disabled={busy === "pose"}
            />
          </label>
          {pageState.capturedImageURL && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={pageState.capturedImageURL}
              alt="Uploaded body photo"
              className="mx-auto max-h-72 rounded-2xl border border-border object-contain"
            />
          )}
          {pageState.ratios && (
            <div className="rounded-2xl bg-muted/50 p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  Body ratios extracted
                </span>
                <Camera className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <RatioRow
                  label="Shoulder / Hip"
                  value={pageState.ratios.shoulderToHip}
                />
                <RatioRow
                  label="Waist / Hip"
                  value={pageState.ratios.waistToHip}
                />
                <RatioRow
                  label="Shoulder / Torso"
                  value={pageState.ratios.shoulderToTorso}
                />
                <RatioRow
                  label="Torso / Leg"
                  value={pageState.ratios.torsoToLeg}
                />
              </dl>
            </div>
          )}
        </JourneyStep>

        {/* Step 2: Size */}
        <JourneyStep
          state={stepState(1)}
          eyebrow="Step 2"
          title="What size fits you"
          description="The model uses your photo's body proportions + your height to predict a size band for upper-body garments."
          collapsedSummary={
            predictedSize
              ? `${predictedSize}${confidence != null ? ` · ${Math.round(confidence * 100)}% confidence` : ""}`
              : undefined
          }
          onExpand={() => setStep(1)}
          primaryAction={
            predictedSize ? (
              <button
                type="button"
                onClick={() => {
                  setStep(2);
                  void fetchStyle();
                }}
                className="inline-flex h-11 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background"
              >
                See styles
                <Shirt className="h-4 w-4" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void predictSize()}
                disabled={!pageState.ratios || busy === "size"}
                className="inline-flex h-11 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background disabled:opacity-40"
              >
                {busy === "size" ? "Predicting…" : "Predict size"}
                <Ruler className="h-4 w-4" />
              </button>
            )
          }
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                Your height (cm)
              </span>
              <input
                type="number"
                value={pageState.height_cm}
                min={140}
                max={200}
                onChange={(e) =>
                  setPageState((p) => ({
                    ...p,
                    height_cm: Number(e.target.value) || 158,
                  }))
                }
                className="rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </label>
          </div>
          {pageState.visualResult && (
            <div className="rounded-2xl bg-muted/50 p-6 text-center">
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                Predicted size
              </span>
              <div className="mt-1 flex items-baseline justify-center gap-3">
                <span className="text-5xl font-black tracking-tight">
                  {predictedSize}
                </span>
                {confidence != null && (
                  <span className="text-xs font-medium text-muted-foreground">
                    {Math.round(confidence * 100)}% conf
                  </span>
                )}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Model: {pageState.visualResult.research_model.model ?? "GBM"}
              </p>
            </div>
          )}
        </JourneyStep>

        {/* Step 3: Style */}
        <JourneyStep
          state={stepState(2)}
          eyebrow="Step 3"
          title="Tops that suit you"
          description="Body-shape-aware recommendations from our catalog. Ranked by fit, flatter (shape), and match (color & occasion)."
          collapsedSummary={
            pageState.styleResult
              ? `${pageState.styleResult.items.length} items ranked`
              : undefined
          }
          onExpand={() => setStep(2)}
          primaryAction={
            pageState.styleResult ? (
              <button
                type="button"
                onClick={() => setStep(3)}
                className="inline-flex h-11 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background"
              >
                Personalize
                <Sparkles className="h-4 w-4" />
              </button>
            ) : null
          }
        >
          {busy === "style" && (
            <p className="rounded-2xl bg-muted/50 px-4 py-6 text-center text-sm text-muted-foreground">
              Ranking the catalog for your body shape…
            </p>
          )}
          {pageState.styleResult && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
              {pageState.styleResult.items.slice(0, 8).map((it) => (
                <div
                  key={it.item_id}
                  className="overflow-hidden rounded-2xl border border-border bg-background"
                >
                  <div className="aspect-[3/4] w-full bg-muted">
                    {it.image_url && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={it.image_url}
                        alt={it.title}
                        className="h-full w-full object-cover"
                        loading="lazy"
                      />
                    )}
                  </div>
                  <div className="space-y-1 p-2">
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>#{it.rank}</span>
                      <span className="font-mono">
                        {it.total_score.toFixed(2)}
                      </span>
                    </div>
                    <p className="truncate text-xs font-medium">{it.title}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </JourneyStep>

        {/* Step 4: Personalize */}
        <JourneyStep
          state={stepState(3)}
          eyebrow="Step 4"
          title="Personalize"
          description="Pick 3 to 8 items from a diverse seed grid. We use your picks (CLIP-centroid) to re-rank the recommendations to your taste."
          primaryAction={null}
        >
          {step === 3 && (
            <CalibrationPanel baseRequest={calibrationBaseRequest} />
          )}
        </JourneyStep>
      </div>
    </main>
  );
}

function RatioRow({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono text-right">
        {value != null ? value.toFixed(2) : "—"}
      </dd>
    </>
  );
}
