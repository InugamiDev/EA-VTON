"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import {
  ArrowLeft,
  ChevronDown,
  ChevronUp,
  Ruler,
  ShoppingBag,
  X,
} from "lucide-react";
import { cn, formatPrice } from "@/lib/utils";
import { GARMENTS } from "@/data/garments";
import { compareSizeRecommendation } from "@/lib/api";
import type { SizeComparisonResult } from "@/lib/api";
import type { Garment } from "@/types";

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

function getDisplayImage(garment: Garment): string {
  const display = garment.images.find((img) => img.type === "display");
  return display?.url ?? garment.images[0]?.url ?? "";
}

const COLOR_MAP: Record<string, string> = {
  White: "#ffffff",
  Black: "#1a1a1a",
  Navy: "#1e3a5f",
  Gray: "#9ca3af",
  Blue: "#3b82f6",
  Pink: "#f472b6",
  Striped: "repeating-linear-gradient(45deg, #e5e7eb, #e5e7eb 2px, #fff 2px, #fff 6px)",
  Olive: "#6b7c3f",
  Beige: "#d4c5a9",
  "Light Blue": "#93c5fd",
  "Red Check": "#dc2626",
  "Blue Check": "#2563eb",
  "Green Check": "#16a34a",
};

/* -------------------------------------------------------------------------- */
/*  Page component                                                            */
/* -------------------------------------------------------------------------- */

export default function ProductPage() {
  const { id } = useParams<{ id: string }>();
  const garment = GARMENTS.find((g) => g.id === id);

  const [selectedSize, setSelectedSize] = useState<string | null>(null);
  const [sizeChartOpen, setSizeChartOpen] = useState(false);
  const [recModalOpen, setRecModalOpen] = useState(false);
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [fit, setFit] = useState<"slim" | "regular" | "relaxed">("regular");
  const [comparison, setComparison] = useState<SizeComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRecommend = async () => {
    if (!garment || !height || !weight) return;
    setLoading(true);
    try {
      const result = await compareSizeRecommendation({
        garment_id: garment.id,
        height_cm: Number(height),
        weight_kg: Number(weight),
        fit_preference: fit,
      });
      setComparison(result);
      setSelectedSize(result.eavton.recommended_size);
    } catch {
      // Fallback: if backend is down, still show something
      setComparison(null);
    } finally {
      setLoading(false);
    }
  };

  if (!garment) {
    return (
      <section className="py-32">
        <div className="mx-auto max-w-7xl px-6 text-center">
          <ShoppingBag className="mx-auto h-16 w-16 text-muted-foreground/40" />
          <h1 className="mt-6 text-3xl font-bold">Garment not found</h1>
          <p className="mt-3 text-muted-foreground">
            The product you&apos;re looking for doesn&apos;t exist or has been
            removed.
          </p>
          <Link
            href="/catalog"
            className="mt-8 inline-flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Catalog
          </Link>
        </div>
      </section>
    );
  }

  const imageUrl = getDisplayImage(garment);

  return (
    <>
      {/* Breadcrumb */}
      <div className="border-b border-border">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <Link
            href="/catalog"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors duration-200 hover:text-foreground motion-reduce:transition-none"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Catalog
          </Link>
        </div>
      </div>

      {/* Product detail */}
      <section className="py-16">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid gap-10 lg:grid-cols-2 lg:gap-16">
            {/* Left: Image */}
            <div className="relative aspect-[3/4] overflow-hidden rounded-2xl border border-border bg-muted shadow-sm">
              <Image
                src={imageUrl}
                alt={garment.name}
                fill
                unoptimized
                className="object-cover"
                sizes="(max-width: 1024px) 100vw, 50vw"
                priority
              />
            </div>

            {/* Right: Details */}
            <div className="flex flex-col">
              <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                {garment.brand}
              </p>
              <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
                {garment.name}
              </h1>
              <p className="mt-4 text-2xl font-bold">
                {formatPrice(garment.price)}
              </p>
              <p className="mt-4 leading-relaxed text-muted-foreground">
                {garment.description}
              </p>

              {/* Colors */}
              <div className="mt-8">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                  Available Colors
                </h3>
                <div className="mt-3 flex flex-wrap gap-3">
                  {garment.colors.map((color) => {
                    const bg = COLOR_MAP[color] ?? "#d4d4d4";
                    const isGradient = bg.includes("gradient");
                    return (
                      <div key={color} className="flex flex-col items-center gap-1">
                        <div
                          className="h-8 w-8 rounded-full border border-border shadow-sm"
                          style={
                            isGradient
                              ? { background: bg }
                              : { backgroundColor: bg }
                          }
                          title={color}
                        />
                        <span className="text-xs text-muted-foreground">
                          {color}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Size selector */}
              <div className="mt-8">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                  Size
                </h3>
                <div className="mt-3 flex flex-wrap gap-2">
                  {garment.sizes.map((size) => (
                    <button
                      key={size}
                      onClick={() => setSelectedSize(size)}
                      className={cn(
                        "rounded-lg border px-6 py-3 text-base font-medium shadow-sm transition-all duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none",
                        selectedSize === size
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border bg-card hover:bg-accent"
                      )}
                      style={{ minHeight: "2.625rem" }}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>

              {/* Size chart (collapsible) */}
              <div className="mt-6">
                <button
                  onClick={() => setSizeChartOpen(!sizeChartOpen)}
                  className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground motion-reduce:transition-none"
                >
                  <Ruler className="h-4 w-4" />
                  Size Chart
                  {sizeChartOpen ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </button>

                {sizeChartOpen && (
                  <div className="mt-4 overflow-hidden rounded-xl border border-border shadow-sm">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted">
                          <th className="px-4 py-3 font-semibold">Size</th>
                          <th className="px-4 py-3 font-semibold">
                            Chest (cm)
                          </th>
                          {garment.sizeChart[0]?.lengthCm != null && (
                            <th className="px-4 py-3 font-semibold">
                              Length (cm)
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {garment.sizeChart.map((entry) => (
                          <tr
                            key={entry.size}
                            className={cn(
                              "border-b border-border last:border-0",
                              selectedSize === entry.size && "bg-accent/50"
                            )}
                          >
                            <td className="px-4 py-3 font-medium">
                              {entry.size}
                            </td>
                            <td className="px-4 py-3 text-muted-foreground">
                              {entry.chestMin}–{entry.chestMax}
                            </td>
                            {entry.lengthCm != null && (
                              <td className="px-4 py-3 text-muted-foreground">
                                {entry.lengthCm}
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* CTAs */}
              <div className="mt-10 flex flex-col gap-4 sm:flex-row">
                <Link
                  href={`/try-on?garment=${garment.id}`}
                  className="flex items-center justify-center gap-2 rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
                  style={{ minHeight: "2.625rem" }}
                >
                  Try On This Garment
                </Link>
                <button
                  onClick={() => {
                    setComparison(null);
                    setRecModalOpen(true);
                  }}
                  className="flex items-center justify-center gap-2 rounded-xl border border-border bg-card px-8 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
                  style={{ minHeight: "2.625rem" }}
                >
                  <Ruler className="h-4 w-4" />
                  Get Size Recommendation
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Size recommendation modal */}
      {recModalOpen && (
        <SizeComparisonModal
          garment={garment}
          height={height}
          weight={weight}
          fit={fit}
          comparison={comparison}
          loading={loading}
          onHeightChange={setHeight}
          onWeightChange={setWeight}
          onFitChange={setFit}
          onRecommend={handleRecommend}
          onClose={() => setRecModalOpen(false)}
          onApplySize={(size) => {
            setSelectedSize(size);
            setRecModalOpen(false);
          }}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*  EA-VTON Size Comparison Modal                                              */
/* -------------------------------------------------------------------------- */

function SizeComparisonModal({
  garment,
  height,
  weight,
  fit,
  comparison,
  loading,
  onHeightChange,
  onWeightChange,
  onFitChange,
  onRecommend,
  onClose,
  onApplySize,
}: {
  garment: Garment;
  height: string;
  weight: string;
  fit: "slim" | "regular" | "relaxed";
  comparison: SizeComparisonResult | null;
  loading: boolean;
  onHeightChange: (v: string) => void;
  onWeightChange: (v: string) => void;
  onFitChange: (v: "slim" | "regular" | "relaxed") => void;
  onRecommend: () => void;
  onClose: () => void;
  onApplySize: (size: string) => void;
}) {
  const canSubmit =
    height && weight && Number(height) > 0 && Number(weight) > 0 && !loading;

  const confidenceColors: Record<string, string> = {
    high: "text-green-600 bg-green-50 border-green-200",
    medium: "text-amber-600 bg-amber-50 border-amber-200",
    low: "text-red-600 bg-red-50 border-red-200",
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label="Size recommendation comparison"
    >
      <div className="w-full max-w-2xl rounded-2xl border border-border bg-card p-8 shadow-lg">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold">EA-VTON Size Recommendation</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {garment.brand} {garment.name} — Baseline vs Vietnamese-Aware
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 transition-colors duration-200 hover:bg-accent motion-reduce:transition-none"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Inputs */}
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="rec-height" className="mb-2 block text-sm font-medium">
              Height (cm)
            </label>
            <input
              id="rec-height"
              type="number"
              min="100"
              max="250"
              placeholder="e.g. 156"
              value={height}
              onChange={(e) => onHeightChange(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-4 py-3 text-base placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
          <div>
            <label htmlFor="rec-weight" className="mb-2 block text-sm font-medium">
              Weight (kg)
            </label>
            <input
              id="rec-weight"
              type="number"
              min="30"
              max="200"
              placeholder="e.g. 52"
              value={weight}
              onChange={(e) => onWeightChange(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-4 py-3 text-base placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
        </div>

        {/* Fit preference */}
        <div className="mt-4">
          <label className="mb-2 block text-sm font-medium">Fit Preference</label>
          <div className="flex gap-2">
            {(["slim", "regular", "relaxed"] as const).map((option) => (
              <button
                key={option}
                onClick={() => onFitChange(option)}
                className={cn(
                  "flex-1 rounded-lg border px-4 py-3 text-sm font-medium capitalize transition-all duration-200 motion-reduce:transition-none",
                  fit === option
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-card hover:bg-accent"
                )}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={onRecommend}
          disabled={!canSubmit}
          className={cn(
            "mt-6 w-full rounded-xl px-8 py-4 text-base font-semibold shadow-sm transition-all duration-200 motion-reduce:transition-none",
            canSubmit
              ? "bg-primary text-primary-foreground hover:opacity-90"
              : "cursor-not-allowed bg-muted text-muted-foreground"
          )}
        >
          {loading ? "Analyzing..." : "Compare Models"}
        </button>

        {/* Comparison results */}
        {comparison && (
          <div className="mt-6 space-y-4">
            {/* Side-by-side cards */}
            <div className="grid gap-4 sm:grid-cols-2">
              {/* Baseline */}
              <div className="rounded-xl border border-border bg-muted/30 p-5">
                <div className="flex items-center gap-2">
                  <span className="inline-block h-2 w-2 rounded-full bg-gray-400" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Baseline (US)
                  </p>
                </div>
                <p className="mt-3 text-4xl font-bold text-muted-foreground">
                  {comparison.baseline.recommended_size}
                </p>
                <div className="mt-3 space-y-1 text-sm text-muted-foreground">
                  <p>Chest: {comparison.baseline.estimated_chest_cm} cm</p>
                  <p>Waist: {comparison.baseline.estimated_waist_cm} cm</p>
                  <p>Hip: {comparison.baseline.estimated_hip_cm} cm</p>
                </div>
                <span
                  className={cn(
                    "mt-3 inline-block rounded-full border px-2 py-0.5 text-xs font-semibold capitalize",
                    confidenceColors[comparison.baseline.confidence]
                  )}
                >
                  {comparison.baseline.confidence}
                </span>
              </div>

              {/* EA-VTON */}
              <div
                className={cn(
                  "rounded-xl border-2 p-5",
                  comparison.comparison.size_changed
                    ? "border-green-500 bg-green-50"
                    : "border-primary bg-primary/5"
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="inline-block h-2 w-2 rounded-full bg-green-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-green-700">
                    EA-VTON (Vietnamese)
                  </p>
                </div>
                <p className="mt-3 text-4xl font-bold">
                  {comparison.eavton.recommended_size}
                </p>
                <div className="mt-3 space-y-1 text-sm">
                  <p>Chest: {comparison.eavton.estimated_chest_cm} cm</p>
                  <p>Waist: {comparison.eavton.estimated_waist_cm} cm</p>
                  <p>Hip: {comparison.eavton.estimated_hip_cm} cm</p>
                </div>
                <span
                  className={cn(
                    "mt-3 inline-block rounded-full border px-2 py-0.5 text-xs font-semibold capitalize",
                    confidenceColors[comparison.eavton.confidence]
                  )}
                >
                  {comparison.eavton.confidence}
                </span>
                {comparison.eavton.nearest_body_type && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Body type: {comparison.eavton.nearest_body_type} (
                    {comparison.eavton.nearest_body_type_pct}% of VN population)
                  </p>
                )}
              </div>
            </div>

            {/* Difference explanation */}
            {comparison.comparison.size_changed && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                <p className="text-sm font-semibold text-amber-800">
                  Size difference detected: {Math.abs(comparison.comparison.chest_difference_cm)} cm chest overestimate
                </p>
                <p className="mt-1 text-sm text-amber-700">
                  {comparison.comparison.explanation}
                </p>
              </div>
            )}

            {!comparison.comparison.size_changed && (
              <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
                <p className="text-sm text-blue-700">
                  Both models agree on size, but EA-VTON estimates{" "}
                  {Math.abs(comparison.comparison.chest_difference_cm)} cm smaller chest
                  — closer to Vietnamese body proportions.
                </p>
              </div>
            )}

            {/* Apply button */}
            <button
              onClick={() => onApplySize(comparison.eavton.recommended_size)}
              className="w-full rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 motion-reduce:transition-none"
            >
              Apply {comparison.eavton.recommended_size} (EA-VTON)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
