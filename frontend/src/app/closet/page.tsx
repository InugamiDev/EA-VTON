// intent: Closet page — unified try-on + size recommendation + occasion filtering
// status: done
// next: integrate real VTON backend when model is deployed
// confidence: high

"use client";

import { useState, useMemo, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  Ruler,
  Shirt,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  X,
  Check,
  AlertTriangle,
} from "lucide-react";
import { cn, formatPrice } from "@/lib/utils";
import { GARMENTS } from "@/data/garments";
import { compareSizeRecommendation } from "@/lib/api";
import type { SizeComparisonResult } from "@/lib/api";
import type { Garment } from "@/types";

/* -------------------------------------------------------------------------- */
/*  Occasion → garment type mapping                                           */
/* -------------------------------------------------------------------------- */

const OCCASIONS = [
  { id: "all", label: "All", types: null },
  { id: "casual", label: "Casual", types: ["t-shirt", "hoodie", "polo"] },
  { id: "business", label: "Business / Office", types: ["shirt", "sweater", "polo"] },
  { id: "outdoor", label: "Outdoor / Active", types: ["jacket", "vest", "hoodie"] },
  { id: "date", label: "Date Night", types: ["shirt", "polo", "sweater"] },
  { id: "layering", label: "Layering", types: ["vest", "jacket", "hoodie", "sweater"] },
] as const;

type OccasionId = (typeof OCCASIONS)[number]["id"];

/* -------------------------------------------------------------------------- */
/*  Size estimation (client-side, matches ea_size_engine.py)                   */
/* -------------------------------------------------------------------------- */

function estimateChestVN(h: number, w: number): number {
  return Math.round((h * 0.485 + w * 0.22 - 2.5) * 10) / 10;
}

function findBestSize(chest: number, chart: Garment["sizeChart"]): string | null {
  let best: string | null = null;
  let bestDist = Infinity;
  for (const entry of chart) {
    const mid = (entry.chestMin + entry.chestMax) / 2;
    const dist = Math.abs(chest - mid);
    if (dist < bestDist) {
      bestDist = dist;
      best = entry.size;
    }
  }
  return best;
}

function sizeConfidence(
  chest: number,
  size: string,
  chart: Garment["sizeChart"]
): "high" | "medium" | "low" {
  const entry = chart.find((e) => e.size === size);
  if (!entry) return "low";
  if (chest >= entry.chestMin && chest <= entry.chestMax) {
    const center = (entry.chestMin + entry.chestMax) / 2;
    const half = (entry.chestMax - entry.chestMin) / 2;
    return Math.abs(chest - center) / half < 0.5 ? "high" : "medium";
  }
  const overshoot = Math.min(
    Math.abs(chest - entry.chestMin),
    Math.abs(chest - entry.chestMax)
  );
  return overshoot <= 2 ? "medium" : "low";
}

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function getDisplayImage(garment: Garment): string {
  return garment.images.find((i) => i.type === "display")?.url ?? garment.images[0]?.url ?? "";
}

const TYPE_LABELS: Record<Garment["type"], string> = {
  "t-shirt": "T-Shirt",
  shirt: "Shirt",
  hoodie: "Hoodie",
  jacket: "Jacket",
  sweater: "Sweater",
  polo: "Polo",
  vest: "Vest",
};

const CONFIDENCE_STYLES = {
  high: "bg-green-100 text-green-700 border-green-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-red-100 text-red-700 border-red-200",
};

/* -------------------------------------------------------------------------- */
/*  Page                                                                       */
/* -------------------------------------------------------------------------- */

export default function ClosetPage() {
  // Body profile
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [profileOpen, setProfileOpen] = useState(true);

  // Filters
  const [occasion, setOccasion] = useState<OccasionId>("all");

  // Comparison modal
  const [selectedGarment, setSelectedGarment] = useState<Garment | null>(null);
  const [comparison, setComparison] = useState<SizeComparisonResult | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const hasProfile = height && weight && Number(height) > 100 && Number(weight) > 30;
  const chestEstimate = hasProfile ? estimateChestVN(Number(height), Number(weight)) : null;

  // Filter garments by occasion
  const filteredGarments = useMemo(() => {
    const occ = OCCASIONS.find((o) => o.id === occasion);
    if (!occ || !occ.types) return GARMENTS;
    return GARMENTS.filter((g) => (occ.types as readonly string[]).includes(g.type));
  }, [occasion]);

  // Enrich garments with size recommendation
  const enrichedGarments = useMemo(() => {
    return filteredGarments.map((g) => {
      if (!chestEstimate) return { garment: g, recSize: null, confidence: null as "high" | "medium" | "low" | null };
      const recSize = findBestSize(chestEstimate, g.sizeChart);
      const conf = recSize ? sizeConfidence(chestEstimate, recSize, g.sizeChart) : null;
      return { garment: g, recSize, confidence: conf };
    });
  }, [filteredGarments, chestEstimate]);

  // Sort: high confidence first, then medium, then low
  const sortedGarments = useMemo(() => {
    if (!hasProfile) return enrichedGarments;
    const order = { high: 0, medium: 1, low: 2 };
    return [...enrichedGarments].sort((a, b) => {
      const aOrder = a.confidence ? order[a.confidence] : 3;
      const bOrder = b.confidence ? order[b.confidence] : 3;
      return aOrder - bOrder;
    });
  }, [enrichedGarments, hasProfile]);

  const handleCompare = useCallback(async (garment: Garment) => {
    if (!hasProfile) return;
    setSelectedGarment(garment);
    setComparison(null);
    setLoadingComparison(true);
    try {
      const result = await compareSizeRecommendation({
        garment_id: garment.id,
        height_cm: Number(height),
        weight_kg: Number(weight),
        fit_preference: "regular",
      });
      setComparison(result);
    } catch {
      setComparison(null);
    } finally {
      setLoadingComparison(false);
    }
  }, [hasProfile, height, weight]);

  return (
    <>
      <section className="border-b border-border bg-muted/30">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">My Closet</h1>
              <p className="mt-2 text-muted-foreground">
                Personalized recommendations sized for your body. Powered by EA-VTON.
              </p>
            </div>
            <button
              onClick={() => setProfileOpen(!profileOpen)}
              className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2.5 text-sm font-medium shadow-sm transition-colors hover:bg-accent"
            >
              <Ruler className="h-4 w-4" />
              Body Profile
              {profileOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          </div>

          {/* Body profile inputs */}
          {profileOpen && (
            <div className="mt-6 rounded-xl border border-border bg-card p-6 shadow-sm">
              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <label htmlFor="closet-height" className="mb-1.5 block text-sm font-medium">
                    Height (cm)
                  </label>
                  <input
                    id="closet-height"
                    type="number"
                    min="100"
                    max="250"
                    placeholder="e.g. 156"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  />
                </div>
                <div>
                  <label htmlFor="closet-weight" className="mb-1.5 block text-sm font-medium">
                    Weight (kg)
                  </label>
                  <input
                    id="closet-weight"
                    type="number"
                    min="30"
                    max="200"
                    placeholder="e.g. 52"
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                    className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  />
                </div>
                <div className="flex items-end">
                  {hasProfile ? (
                    <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 px-4 py-2.5 text-sm text-green-700">
                      <Check className="h-4 w-4" />
                      Estimated chest: {chestEstimate} cm (VN model)
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 rounded-lg border border-border bg-muted px-4 py-2.5 text-sm text-muted-foreground">
                      <AlertTriangle className="h-4 w-4" />
                      Enter measurements to see recommendations
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Occasion filter */}
      <section className="border-b border-border">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center gap-2 overflow-x-auto">
            <Sparkles className="h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="shrink-0 text-sm font-medium text-muted-foreground">Occasion:</span>
            {OCCASIONS.map((occ) => (
              <button
                key={occ.id}
                onClick={() => setOccasion(occ.id)}
                className={cn(
                  "shrink-0 rounded-full border px-4 py-1.5 text-sm font-medium transition-colors",
                  occasion === occ.id
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-card hover:bg-accent"
                )}
              >
                {occ.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Garment grid */}
      <section className="py-8">
        <div className="mx-auto max-w-7xl px-6">
          <p className="mb-6 text-sm text-muted-foreground">
            {sortedGarments.length} items
            {hasProfile && " — sorted by fit confidence"}
          </p>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {sortedGarments.map(({ garment, recSize, confidence }) => (
              <div
                key={garment.id}
                className="group overflow-hidden rounded-xl border border-border bg-card shadow-sm transition-shadow hover:shadow-md"
              >
                {/* Image */}
                <div className="relative aspect-[3/4] overflow-hidden bg-muted">
                  <Image
                    src={getDisplayImage(garment)}
                    alt={garment.name}
                    fill
                    unoptimized
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
                  />

                  {/* Size badge */}
                  {recSize && confidence && (
                    <div className="absolute left-3 top-3 flex items-center gap-1.5">
                      <span
                        className={cn(
                          "rounded-md border px-2 py-1 text-xs font-bold shadow-sm",
                          CONFIDENCE_STYLES[confidence]
                        )}
                      >
                        {recSize}
                      </span>
                      {confidence === "high" && (
                        <span className="rounded-md border border-green-200 bg-green-100 px-1.5 py-1 text-[10px] font-semibold text-green-700 shadow-sm">
                          PERFECT FIT
                        </span>
                      )}
                    </div>
                  )}

                  {/* Type tag */}
                  <span className="absolute bottom-3 left-3 rounded-md border border-border bg-card/90 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground shadow-sm backdrop-blur-sm">
                    {TYPE_LABELS[garment.type]}
                  </span>
                </div>

                {/* Info */}
                <div className="p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {garment.brand}
                  </p>
                  <h3 className="mt-1 text-sm font-semibold leading-snug line-clamp-2">
                    {garment.name}
                  </h3>
                  <p className="mt-2 text-base font-bold">{formatPrice(garment.price)}</p>

                  {/* Actions */}
                  <div className="mt-4 flex gap-2">
                    {hasProfile ? (
                      <button
                        onClick={() => handleCompare(garment)}
                        className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2.5 text-xs font-semibold text-primary-foreground transition-opacity hover:opacity-90"
                      >
                        <Ruler className="h-3.5 w-3.5" />
                        Compare Sizes
                      </button>
                    ) : (
                      <Link
                        href={`/product/${garment.id}`}
                        className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2.5 text-xs font-semibold text-primary-foreground transition-opacity hover:opacity-90"
                      >
                        View Details
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    )}
                    <Link
                      href={`/try-on?garment=${garment.id}`}
                      className="flex items-center justify-center gap-1.5 rounded-lg border border-border bg-card px-3 py-2.5 text-xs font-semibold transition-colors hover:bg-accent"
                    >
                      <Shirt className="h-3.5 w-3.5" />
                      Try On
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison modal */}
      {selectedGarment && (
        <ComparisonModal
          garment={selectedGarment}
          comparison={comparison}
          loading={loadingComparison}
          onClose={() => {
            setSelectedGarment(null);
            setComparison(null);
          }}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*  Comparison modal (inline)                                                  */
/* -------------------------------------------------------------------------- */

function ComparisonModal({
  garment,
  comparison,
  loading,
  onClose,
}: {
  garment: Garment;
  comparison: SizeComparisonResult | null;
  loading: boolean;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-6 shadow-lg">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="relative h-14 w-14 overflow-hidden rounded-lg border border-border bg-muted">
              <Image
                src={getDisplayImage(garment)}
                alt={garment.name}
                fill
                unoptimized
                className="object-cover"
              />
            </div>
            <div>
              <h3 className="font-semibold">{garment.name}</h3>
              <p className="text-sm text-muted-foreground">{garment.brand} — {formatPrice(garment.price)}</p>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-2 hover:bg-accent">
            <X className="h-5 w-5" />
          </button>
        </div>

        {loading && (
          <div className="mt-6 flex items-center justify-center py-8 text-sm text-muted-foreground">
            Analyzing body measurements...
          </div>
        )}

        {comparison && (
          <div className="mt-6 space-y-4">
            {/* Side by side */}
            <div className="grid gap-3 sm:grid-cols-2">
              {/* Baseline */}
              <div className="rounded-xl border border-border bg-muted/30 p-4">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  Baseline (US)
                </p>
                <p className="mt-2 text-3xl font-bold text-muted-foreground/60">
                  {comparison.baseline.recommended_size}
                </p>
                <div className="mt-2 space-y-0.5 text-xs text-muted-foreground">
                  <p>Chest: {comparison.baseline.estimated_chest_cm} cm</p>
                  <p>Waist: {comparison.baseline.estimated_waist_cm} cm</p>
                  <p>Hip: {comparison.baseline.estimated_hip_cm} cm</p>
                </div>
              </div>

              {/* EA-VTON */}
              <div
                className={cn(
                  "rounded-xl border-2 p-4",
                  comparison.comparison.size_changed
                    ? "border-green-500 bg-green-50"
                    : "border-primary bg-primary/5"
                )}
              >
                <p className="text-[10px] font-bold uppercase tracking-wider text-green-700">
                  EA-VTON (Vietnamese)
                </p>
                <p className="mt-2 text-3xl font-bold">
                  {comparison.eavton.recommended_size}
                </p>
                <div className="mt-2 space-y-0.5 text-xs">
                  <p>Chest: {comparison.eavton.estimated_chest_cm} cm</p>
                  <p>Waist: {comparison.eavton.estimated_waist_cm} cm</p>
                  <p>Hip: {comparison.eavton.estimated_hip_cm} cm</p>
                </div>
                {comparison.eavton.nearest_body_type && (
                  <p className="mt-2 text-[10px] text-muted-foreground">
                    Body type: {comparison.eavton.nearest_body_type} ({comparison.eavton.nearest_body_type_pct}% of VN population)
                  </p>
                )}
              </div>
            </div>

            {/* Explanation */}
            {comparison.comparison.size_changed ? (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
                <p className="text-xs font-semibold text-amber-800">
                  Baseline oversizes by {Math.abs(comparison.comparison.chest_difference_cm)} cm
                </p>
                <p className="mt-1 text-xs text-amber-700">
                  {comparison.comparison.explanation}
                </p>
              </div>
            ) : (
              <div className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3">
                <p className="text-xs text-blue-700">
                  Both models agree on {comparison.baseline.recommended_size}, but EA-VTON
                  estimates {Math.abs(comparison.comparison.chest_difference_cm)} cm smaller
                  chest — closer to Vietnamese proportions.
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Link
                href={`/try-on?garment=${garment.id}`}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:opacity-90"
              >
                <Shirt className="h-4 w-4" />
                Try On ({comparison.eavton.recommended_size})
              </Link>
              <Link
                href={`/product/${garment.id}`}
                className="flex items-center justify-center gap-2 rounded-xl border border-border bg-card px-4 py-3 text-sm font-semibold hover:bg-accent"
              >
                Details
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
