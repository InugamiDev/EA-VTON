"use client";

// intent: cold-start preference UX — 18 stratified seeds → pick 3+ → CLIP-centroid re-rank
// status: done
// next: persist picks in profile store; let user A/B-compare base vs calibrated
// confidence: high

import { useCallback, useEffect, useState } from "react";
import {
  getStyleSeedItems,
  calibrateUpperStyle,
  type StyleSeedItem,
  type UpperStyleCalibrateResult,
  type UpperStyleRecommendationRequest,
} from "@/lib/api";

const MIN_PICKS = 3;
const MAX_PICKS = 8;

type Props = {
  baseRequest: UpperStyleRecommendationRequest | null;
  className?: string;
};

export function CalibrationPanel({ baseRequest, className }: Props) {
  const [seedsLoading, setSeedsLoading] = useState(false);
  const [seedsError, setSeedsError] = useState<string | null>(null);
  const [seeds, setSeeds] = useState<StyleSeedItem[]>([]);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [calLoading, setCalLoading] = useState(false);
  const [calError, setCalError] = useState<string | null>(null);
  const [calResult, setCalResult] = useState<UpperStyleCalibrateResult | null>(
    null,
  );

  const loadSeeds = useCallback(async () => {
    setSeedsLoading(true);
    setSeedsError(null);
    setCalResult(null);
    setPicked(new Set());
    try {
      const size = baseRequest?.predicted_size ?? "M";
      const res = await getStyleSeedItems(size, 18);
      setSeeds(res.items);
    } catch (err) {
      setSeedsError(
        err instanceof Error ? err.message : "Failed to load seeds",
      );
    } finally {
      setSeedsLoading(false);
    }
  }, [baseRequest?.predicted_size]);

  // Auto-load when a size is first available
  useEffect(() => {
    if (baseRequest && seeds.length === 0 && !seedsLoading && !seedsError) {
      void loadSeeds();
    }
  }, [baseRequest, seeds.length, seedsLoading, seedsError, loadSeeds]);

  const togglePick = (id: string) => {
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < MAX_PICKS) {
        next.add(id);
      }
      return next;
    });
  };

  const canCalibrate =
    baseRequest !== null &&
    picked.size >= MIN_PICKS &&
    seeds.length > 0 &&
    !calLoading;

  const handleCalibrate = async () => {
    if (!baseRequest) return;
    setCalLoading(true);
    setCalError(null);
    try {
      const res = await calibrateUpperStyle({
        ...baseRequest,
        liked_garment_ids: Array.from(picked),
        preference_weight: 0.3,
      });
      setCalResult(res);
    } catch (err) {
      setCalError(err instanceof Error ? err.message : "Calibration failed");
    } finally {
      setCalLoading(false);
    }
  };

  if (!baseRequest) {
    return (
      <div className={className}>
        <div className="rounded-3xl border border-border bg-card p-6 text-sm text-muted-foreground">
          Predict a size first to unlock cold-start style calibration.
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="rounded-3xl border border-border bg-card p-6">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="text-base font-semibold">Personalize with picks</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Pick {MIN_PICKS}–{MAX_PICKS} items you find appealing. We use a
              CLIP embedding centroid of your picks to re-rank the top-K. The
              seed grid is stratified across neckline × silhouette × color
              temperature so the spread is diverse.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void loadSeeds()}
            disabled={seedsLoading}
            className="rounded-full border border-border bg-background px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-50"
          >
            {seedsLoading ? "Loading…" : "Refresh seeds"}
          </button>
        </div>

        {seedsError && (
          <div className="mb-3 rounded-xl bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {seedsError}
          </div>
        )}

        {seeds.length > 0 && (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {seeds.map((s) => {
              const isPicked = picked.has(s.item_id);
              return (
                <button
                  type="button"
                  key={s.item_id}
                  onClick={() => togglePick(s.item_id)}
                  className={`group relative overflow-hidden rounded-2xl border-2 transition-all ${
                    isPicked
                      ? "border-foreground ring-2 ring-foreground/30"
                      : "border-transparent hover:border-border"
                  }`}
                  title={`${s.attributes.neckline} · ${s.attributes.silhouette} · ${s.attributes.color_temperature}`}
                >
                  <div
                    className="aspect-[3/4] w-full"
                    style={{ backgroundColor: s.primary_color_hex || "#ddd" }}
                  >
                    {s.image_url && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={s.image_url}
                        alt={s.title}
                        className="h-full w-full object-cover"
                        loading="lazy"
                      />
                    )}
                  </div>
                  {isPicked && (
                    <div className="absolute right-1 top-1 rounded-full bg-foreground px-2 py-0.5 text-[10px] font-bold text-background">
                      ✓
                    </div>
                  )}
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 text-[10px] font-medium text-white">
                    {s.attributes.silhouette}
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="text-xs text-muted-foreground">
            {picked.size} of {MAX_PICKS} picked
            {picked.size < MIN_PICKS &&
              ` · pick ${MIN_PICKS - picked.size} more`}
          </span>
          <button
            type="button"
            onClick={handleCalibrate}
            disabled={!canCalibrate}
            className="rounded-full bg-foreground px-5 py-2 text-sm font-semibold text-background transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {calLoading ? "Calibrating…" : "Calibrate recommendations"}
          </button>
        </div>

        {calError && (
          <div className="mt-3 rounded-xl bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {calError}
          </div>
        )}

        {calResult && (
          <div className="mt-6 space-y-3">
            <div className="flex items-baseline justify-between">
              <h4 className="text-sm font-semibold">
                Calibrated top {calResult.items.length}
              </h4>
              <span className="text-[10px] text-muted-foreground">
                δ = {calResult.calibration.preference_weight} · re-ranked from a
                pool of {calResult.calibration.base_pool_size}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
              {calResult.items.map((it) => (
                <div
                  key={it.item_id}
                  className="overflow-hidden rounded-2xl border border-border bg-background"
                >
                  <div
                    className="aspect-[3/4] w-full"
                    style={{
                      backgroundColor: "#eee",
                    }}
                  >
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
                      <span>{it.total_score.toFixed(3)}</span>
                    </div>
                    {typeof it.preference_score === "number" && (
                      <div className="flex items-center gap-1 text-[10px]">
                        <span className="rounded-full bg-foreground/10 px-1.5 py-0.5">
                          pref {it.preference_score.toFixed(2)}
                        </span>
                        {typeof it.base_ffm_score === "number" && (
                          <span className="rounded-full bg-foreground/5 px-1.5 py-0.5">
                            ffm {it.base_ffm_score.toFixed(2)}
                          </span>
                        )}
                      </div>
                    )}
                    <div className="truncate text-xs font-medium">
                      {it.title}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
