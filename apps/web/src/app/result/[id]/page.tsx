"use client";

import { useState, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import {
  ChevronRight,
  Shirt,
  Sparkles,
  Ruler,
  ArrowLeft,
} from "lucide-react";
import { formatPrice } from "@/lib/utils";
import { GARMENTS } from "@/data/garments";

/* ------------------------------------------------------------------ */
/*  Demo assets — in production these would come from the API          */
/* ------------------------------------------------------------------ */

const DEMO_BEFORE =
  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=800&fit=crop";
const DEMO_AFTER =
  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=800&fit=crop";

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function ResultPage() {
  const { id } = useParams<{ id: string }>();

  /* Find the garment used in this "result" — fallback to first garment */
  const garment = GARMENTS.find((g) => g.id === id) ?? GARMENTS[0];
  const garmentImage =
    garment.images.find((i) => i.type === "tryon_input")?.url ??
    garment.images[0].url;

  /* ---- slider state ---- */
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateSlider = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updateSlider(e.clientX);
    },
    [updateSlider],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      updateSlider(e.clientX);
    },
    [updateSlider],
  );

  const handlePointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  /* keyboard support */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowLeft") setSliderPos((p) => Math.max(0, p - 2));
      if (e.key === "ArrowRight") setSliderPos((p) => Math.min(100, p + 2));
    },
    [],
  );

  return (
    <section className="py-16">
      <div className="mx-auto max-w-4xl px-6">
        {/* Back link */}
        <Link
          href="/try-on"
          className="mb-8 inline-flex items-center gap-2 text-sm font-medium text-muted-foreground transition-all duration-200 hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Try-On
        </Link>

        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight">
            Try-On Result
          </h1>
          <p className="mt-2 text-muted-foreground">
            Drag the slider to compare before and after.
          </p>
        </div>

        {/* Confidence badge */}
        <div className="mt-6 flex justify-center">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1.5 text-xs font-semibold text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
            <Sparkles className="h-3.5 w-3.5" />
            Demo Mode &mdash; High Confidence
          </span>
        </div>

        {/* ---- Before/After comparison slider ---- */}
        <div className="mt-10">
          <div
            ref={containerRef}
            className="relative mx-auto aspect-[3/4] max-w-md select-none overflow-hidden rounded-xl border border-border shadow-sm"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            role="slider"
            aria-label="Before and after comparison slider"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(sliderPos)}
            tabIndex={0}
            onKeyDown={handleKeyDown}
          >
            {/* After (full width behind) */}
            <div className="absolute inset-0">
              <Image
                src={DEMO_AFTER}
                alt="After — with garment"
                fill
                className="object-cover"
                unoptimized
                draggable={false}
              />
              {/* Garment overlay for "after" side */}
              <div className="absolute inset-0">
                <Image
                  src={garmentImage}
                  alt={garment.name}
                  fill
                  className="object-contain opacity-50 mix-blend-multiply dark:mix-blend-screen"
                  unoptimized
                  draggable={false}
                />
              </div>
              <div className="absolute bottom-3 right-3 rounded-md bg-black/70 px-2.5 py-1 text-xs font-medium text-white backdrop-blur">
                After
              </div>
            </div>

            {/* Before (clipped by slider position) */}
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            >
              <div className="absolute inset-0">
                <Image
                  src={DEMO_BEFORE}
                  alt="Before — original photo"
                  fill
                  className="object-cover"
                  unoptimized
                  draggable={false}
                />
              </div>
              <div className="absolute bottom-3 left-3 rounded-md bg-black/70 px-2.5 py-1 text-xs font-medium text-white backdrop-blur">
                Before
              </div>
            </div>

            {/* Slider handle */}
            <div
              className="absolute top-0 bottom-0 z-10 w-1 -translate-x-1/2 bg-white shadow-lg"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute left-1/2 top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-primary text-primary-foreground shadow-lg">
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  className="shrink-0"
                >
                  <path
                    d="M4 3L1 8L4 13M12 3L15 8L12 13"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Garment info card */}
        <div className="mx-auto mt-10 max-w-md rounded-xl border border-border bg-card p-6 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-lg">
              <Image
                src={
                  garment.images.find((i) => i.type === "display")?.url ??
                  garment.images[0].url
                }
                alt={garment.name}
                fill
                className="object-cover"
                unoptimized
              />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-muted-foreground">
                {garment.brand}
              </p>
              <p className="truncate font-semibold">{garment.name}</p>
              <p className="text-sm font-semibold">
                {formatPrice(garment.price)}
              </p>
            </div>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
            {garment.description}
          </p>
          <div className="mt-4 flex flex-wrap gap-1.5">
            {garment.sizes.map((s) => (
              <span
                key={s}
                className="rounded-full border border-border px-2 py-1 text-xs font-medium"
              >
                {s}
              </span>
            ))}
          </div>
        </div>

        {/* Action buttons */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/try-on"
            className="flex min-h-[2.625rem] items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
          >
            <Shirt className="h-4 w-4" />
            Try Another
          </Link>
          <Link
            href={`/product/${garment.id}`}
            className="flex min-h-[2.625rem] items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
          >
            <Ruler className="h-4 w-4" />
            Get Size Recommendation
            <ChevronRight className="h-4 w-4" />
          </Link>
        </div>

        {/* Processing details */}
        <div className="mt-10 rounded-xl border border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Processing info:</span>{" "}
            Result generated in 3.2s &bull; Confidence: 87% (High) &bull; Model: demo-v1
          </p>
          <p className="mt-1">
            This is a simulated result for demonstration purposes. No actual AI processing was performed.
          </p>
        </div>
      </div>
    </section>
  );
}
