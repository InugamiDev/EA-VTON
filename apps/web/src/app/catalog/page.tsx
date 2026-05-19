"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Shirt, SlidersHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatPrice } from "@/lib/utils";
import { GARMENTS } from "@/data/garments";
import type { Garment } from "@/types";

const GARMENT_TYPES = [
  { label: "All", value: "all" },
  { label: "T-Shirts", value: "t-shirt" },
  { label: "Shirts", value: "shirt" },
  { label: "Hoodies", value: "hoodie" },
  { label: "Jackets", value: "jacket" },
  { label: "Polos", value: "polo" },
  { label: "Sweaters", value: "sweater" },
  { label: "Vests", value: "vest" },
] as const;

type FilterValue = (typeof GARMENT_TYPES)[number]["value"];

function getDisplayImage(garment: Garment): string {
  const display = garment.images.find((img) => img.type === "display");
  return display?.url ?? garment.images[0]?.url ?? "";
}

function getTypeLabel(type: Garment["type"]): string {
  const map: Record<Garment["type"], string> = {
    "t-shirt": "T-Shirt",
    shirt: "Shirt",
    hoodie: "Hoodie",
    jacket: "Jacket",
    polo: "Polo",
    sweater: "Sweater",
    vest: "Vest",
  };
  return map[type];
}

export default function CatalogPage() {
  const [activeFilter, setActiveFilter] = useState<FilterValue>("all");

  const filtered =
    activeFilter === "all"
      ? GARMENTS
      : GARMENTS.filter((g) => g.type === activeFilter);

  return (
    <>
      {/* Hero */}
      <section className="border-b border-border py-12 md:py-16">
        <div className="mx-auto max-w-6xl px-6">
          <div className="flex flex-col gap-2">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Catalog
            </span>
            <h1 className="text-balance text-3xl font-bold tracking-tight md:text-4xl">
              Browse the catalog
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground md:text-base">
              Essentials catalog used by the recommender as the{" "}
              <Link
                href="/live-size"
                className="underline underline-offset-4 hover:text-foreground"
              >
                live-size flow
              </Link>
              . Pick any garment to see details or try it on virtually.
            </p>
          </div>
        </div>
      </section>

      {/* Filters + Grid */}
      <section className="py-10 md:py-12">
        <div className="mx-auto max-w-6xl px-6">
          {/* Filter bar */}
          <div className="mb-8 flex flex-wrap items-center gap-2">
            <SlidersHorizontal className="mr-1 h-3.5 w-3.5 text-muted-foreground" />
            {GARMENT_TYPES.map((type) => (
              <button
                key={type.value}
                onClick={() => setActiveFilter(type.value)}
                className={cn(
                  "inline-flex h-9 items-center rounded-full px-4 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  activeFilter === type.value
                    ? "bg-foreground text-background"
                    : "border border-border bg-card text-foreground hover:bg-accent",
                )}
              >
                {type.label}
              </button>
            ))}
          </div>

          {/* Results count */}
          <p className="mb-6 text-sm text-muted-foreground">
            Showing {filtered.length}{" "}
            {filtered.length === 1 ? "garment" : "garments"}
          </p>

          {/* Grid */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 md:gap-6 lg:grid-cols-4">
            {filtered.map((garment) => (
              <GarmentCard key={garment.id} garment={garment} />
            ))}
          </div>

          {filtered.length === 0 && (
            <div className="py-20 text-center">
              <Shirt className="mx-auto h-12 w-12 text-muted-foreground/40" />
              <p className="mt-4 text-lg font-medium text-muted-foreground">
                No garments found for this filter.
              </p>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

function GarmentCard({ garment }: { garment: Garment }) {
  const imageUrl = getDisplayImage(garment);
  const sizeRange =
    garment.sizes.length > 1
      ? `${garment.sizes[0]}–${garment.sizes[garment.sizes.length - 1]}`
      : garment.sizes[0];

  return (
    <Link
      href={`/product/${garment.id}`}
      className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
    >
      {/* Image */}
      <div className="relative aspect-[3/4] overflow-hidden bg-muted">
        <Image
          src={imageUrl}
          alt={garment.name}
          fill
          unoptimized
          className="object-cover transition-transform duration-300 group-hover:scale-105 motion-reduce:transition-none"
          sizes="(max-width: 768px) 50vw, (max-width: 1024px) 33vw, 25vw"
        />
      </div>

      {/* Content */}
      <div className="p-4 md:p-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {garment.brand}
        </p>
        <h3 className="mt-1 line-clamp-2 text-base font-semibold leading-snug">
          {garment.name}
        </h3>
        <p className="mt-2 text-lg font-bold">{formatPrice(garment.price)}</p>

        {/* Badges */}
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="inline-flex items-center rounded-full bg-accent px-2 py-1 text-xs font-medium text-accent-foreground">
            {sizeRange}
          </span>
          <span className="inline-flex items-center rounded-full bg-accent px-2 py-1 text-xs font-medium text-accent-foreground">
            {getTypeLabel(garment.type)}
          </span>
        </div>
      </div>
    </Link>
  );
}
