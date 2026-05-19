"use client";

// intent: dataset visualizer — show everything, cite all sources per DESIGN.md
// status: done
// next: add live progress when scraping/labeling runs in dev mode
// confidence: high

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BookOpen,
  Database,
  ExternalLink,
  Image as ImageIcon,
  Ruler,
  Shield,
  Users,
} from "lucide-react";
import { getDatasetStats, type DatasetStats } from "@/lib/api";

export default function DataPage() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDatasetStats()
      .then(setStats)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to load"),
      );
  }, []);

  if (error) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-12">
        <h1 className="text-2xl font-bold">Dataset</h1>
        <div className="mt-4 rounded-2xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
          <p className="mt-2 text-xs text-muted-foreground">
            Make sure the recommendation service is running at the configured
            URL.
          </p>
        </div>
      </main>
    );
  }

  if (!stats) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-12">
        <h1 className="text-2xl font-bold">Dataset</h1>
        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-24 animate-pulse rounded-2xl border border-border bg-muted/40"
            />
          ))}
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-8 md:py-12">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <Link
          href="/"
          className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-3 py-1.5 text-[11px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <ArrowLeft className="h-3 w-3" />
          Home
        </Link>
        <span className="rounded-full bg-muted px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Dataset visualizer
        </span>
      </div>

      <div className="mb-12 max-w-3xl">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Dataset
        </span>
        <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
          What we built on, exactly
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground md:text-base">
          Every label, every embedding, every person cluster shown live. Six
          source datasets stitched into a single privacy-clean catalog of{" "}
          <span className="font-semibold text-foreground">
            {stats.total_rows.toLocaleString()}
          </span>{" "}
          upper-body items.
        </p>
      </div>

      {/* Top-level stats */}
      <section className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          icon={<Database className="h-4 w-4" />}
          label="Total items"
          value={stats.total_rows.toLocaleString()}
        />
        <StatCard
          icon={<ImageIcon className="h-4 w-4" />}
          label="With CLIP embedding"
          value={`${Math.round((stats.labeling.with_clip_embedding / stats.total_rows) * 100)}%`}
          subtext={`${stats.labeling.with_clip_embedding.toLocaleString()} items`}
        />
        <StatCard
          icon={<Shield className="h-4 w-4" />}
          label="Face-redacted"
          value={`${Math.round((stats.labeling.redacted_face_detected / stats.total_rows) * 100)}%`}
          subtext={`${stats.labeling.redacted_face_detected.toLocaleString()} items`}
        />
        <StatCard
          icon={<Users className="h-4 w-4" />}
          label="Person-clustered"
          value={stats.person_clustering.rows_assigned.toLocaleString()}
          subtext={`anon person_ids`}
        />
      </section>

      {/* Per-source */}
      <Section title="By source" eyebrow="Composition">
        <div className="overflow-hidden rounded-2xl border border-border bg-card">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Source</th>
                <th className="px-4 py-2 text-right font-medium">Items</th>
                <th className="px-4 py-2 text-right font-medium">Share</th>
              </tr>
            </thead>
            <tbody>
              {stats.by_source.map((row) => (
                <tr key={row.source} className="border-t border-border">
                  <td className="px-4 py-2 font-mono text-xs">{row.source}</td>
                  <td className="px-4 py-2 text-right font-mono">
                    {row.n.toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-right text-muted-foreground">
                    {((row.n / stats.total_rows) * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Attribute distributions */}
      <Section title="Attribute distributions" eyebrow="Labels">
        <p className="mb-4 text-sm text-muted-foreground">
          All attribute labels derived via OpenCLIP ViT-B-32 zero-shot
          (laion2b_s34b_b79k). Argmax over 4-8 calibrated text prompts per
          attribute dimension. Run on the face-redacted images, so no facial
          information leaks into the labels.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {(
            [
              ["neckline", "Neckline"],
              ["silhouette", "Silhouette"],
              ["color_temperature", "Color temperature"],
              ["best_season", "Season palette"],
              ["style_personality", "Style personality"],
              ["best_suits_cluster", "Body-cluster suitability"],
            ] as const
          ).map(([key, label]) => (
            <DistCard key={key} title={label} rows={stats.attributes[key]} />
          ))}
        </div>
      </Section>

      {/* Person clustering */}
      <Section title="Person clustering (anon)" eyebrow="Re-identification">
        <p className="mb-4 text-sm text-muted-foreground">
          For items where a face was detected in the un-redacted image, we
          extracted a 512-d OpenCLIP face embedding → ran agglomerative
          clustering (cosine distance, average linkage, eps=0.15) → kept{" "}
          <strong>only the integer cluster id</strong> per item. Embeddings were
          discarded after clustering; no facial biometric data persists in the
          catalog.
        </p>
        <div className="rounded-2xl border border-border bg-card p-4">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Top 10 clusters by size
          </div>
          <div className="mt-3 grid grid-cols-5 gap-2">
            {stats.person_clustering.top_clusters_by_size.map((c, i) => (
              <div
                key={c.person_id}
                className="rounded-xl bg-muted/50 px-2 py-3 text-center"
              >
                <div className="text-[10px] text-muted-foreground">
                  #{i + 1}
                </div>
                <div className="font-mono text-base font-semibold">
                  {c.size}
                </div>
                <div className="text-[9px] text-muted-foreground">items</div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-muted-foreground">
            Each cluster groups items featuring the same catalog model. At
            recommendation time, a user's selfie embedding maps to the nearest
            cluster centroid → those items become a personalized re-ranking
            signal.
          </p>
        </div>
      </Section>

      {/* Size model */}
      <Section title="Size model" eyebrow="Method">
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="flex items-start gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
              <Ruler className="h-4 w-4" />
            </div>
            <div className="flex-1 space-y-3 text-sm">
              <div>
                <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  Variant
                </span>
                <div className="font-mono text-foreground">
                  {stats.size_model.primary}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Metric
                  label="Exact accuracy"
                  value={`${(stats.size_model.exact_accuracy * 100).toFixed(1)}%`}
                />
                <Metric
                  label="Within-1 accuracy"
                  value={`${(stats.size_model.within1_accuracy * 100).toFixed(1)}%`}
                />
              </div>
              <Detail
                label="Training data"
                value={stats.size_model.training_data}
              />
              <Detail
                label="Reweighting"
                value={stats.size_model.reweighting}
              />
            </div>
          </div>
        </div>
      </Section>

      {/* Citations */}
      <Section
        title="Source citations"
        eyebrow="Provenance"
        icon={<BookOpen className="h-4 w-4" />}
      >
        <div className="space-y-3">
          {stats.sources.map((s) => (
            <CitationCard
              key={s.id}
              name={s.name}
              citation={s.citation}
              url={s.url}
              license={s.license}
            />
          ))}
        </div>
      </Section>

      <Section
        title="Anthropometric & methodology sources"
        eyebrow="Cited methods"
        icon={<BookOpen className="h-4 w-4" />}
      >
        <div className="space-y-3">
          {stats.anthropometric_sources.map((s) => (
            <CitationCard
              key={s.id}
              name={s.name}
              citation={s.citation}
              url={s.url}
            />
          ))}
        </div>
      </Section>

      <p className="mt-8 text-center text-[11px] text-muted-foreground">
        All derived labels licensed CC BY 4.0. Source images keep their upstream
        licenses — see{" "}
        <Link
          href="/privacy"
          className="underline underline-offset-4 hover:text-foreground"
        >
          Privacy
        </Link>
        .
      </p>
    </main>
  );
}

// ── Sub-components ──

function Section({
  title,
  eyebrow,
  icon,
  children,
}: {
  title: string;
  eyebrow: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-12">
      <div className="mb-4 flex items-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {eyebrow}
        </span>
      </div>
      <h2 className="mb-4 flex items-center gap-2 text-xl font-bold tracking-tight">
        {icon}
        {title}
      </h2>
      {children}
    </section>
  );
}

function StatCard({
  icon,
  label,
  value,
  subtext,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtext?: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="text-[11px] font-medium uppercase tracking-wider">
          {label}
        </span>
      </div>
      <div className="mt-2 font-mono text-2xl font-bold">{value}</div>
      {subtext && (
        <div className="text-[11px] text-muted-foreground">{subtext}</div>
      )}
    </div>
  );
}

function DistCard({
  title,
  rows,
}: {
  title: string;
  rows: { value: string; n: number }[];
}) {
  const max = Math.max(...rows.map((r) => r.n), 1);
  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div
            key={String(r.value)}
            className="flex items-center gap-2 text-xs"
          >
            <span className="w-32 shrink-0 truncate text-muted-foreground">
              {String(r.value)}
            </span>
            <div className="relative h-4 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="absolute inset-y-0 left-0 bg-foreground/85"
                style={{ width: `${(r.n / max) * 100}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-[11px]">
              {r.n.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-muted/50 px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-base font-semibold">{value}</div>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <div className="text-xs leading-relaxed text-foreground">{value}</div>
    </div>
  );
}

function CitationCard({
  name,
  citation,
  url,
  license,
}: {
  name: string;
  citation: string;
  url: string;
  license?: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm font-semibold hover:underline"
          >
            {name}
            <ExternalLink className="h-3 w-3" />
          </a>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {citation}
          </p>
        </div>
        {license && (
          <span className="shrink-0 rounded-full bg-muted px-2 py-1 text-[10px] font-medium text-muted-foreground">
            {license}
          </span>
        )}
      </div>
    </div>
  );
}
