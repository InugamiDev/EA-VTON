"use client";

// intent: container card for a single step in the guided journey
// status: done — applies surface-card tokens from DESIGN.md
// next: animate expand/collapse on step change (respect prefers-reduced-motion)
// confidence: high

import { ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

type Props = {
  eyebrow?: string;
  title: string;
  description?: string;
  state: "active" | "complete" | "upcoming";
  /** Only used when state === "complete"; a 1-line summary shown in the
   * collapsed pill so the user can see what they did without expanding. */
  collapsedSummary?: string;
  /** Click handler to re-open a completed step. */
  onExpand?: () => void;
  /** Sticky bottom CTA — anchored INSIDE the card on mobile and below content
   * on desktop. */
  primaryAction?: ReactNode;
  children?: ReactNode;
  className?: string;
};

export function JourneyStep({
  eyebrow,
  title,
  description,
  state,
  collapsedSummary,
  onExpand,
  primaryAction,
  children,
  className,
}: Props) {
  if (state === "complete" && collapsedSummary) {
    return (
      <button
        type="button"
        onClick={onExpand}
        className={cn(
          "group flex w-full items-center justify-between gap-3 rounded-2xl border border-border bg-card px-4 py-3 text-left shadow-sm transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          className,
        )}
        aria-label={`Re-open: ${title}`}
      >
        <div className="flex min-w-0 flex-1 flex-col">
          {eyebrow && (
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              {eyebrow}
            </span>
          )}
          <div className="flex items-baseline gap-2">
            <span className="truncate text-sm font-semibold">{title}</span>
            <span className="truncate text-xs text-muted-foreground">
              {collapsedSummary}
            </span>
          </div>
        </div>
        <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:rotate-180" />
      </button>
    );
  }

  if (state === "upcoming") {
    return (
      <div
        className={cn(
          "rounded-2xl border border-dashed border-border px-4 py-3 opacity-60",
          className,
        )}
      >
        <div className="flex flex-col">
          {eyebrow && (
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              {eyebrow}
            </span>
          )}
          <span className="text-sm font-medium text-muted-foreground">
            {title}
          </span>
        </div>
      </div>
    );
  }

  // active
  return (
    <section
      className={cn(
        "flex flex-col gap-4 rounded-3xl border border-border bg-card p-5 shadow-sm md:p-6",
        className,
      )}
      aria-current="step"
    >
      <header className="flex flex-col gap-1">
        {eyebrow && (
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            {eyebrow}
          </span>
        )}
        <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
          {title}
        </h2>
        {description && (
          <p className="text-sm leading-relaxed text-muted-foreground md:text-[15px]">
            {description}
          </p>
        )}
      </header>
      {children && <div className="flex flex-col gap-4">{children}</div>}
      {primaryAction && (
        <div className="flex justify-end pt-2">{primaryAction}</div>
      )}
    </section>
  );
}
