"use client";

// intent: 4-dot horizontal step indicator at the top of the guided journey
// status: done — matches DESIGN.md step-pill-* tokens
// next: optional click-to-jump back to a completed step
// confidence: high

import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export type JourneyStep = {
  id: string;
  label: string;
};

type Props = {
  steps: JourneyStep[];
  /** Index of the currently-active step (0-based). */
  current: number;
  /** Optional click handler — when provided, completed steps become clickable. */
  onJump?: (idx: number) => void;
  className?: string;
};

export function StepIndicator({ steps, current, onJump, className }: Props) {
  return (
    <ol
      className={cn(
        "flex w-full items-center justify-between gap-1 sm:gap-2",
        className,
      )}
      aria-label="Journey progress"
    >
      {steps.map((step, i) => {
        const state =
          i < current ? "complete" : i === current ? "active" : "inactive";
        const clickable = state === "complete" && onJump != null;
        return (
          <li
            key={step.id}
            className="flex min-w-0 flex-1 items-center gap-1 sm:gap-2"
          >
            <button
              type="button"
              onClick={clickable ? () => onJump?.(i) : undefined}
              disabled={!clickable}
              aria-current={state === "active" ? "step" : undefined}
              className={cn(
                "flex h-7 min-w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold tracking-wide transition-colors",
                state === "active" &&
                  "bg-foreground text-background ring-2 ring-foreground/15 ring-offset-2 ring-offset-background",
                state === "complete" && "bg-green-600 text-white",
                state === "inactive" &&
                  "border border-border bg-background text-muted-foreground",
                clickable && "cursor-pointer hover:opacity-90",
              )}
              title={step.label}
            >
              {state === "complete" ? (
                <Check className="h-3.5 w-3.5" strokeWidth={3} />
              ) : (
                i + 1
              )}
            </button>
            <span
              className={cn(
                "hidden truncate text-[11px] font-medium uppercase tracking-wider sm:inline",
                state === "active" && "text-foreground",
                state === "complete" && "text-foreground/70",
                state === "inactive" && "text-muted-foreground",
              )}
            >
              {step.label}
            </span>
            {i < steps.length - 1 && (
              <span
                className={cn(
                  "h-px flex-1 transition-colors",
                  i < current ? "bg-green-600" : "bg-border",
                )}
                aria-hidden
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
