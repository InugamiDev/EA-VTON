"use client";

// intent: SVG body silhouette + landmark dots that show users how to frame their photo
// status: done — shown behind the upload dropzone as a passive guide
// next: animate the landmark dots in sequence (head → shoulders → hips → knees) for first-time users
// confidence: high

import { cn } from "@/lib/utils";

type Props = {
  className?: string;
  /** When true, also annotate which landmarks the pipeline needs. */
  showLabels?: boolean;
};

/**
 * A neutral SVG silhouette of a standing figure that visualises the framing
 * the size + style pipeline expects:
 *   - head visible (top margin)
 *   - both shoulders visible (broad markers)
 *   - both hips visible (mid markers)
 *   - knees visible (improves leg-length ratio)
 *   - ankles ideal but not required
 *
 * Used as a passive guide behind the upload dropzone. Strictly decorative —
 * does not interact with the camera feed or pose detector.
 */
export function BodyGhostOverlay({ className, showLabels = false }: Props) {
  return (
    <svg
      viewBox="0 0 200 400"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-full w-full text-muted-foreground/30", className)}
      aria-hidden
    >
      {/* Head */}
      <circle
        cx="100"
        cy="40"
        r="22"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />

      {/* Neck */}
      <path
        d="M 90 62 L 90 75 L 110 75 L 110 62"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />

      {/* Torso */}
      <path
        d="M 60 80 L 140 80 L 135 200 L 65 200 Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      {/* Arms (relaxed, slight away from torso) */}
      <path
        d="M 60 80 L 38 180 L 42 200"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M 140 80 L 162 180 L 158 200"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Legs */}
      <path
        d="M 78 200 L 75 310 L 78 380"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M 122 200 L 125 310 L 122 380"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Landmark dots — match MediaPipe Pose indices used by the pipeline */}
      <Dot x={60} y={80} label={showLabels ? "L shoulder" : undefined} />
      <Dot x={140} y={80} label={showLabels ? "R shoulder" : undefined} />
      <Dot x={70} y={195} label={showLabels ? "L hip" : undefined} />
      <Dot x={130} y={195} label={showLabels ? "R hip" : undefined} />
      <Dot x={75} y={310} label={showLabels ? "L knee" : undefined} />
      <Dot x={125} y={310} label={showLabels ? "R knee" : undefined} />

      {/* Top-bottom framing guides */}
      <line
        x1="20"
        y1="18"
        x2="40"
        y2="18"
        stroke="currentColor"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
      <line
        x1="160"
        y1="18"
        x2="180"
        y2="18"
        stroke="currentColor"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
      <line
        x1="20"
        y1="382"
        x2="40"
        y2="382"
        stroke="currentColor"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
      <line
        x1="160"
        y1="382"
        x2="180"
        y2="382"
        stroke="currentColor"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
    </svg>
  );
}

function Dot({ x, y, label }: { x: number; y: number; label?: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r="4" fill="currentColor" opacity="0.7" />
      {label && (
        <text
          x={x < 100 ? x - 8 : x + 8}
          y={y + 3}
          fontSize="9"
          textAnchor={x < 100 ? "end" : "start"}
          fill="currentColor"
          opacity="0.6"
          fontFamily="var(--font-geist-mono)"
        >
          {label}
        </text>
      )}
    </g>
  );
}
