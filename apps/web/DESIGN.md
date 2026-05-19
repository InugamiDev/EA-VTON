---
version: "1.0"
name: "FitView — Live Size & Style"
description: "Monochrome, calm, mobile-first design language for the size + style recommender. Privacy-first; visual hierarchy prioritises a single guided journey: camera → size → style → personalize."

colors:
  background:          "#fafafa"
  foreground:          "#0a0a0a"
  muted:               "#f5f5f5"
  muted-foreground:    "#737373"
  border:              "#e5e5e5"
  ring:                "#0a0a0a"
  primary:             "#0a0a0a"
  primary-foreground:  "#fafafa"
  accent:              "#f5f5f5"
  accent-foreground:   "#0a0a0a"
  card:                "#ffffff"
  card-foreground:     "#0a0a0a"
  destructive:         "#ef4444"
  on-destructive:      "#ffffff"
  success:             "#16a34a"
  on-success:          "#ffffff"
  step-active:         "#0a0a0a"
  step-inactive:       "#e5e5e5"
  step-complete:       "#16a34a"
  step-foreground:     "#fafafa"

typography:
  display-xl:
    fontFamily: "{typography.font-sans}"
    fontSize: "2.5rem"          # 40px
    fontWeight: 800
    lineHeight: "1.1"
    letterSpacing: "-0.02em"
  display-lg:
    fontFamily: "{typography.font-sans}"
    fontSize: "2rem"            # 32px
    fontWeight: 700
    lineHeight: "1.15"
    letterSpacing: "-0.015em"
  heading-md:
    fontFamily: "{typography.font-sans}"
    fontSize: "1.25rem"         # 20px
    fontWeight: 600
    lineHeight: "1.3"
  heading-sm:
    fontFamily: "{typography.font-sans}"
    fontSize: "1rem"            # 16px
    fontWeight: 600
    lineHeight: "1.4"
  body-md:
    fontFamily: "{typography.font-sans}"
    fontSize: "0.9375rem"       # 15px
    fontWeight: 400
    lineHeight: "1.55"
  body-sm:
    fontFamily: "{typography.font-sans}"
    fontSize: "0.8125rem"       # 13px
    fontWeight: 400
    lineHeight: "1.5"
  label:
    fontFamily: "{typography.font-sans}"
    fontSize: "0.6875rem"       # 11px
    fontWeight: 500
    lineHeight: "1.4"
    letterSpacing: "0.04em"
  mono-sm:
    fontFamily: "{typography.font-mono}"
    fontSize: "0.75rem"         # 12px
    fontWeight: 500
  font-sans: "var(--font-geist-sans), system-ui, sans-serif"
  font-mono: "var(--font-geist-mono), ui-monospace, monospace"

rounded:
  none:  "0"
  sm:    "0.5rem"      # 8px
  md:    "1rem"        # 16px
  lg:    "1.5rem"      # 24px
  xl:    "2rem"        # 32px
  full:  "9999px"

spacing:
  0:   "0"
  px:  "1px"
  1:   "0.25rem"    # 4px
  2:   "0.5rem"     # 8px
  3:   "0.75rem"    # 12px
  4:   "1rem"       # 16px
  5:   "1.25rem"    # 20px
  6:   "1.5rem"     # 24px
  8:   "2rem"       # 32px
  10:  "2.5rem"     # 40px
  12:  "3rem"       # 48px
  16:  "4rem"       # 64px

components:
  surface-card:
    backgroundColor: "{colors.card}"
    textColor:       "{colors.card-foreground}"
    rounded:         "{rounded.lg}"
    padding:         "{spacing.6}"
  surface-card-tight:
    backgroundColor: "{colors.card}"
    textColor:       "{colors.card-foreground}"
    rounded:         "{rounded.md}"
    padding:         "{spacing.4}"
  step-pill-active:
    backgroundColor: "{colors.step-active}"
    textColor:       "{colors.step-foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.label}"
    height:          "1.75rem"
  step-pill-inactive:
    backgroundColor: "{colors.step-inactive}"
    textColor:       "{colors.muted-foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.label}"
    height:          "1.75rem"
  step-pill-complete:
    backgroundColor: "{colors.step-complete}"
    textColor:       "{colors.step-foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.label}"
    height:          "1.75rem"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor:       "{colors.primary-foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.heading-sm}"
    padding:         "{spacing.3}"
    height:          "2.75rem"
  button-secondary:
    backgroundColor: "{colors.card}"
    textColor:       "{colors.foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.body-sm}"
    padding:         "{spacing.3}"
    height:          "2.5rem"
  badge:
    backgroundColor: "{colors.muted}"
    textColor:       "{colors.muted-foreground}"
    rounded:         "{rounded.full}"
    typography:      "{typography.label}"
    padding:         "{spacing.2}"
---

## Overview

**Product fit.** FitView is a camera-first size + style recommendation experience for Vietnamese women's upper-body fashion. The live-size page is the system's only "active" interaction surface — every other page (catalog, closet, try-on) is reactive. The page must feel calm, intentional, and trustworthy because users are sharing camera footage of their body.

**Personality.** Monochrome minimalism. Function over decoration. The visual language draws from Linear, Vercel, and Notion — small surface count, generous whitespace, type carries the meaning, color is reserved for state (active, success, danger).

**Density.** Low-medium. Each screen step shows ONE thing well rather than five at once. The previous layout violates this by competing five sections (camera + size + style + calibration + diagnostics) for attention. The redesign collapses the right column into a vertical journey on mobile and a guided sidebar on desktop.

**Audience.** Vietnamese women 18-45, smartphone-primary, comfortable with TikTok-style flows. Implication: large touch targets, swipe-friendly gestures, no dense information dashboards.

## Colors

| Token | Hex | Role |
|---|---|---|
| `background` | `#fafafa` | Page background; rare in cards. |
| `foreground` | `#0a0a0a` | Body text; primary CTA fill. |
| `card` | `#ffffff` | Default surface. |
| `muted` | `#f5f5f5` | Secondary surface; badge fill. |
| `muted-foreground` | `#737373` | Helper text. |
| `border` | `#e5e5e5` | Hairline dividers; never used as background. |
| `step-active` | `#0a0a0a` | Current step indicator. |
| `step-complete` | `#16a34a` | Completed step indicator. |
| `step-inactive` | `#e5e5e5` | Upcoming step (looks like a hairline). |
| `destructive` | `#ef4444` | Errors only; never used as decoration. |

WCAG AA: foreground on card = 18.8:1; muted-foreground on card = 4.8:1; primary-foreground on primary = 18.8:1. All pass for body text.

## Typography

The Geist family from Vercel — `Geist Sans` for everything except monospace metrics (camera latency, model confidence numbers) which use `Geist Mono`.

| Token | Use |
|---|---|
| `display-xl` | Page header on desktop. Mobile drops to `display-lg`. |
| `display-lg` | Step heading ("Step 1 — Stand in frame"). |
| `heading-md` | Size result ("Recommended size: M"). |
| `heading-sm` | Card headings; button text. |
| `body-md` | Default reading copy. |
| `body-sm` | Secondary copy (e.g., model confidence breakdown). |
| `label` | All-caps eyebrows above section heads. |
| `mono-sm` | Live latency, model name, confidence percentage. |

Never mix three weights in the same section. Default flow: `display` for the step name, `body-md` for the helper sentence, optional `label` for the section eyebrow.

## Layout

### Breakpoints

| Name | Min width | Notes |
|---|---|---|
| `xs` | 0 | Default mobile. Single-column. |
| `sm` | 640px | Tighter padding, still single-column. |
| `md` | 768px | Two-column appears on the size-result step only. |
| `lg` | 1024px | Guided sidebar on the right; camera dominates left 60%. |
| `xl` | 1280px | Max content width 1280px; centred. |

### Grid

- Mobile: 100vw, padding `spacing.4` (16px) on the body, full-bleed inside `surface-card`s.
- Desktop: `[camera 60% | journey 40%]` two-track layout. Camera is sticky-top; journey scrolls inside its track.
- Within each step card, content uses an internal 8px rhythm (`spacing.2`, `spacing.4`, `spacing.6`).

### Vertical journey rhythm

```
┌──────────────────────────────┐
│ ① ━━━ ② ━━━ ③ ━━━ ④          │   step indicator pills
└──────────────────────────────┘
┌──────────────────────────────┐
│ Step heading (display-lg)    │
│ Helper sentence (body-md)    │
│                              │
│ [content: camera / result /  │
│  recs / calibration grid]    │
│                              │
│ [primary CTA pill]           │
└──────────────────────────────┘
```

One step is *active* at a time. Completed steps collapse into a 1-line summary tile that can be tapped to re-expand.

## Elevation & Depth

- Cards: `shadow-sm` (1px 2px rgba(0,0,0,0.04)).
- Sticky chrome (navbar, step indicator on mobile): `backdrop-blur-md` over `bg-background/90`.
- Modals/sheets (calibration grid expanded): elevated card with `shadow-lg` and a `bg-foreground/30` scrim.
- **No drop shadows on inline elements.** Borders carry depth.

## Shapes

- Cards: `rounded.lg` (24px) on desktop, `rounded.md` (16px) on mobile narrow widths.
- Pills (buttons, step indicators, badges): `rounded.full`.
- Camera feed: `rounded.lg` with a 1px `border`. Never full-bleed; always inset by 16px.
- Item thumbnails (catalog cards, calibration grid): `rounded.md`.

## Components

### Step indicator (`step-pill-*`)

4 dots horizontal at top of journey track. Each dot is 28px high, `rounded.full`.

```
   ●━━━●━━━○━━━○
   1   2   3   4
   ✓   →
```

- ✓ icon for completed; number for active and upcoming.
- Active dot: filled with `step-active`. Inactive: hairline `step-inactive`. Complete: `step-complete` filled with check.
- Connecting lines: 2px, color matches the higher state of the two adjacent dots.

### Primary CTA (`button-primary`)

Full-width on mobile (within step card padding); auto width on desktop. Height 44px (touch target). Disabled state: `opacity 0.4`, no pointer events.

### Camera card

Aspect ratio 3:4 on mobile, 4:5 on desktop. Pose overlay opacity 0.7. Latency badge bottom-right uses `mono-sm`. Snapshot/Live/Upload mode toggle is a 3-segment pill at the top of the card.

### Size result card

Single hero number ("M") at `display-xl` weight, with a `body-sm` confidence percentage in `mono-sm` directly below. Beside the number: a stacked `Within-1: 71%` chip.

### Style recommendations grid

3 cols on mobile, 4-5 on desktop. Each item: rounded square thumb + 2-line title + 1-line rationale. Tap → expand to a sheet with the full explanation DAG (fit/flatter/match).

### Calibration grid

Stratified 18-item grid (3×6 mobile, 6×3 desktop). Tap to select; selected items get a 2px `step-active` border + a checkmark badge. "Calibrate" CTA appears as a sticky bottom action bar when ≥3 are selected. After calibration, results appear inline below the grid in a `surface-card-tight`.

## Do's and Don'ts

### Do

- Treat the page as a guided journey, not a dashboard.
- Use `display-lg` as a step heading exactly once per step.
- Give the camera card the most visual weight on its step; collapse all others.
- Animate step transitions (150-300ms), respect `prefers-reduced-motion`.
- Use `mono-sm` for any latency / confidence / model-name strings.

### Don't

- Don't show diagnostics by default — gate behind a "Debug" toggle.
- Don't use shadow on top of border for the same surface — pick one.
- Don't render the calibration panel until a size has been predicted (it depends on `predicted_size`).
- Don't use decorative colour outside the small palette (no purples, blues, gradients).
- Don't auto-scroll between steps; user controls progression via CTA.
- Don't crowd the camera step with size sliders — those go on the size result step.
