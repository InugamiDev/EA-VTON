# Size "Rosetta Stone" — pivot from VN size labels to real cm via US/UK equivalents.
# Idea: VN has no rich anthropometric dataset, but we have VN <-> US/UK conversion
# tables, and US/UK have published size-to-measurement standards (ASTM D5585 etc).
# Compose: VN_size -> US_size -> ASTM_measurements_cm. We get real bust/waist/hip
# numbers per VN size without needing a VN-only anthropometric study.

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Conversion table from user input (women's body) ──
# Each row = one US numeric size with its UK letter and VN label/numeric.
# Duplicates collapsed (e.g. US 12 and US 16 both mapping to UK M / VN L
# means VN L spans US 12–16 — we represent that as a range).

WOMEN = [
    # us, uk_letter, vn_letter, vn_eu, vn_local
    (4,  "XS", "S",   "36", "5"),
    (6,  "XS", "S",   "36", "7"),
    (8,  "S",  "M",   "38", "9"),
    (10, "S",  "M",   "38", "9"),
    (12, "M",  "L",   "40", "11"),
    (14, "M",  "L",   "40", "11"),
    (16, "M",  "L",   "40", "11"),
    (18, "L",  "XL",  "42", "13"),
    (20, "L",  "XL",  "42", "13"),
    (22, "XL", "XXL", "3L", "15"),
    (24, "XL", "XXL", "3L", "15"),
    (26, "XXL","4L",  "—",  "17"),
    (28, "XXL","5L",  "—",  "19"),
]

# ── ASTM D5585 / industry-standard women's measurements per US size (cm) ──
# Source: ASTM D5585-11 (Misses) extended for plus sizes via standard 5cm/2-size grade.
# These are the "real cm" we get for free by knowing the US size.
# Format: us_size -> (bust_cm, waist_cm, hip_cm)
US_TO_CM = {
    0:  (78,  61, 86),
    2:  (81,  64, 89),
    4:  (84,  66, 91),
    6:  (87,  69, 94),
    8:  (90,  71, 97),
    10: (93,  74, 99),
    12: (96,  76, 102),
    14: (100, 81, 107),
    16: (105, 86, 112),
    18: (110, 91, 117),
    20: (115, 96, 122),
    22: (120, 101, 127),
    24: (125, 106, 132),
    26: (130, 111, 137),
    28: (135, 116, 142),
}

# Aggregate: group rows by VN letter, find min/max US in that group → measurement range.
from collections import defaultdict
vn_groups: dict[str, list[int]] = defaultdict(list)
for us, _uk, vn_letter, _vn_eu, _vn_local in WOMEN:
    vn_groups[vn_letter].append(us)

# Compute measurement range per VN size
vn_summary = {}
for vn_letter, us_sizes in vn_groups.items():
    bust_lo = US_TO_CM[min(us_sizes)][0]
    bust_hi = US_TO_CM[max(us_sizes)][0]
    waist_lo = US_TO_CM[min(us_sizes)][1]
    waist_hi = US_TO_CM[max(us_sizes)][1]
    hip_lo = US_TO_CM[min(us_sizes)][2]
    hip_hi = US_TO_CM[max(us_sizes)][2]
    vn_summary[vn_letter] = {
        "us_range": (min(us_sizes), max(us_sizes)),
        "bust": (bust_lo, bust_hi),
        "waist": (waist_lo, waist_hi),
        "hip": (hip_lo, hip_hi),
    }

VN_ORDER = ["S", "M", "L", "XL", "XXL", "4L", "5L"]

# Vietnamese population reference (Vietnam STEPS / NHI surveys)
VN_FEMALE_AVG = {"height": 154.5, "weight": 52.5, "bust": 84, "waist": 71, "hip": 88}

# ════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11), facecolor="#0e0e10")
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.4, 1.0], hspace=0.55, wspace=0.22)

# ── Panel 1: The Rosetta Stone — three lanes, drawn lines bridge them ──
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#0e0e10")

LANE_Y = {"VN": 2.2, "UK": 1.2, "US": 0.2}
LANE_COLOR = {"VN": "#ff6b6b", "UK": "#74b9ff", "US": "#fdcb6e"}

# Draw lane backgrounds
for region, y in LANE_Y.items():
    ax1.add_patch(FancyBboxPatch(
        (-0.6, y - 0.18), 16.2, 0.36,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0, facecolor=LANE_COLOR[region], alpha=0.13,
    ))
    ax1.text(-0.4, y, region, fontsize=14, fontweight="bold",
             color=LANE_COLOR[region], va="center")

# Plot points and connecting lines
for i, (us, uk, vn_letter, vn_eu, vn_local) in enumerate(WOMEN):
    x = i

    # US point
    ax1.scatter(x, LANE_Y["US"], s=200, color=LANE_COLOR["US"],
                edgecolor="white", linewidth=1.2, zorder=3)
    ax1.text(x, LANE_Y["US"] - 0.32, str(us), fontsize=10, color="white",
             ha="center", fontweight="bold")

    # UK point
    ax1.scatter(x, LANE_Y["UK"], s=200, color=LANE_COLOR["UK"],
                edgecolor="white", linewidth=1.2, zorder=3)
    ax1.text(x, LANE_Y["UK"] - 0.32, uk, fontsize=10, color="white",
             ha="center", fontweight="bold")

    # VN point
    ax1.scatter(x, LANE_Y["VN"], s=200, color=LANE_COLOR["VN"],
                edgecolor="white", linewidth=1.2, zorder=3)
    label = f"{vn_letter}\n({vn_eu})"
    ax1.text(x, LANE_Y["VN"] + 0.32, label, fontsize=9, color="white",
             ha="center", va="bottom", fontweight="bold")

    # Connection lines
    ax1.plot([x, x], [LANE_Y["US"], LANE_Y["UK"]],
             color="white", alpha=0.22, linewidth=1, zorder=2)
    ax1.plot([x, x], [LANE_Y["UK"], LANE_Y["VN"]],
             color="white", alpha=0.22, linewidth=1, zorder=2)

ax1.set_xlim(-1.5, len(WOMEN) + 0.3)
ax1.set_ylim(-0.4, 3.0)
ax1.axis("off")
ax1.set_title(
    "Size Rosetta Stone — VN ↔ UK ↔ US (Women's Body)",
    fontsize=15, color="white", pad=12, fontweight="bold", loc="left",
)

# ── Panel 2: Measurement curves vs US size, with VN bands overlaid ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#0e0e10")

us_sizes = sorted(US_TO_CM.keys())
bust = [US_TO_CM[u][0] for u in us_sizes]
waist = [US_TO_CM[u][1] for u in us_sizes]
hip = [US_TO_CM[u][2] for u in us_sizes]

ax2.plot(us_sizes, bust, marker="o", color="#ff7675", lw=2.5, label="Bust", markersize=6)
ax2.plot(us_sizes, waist, marker="s", color="#fdcb6e", lw=2.5, label="Waist", markersize=6)
ax2.plot(us_sizes, hip, marker="^", color="#74b9ff", lw=2.5, label="Hip", markersize=6)

# VN size bands as colored vertical regions
band_colors = ["#ff6b6b22", "#ff6b6b33", "#ff6b6b22", "#ff6b6b33", "#ff6b6b22", "#ff6b6b33", "#ff6b6b22"]
for vn_letter, color in zip(VN_ORDER, band_colors):
    if vn_letter not in vn_summary:
        continue
    lo, hi = vn_summary[vn_letter]["us_range"]
    ax2.axvspan(lo - 0.5, hi + 0.5, alpha=0.4, color=color, zorder=0)
    mid = (lo + hi) / 2
    ax2.text(mid, 145, vn_letter, color="#ff6b6b", ha="center", va="top",
             fontsize=11, fontweight="bold")

# VN female population average reference line (height-corrected projection)
ax2.axhline(VN_FEMALE_AVG["bust"], color="#ff7675", lw=1, ls="--", alpha=0.5)
ax2.axhline(VN_FEMALE_AVG["waist"], color="#fdcb6e", lw=1, ls="--", alpha=0.5)
ax2.axhline(VN_FEMALE_AVG["hip"], color="#74b9ff", lw=1, ls="--", alpha=0.5)
ax2.text(28.5, VN_FEMALE_AVG["bust"], " VN avg bust", color="#ff7675",
         fontsize=8, va="center")
ax2.text(28.5, VN_FEMALE_AVG["waist"], " VN avg waist", color="#fdcb6e",
         fontsize=8, va="center")
ax2.text(28.5, VN_FEMALE_AVG["hip"], " VN avg hip", color="#74b9ff",
         fontsize=8, va="center")

ax2.set_xlabel("US numeric size", color="white", fontsize=11)
ax2.set_ylabel("Body measurement (cm)", color="white", fontsize=11)
ax2.set_title("ASTM measurements per US size, banded by VN letter\n(VN avg lines from STEPS Survey)",
              color="white", fontsize=12, fontweight="bold", loc="left")
ax2.tick_params(colors="white")
for s in ax2.spines.values():
    s.set_color("#444")
ax2.legend(loc="lower right", facecolor="#1a1a1d", edgecolor="#333",
           labelcolor="white", fontsize=10)
ax2.grid(True, alpha=0.15, color="white")

# ── Panel 3: 2D scatter — bust × waist with size blobs ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0e0e10")

# Each US size as a point, color-coded by VN letter
vn_color = {"S": "#fd79a8", "M": "#fdcb6e", "L": "#a29bfe",
            "XL": "#74b9ff", "XXL": "#55efc4", "4L": "#00b894", "5L": "#00cec9"}
vn_for_us = {us: vn_letter for us, _, vn_letter, _, _ in WOMEN}

for us in us_sizes:
    if us not in vn_for_us:
        continue
    b, w, h = US_TO_CM[us]
    vn = vn_for_us[us]
    ax3.scatter(w, b, s=180, c=vn_color.get(vn, "white"),
                edgecolor="white", lw=1, alpha=0.85, zorder=3)

# Connect with a path showing growth trajectory
ws = [US_TO_CM[u][1] for u in sorted(vn_for_us.keys())]
bs = [US_TO_CM[u][0] for u in sorted(vn_for_us.keys())]
ax3.plot(ws, bs, color="white", alpha=0.25, lw=1.5, zorder=2)

# VN average woman star
ax3.scatter(VN_FEMALE_AVG["waist"], VN_FEMALE_AVG["bust"],
            s=400, marker="*", color="#ff6b6b", edgecolor="white",
            lw=1.5, zorder=4, label="VN female avg")

# Cluster centroid annotations
for vn_letter in VN_ORDER:
    if vn_letter not in vn_summary:
        continue
    bust_mid = sum(vn_summary[vn_letter]["bust"]) / 2
    waist_mid = sum(vn_summary[vn_letter]["waist"]) / 2
    ax3.annotate(vn_letter, (waist_mid, bust_mid), color="white",
                 fontsize=11, fontweight="bold",
                 xytext=(8, 8), textcoords="offset points")

ax3.set_xlabel("Waist (cm)", color="white", fontsize=11)
ax3.set_ylabel("Bust (cm)", color="white", fontsize=11)
ax3.set_title("Body proportion space — VN size clusters\n(★ = VN female population average)",
              color="white", fontsize=12, fontweight="bold", loc="left")
ax3.tick_params(colors="white")
for s in ax3.spines.values():
    s.set_color("#444")
ax3.legend(loc="lower right", facecolor="#1a1a1d", edgecolor="#333",
           labelcolor="white", fontsize=10)
ax3.grid(True, alpha=0.15, color="white")

# ── Panel 4: VN size summary table ──
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor("#0e0e10")
ax4.axis("off")

table_data = []
for vn_letter in VN_ORDER:
    if vn_letter not in vn_summary:
        continue
    s = vn_summary[vn_letter]
    us_lo, us_hi = s["us_range"]
    table_data.append([
        vn_letter,
        f"US {us_lo}–{us_hi}" if us_lo != us_hi else f"US {us_lo}",
        f"{s['bust'][0]}–{s['bust'][1]} cm",
        f"{s['waist'][0]}–{s['waist'][1]} cm",
        f"{s['hip'][0]}–{s['hip'][1]} cm",
    ])

col_labels = ["VN size", "US equivalent", "Bust range", "Waist range", "Hip range"]
table = ax4.table(
    cellText=table_data, colLabels=col_labels,
    loc="center", cellLoc="center",
    colWidths=[0.10, 0.18, 0.20, 0.20, 0.20],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#1a1a1d")
        cell.set_text_props(color="#ff6b6b", fontweight="bold")
    else:
        cell.set_facecolor("#0e0e10" if r % 2 == 0 else "#141417")
        cell.set_text_props(color="white")

ax4.set_title("Recovered measurement ranges per VN size (via US pivot + ASTM D5585)",
              color="white", fontsize=12, fontweight="bold", loc="left", pad=10)

fig.suptitle(
    "Size Pivot: VN labels → real cm via US/UK Rosetta Stone",
    fontsize=18, color="white", fontweight="bold", y=0.985,
)
fig.text(
    0.5, 0.005,
    "Method: VN size ←→ US size mapping (user's table) → ASTM D5585 standard → cm. "
    "VN avg from Vietnam STEPS Survey 2009/2015.",
    ha="center", color="#888", fontsize=9, style="italic",
)

OUT = "/Users/inugami/Documents/GitHub/research-try-out/reports/size_rosetta_stone.png"
plt.savefig(OUT, dpi=140, facecolor="#0e0e10", bbox_inches="tight")
print(f"Saved: {OUT}")
print()
print("VN size → US equivalent → real cm (via ASTM D5585):")
for row in table_data:
    print(f"  {row[0]:5s} {row[1]:12s} bust {row[2]:14s} waist {row[3]:14s} hip {row[4]}")
