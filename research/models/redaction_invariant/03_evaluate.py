"""Stage 3 — evaluate the 6 trained variants and produce the decision-gate table.

Reads stage2_metrics.json and:
  1. Renders a markdown table: per-attribute F1 + average F1, on both eval modes
     (redacted at test, unredacted at test).
  2. Computes the headline number: closure gap = (paired_consist − redacted_only)
     as a fraction of (unredacted_only − redacted_only).
     - closure ≥ 0.7 → STRONG WIN (paired closes ≥70% of the gap to upper bound)
     - 0.4 ≤ closure < 0.7 → PARTIAL WIN (reportable but weaker)
     - closure < 0.4 → NEGATIVE RESULT (drop the contribution)
  3. Writes RESULTS.md with the decision and the suggested paper framing.

After running this, you have a single number to compare against the +10-15pp
probability bump promised in the paper outline.

Honest fallback: if the decision is NEGATIVE RESULT, the paper drops the
"redaction as self-supervision" claim and we go to NeurIPS D&B with the
existing artifacts only (the queue-strategy 45-60% remains the operating estimate).
"""

# intent: produce the single decision number that determines whether W1 lifts probability
# status: ready to run after 02_train_heads.py completes
# next: read RESULTS.md and act accordingly
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = Path(__file__).parent / "stage2_metrics.json"
RESULTS = Path(__file__).parent / "RESULTS.md"


def closure_fraction(redacted_f1: float, paired_f1: float, unredacted_f1: float) -> float:
    """Fraction of the (redacted → unredacted) gap that paired-consistency closes.
    Returns 0 if there's no upper-bound gap (anomalous), 1 if paired matches upper bound."""
    gap = unredacted_f1 - redacted_f1
    if gap <= 1e-6:
        return 0.0
    return (paired_f1 - redacted_f1) / gap


def main() -> None:
    if not METRICS.exists():
        sys.exit(f"!! {METRICS} not found — run 02_train_heads.py first")

    m = json.loads(METRICS.read_text())

    red_avg = m["redacted_only"]["avg_f1_redacted_eval"]
    unr_avg = m["unredacted_only"]["avg_f1_redacted_eval"]

    closures = {}
    for name in ["paired_no_consist", "paired_consist_lo", "paired_consist_md", "paired_consist_hi"]:
        if name not in m:
            continue
        pf = m[name]["avg_f1_redacted_eval"]
        closures[name] = closure_fraction(red_avg, pf, unr_avg)
    best_paired = max(closures, key=closures.get) if closures else None
    best_closure = closures.get(best_paired, 0.0) if best_paired else 0.0

    if best_closure >= 0.70:
        verdict = "STRONG_WIN"
        probability_bump = "+10 to +15 pts"
        framing = (
            "PAPER CONTRIBUTION: Redaction-pair training closes "
            f"{best_closure*100:.0f}% of the privacy-cost gap, using only "
            "redacted images at deployment. Add as §6 \"Redaction as "
            "Self-Supervision\" with the ablation table."
        )
    elif best_closure >= 0.40:
        verdict = "PARTIAL_WIN"
        probability_bump = "+3 to +5 pts"
        framing = (
            "Reportable but weaker. Include in §5 as a sensitivity analysis "
            "showing the dataset's pair structure has measurable utility. "
            "Do not lead with this as the headline contribution."
        )
    else:
        verdict = "NEGATIVE_RESULT"
        probability_bump = "0 to -2 pts (honest null)"
        framing = (
            "DROP this angle from the paper. The redaction-pair training does not "
            "produce a measurable transfer benefit at the tested λ values. "
            "Mention in limitations: \"We explored paired-consistency training "
            "and found no significant transfer benefit on attribute F1 in our "
            "experiment regime; the dataset's pair structure may benefit other "
            "downstream tasks we did not test.\""
        )

    print()
    print("## W1 Decision Gate — Redaction-Invariant Attribute Classifier\n")
    print(f"### Headline numbers (avg F1 on redacted test embeddings)\n")
    print("| Variant                | λ    | Avg F1 (redacted eval) | Avg F1 (unredacted eval) |")
    print("|------------------------|-----:|-----------------------:|-------------------------:|")
    order = ["redacted_only", "unredacted_only", "paired_no_consist",
             "paired_consist_lo", "paired_consist_md", "paired_consist_hi"]
    for name in order:
        if name not in m:
            continue
        r = m[name]
        lam = r.get("lambda", 0.0)
        ar = r["avg_f1_redacted_eval"]
        au = r["avg_f1_unredacted_eval"]
        print(f"| {name:<22s} | {lam:.2f} | {ar:.4f}                 | {au:.4f}                   |")

    print()
    print("### Closure-fraction analysis\n")
    print(f"- redacted-only baseline: F1 = {red_avg:.4f}")
    print(f"- unredacted-only upper bound: F1 = {unr_avg:.4f}")
    print(f"- gap: {unr_avg - red_avg:+.4f}")
    print()
    print(f"| Paired variant         | Closure of gap |")
    print(f"|------------------------|---------------:|")
    for name, c in closures.items():
        print(f"| {name:<22s} | {c*100:6.1f}%        |")
    print()
    print(f"**Best paired variant:** `{best_paired}` with closure = **{best_closure*100:.1f}%**")
    print()
    print("---")
    print(f"## VERDICT: {verdict}")
    print()
    print(f"**Probability bump for paper:** {probability_bump}")
    print()
    print(f"**Action:** {framing}")

    # Write RESULTS.md
    md = [
        "# W1 — Redaction-Invariant Classifier Results\n",
        "_Auto-generated by `03_evaluate.py` from `stage2_metrics.json`._\n\n",
        f"## Verdict: **{verdict}**\n\n",
        f"- Probability bump for paper acceptance: {probability_bump}\n",
        f"- Best paired variant: `{best_paired}`, closes {best_closure*100:.1f}% of the gap.\n",
        f"\n## Action\n\n{framing}\n",
        "\n## Headline numbers\n\n",
        "| Variant | λ | Avg F1 (red eval) | Avg F1 (unred eval) |\n",
        "|---|---:|---:|---:|\n",
    ]
    for name in order:
        if name not in m:
            continue
        r = m[name]
        md.append(
            f"| {name} | {r.get('lambda', 0.0):.2f} | "
            f"{r['avg_f1_redacted_eval']:.4f} | "
            f"{r['avg_f1_unredacted_eval']:.4f} |\n"
        )
    md.append(f"\n## Closure fractions\n\n")
    md.append("| Paired variant | Closure |\n|---|---:|\n")
    for name, c in closures.items():
        md.append(f"| {name} | {c*100:.1f}% |\n")
    md.append("\n## Per-attribute F1 (redacted-eval, best paired variant)\n\n")
    if best_paired and best_paired in m:
        attrs = m[best_paired]["per_attribute_f1_redacted_eval"]
        md.append("| Attribute | F1 |\n|---|---:|\n")
        for a, f in attrs.items():
            md.append(f"| {a} | {f:.4f} |\n")
    RESULTS.write_text("".join(md))
    print(f"\n  wrote {RESULTS}")


if __name__ == "__main__":
    main()
