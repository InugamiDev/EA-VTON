"""B1 Size Prediction evaluator.

Loads the ablation results produced by:
  - research/models/ablation_female_upper.py        (RTR rows)
  - research/models/train_modcloth_upper_size.py    (ModCloth rows)

Then prints a markdown table and writes results.json.

A third party reproducing this benchmark:
  1. Trains their model on the RTR fit='fit' female upper-body subset
     (same 80/20 stratified split, seed 42)
  2. Reports {exact, within1, mae} on the test split
  3. Optionally reports the same on the VN-range subset (h<160, w<55)
  4. PRs a row into leaderboard.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ROOT / "research/models/variants"
ABLATION = VARIANTS / "ablation_female_upper.json"
MODCLOTH = VARIANTS / "gbm_modcloth_upper.pkl"  # via train_modcloth script
B1_OUT = Path(__file__).parent / "results.json"


def main() -> None:
    if not ABLATION.exists():
        sys.exit(
            f"!! {ABLATION} not found — first run research/models/ablation_female_upper.py"
        )

    ablation = json.loads(ABLATION.read_text())

    results: dict = {"primary_rtr_female_upper": ablation["results"]}

    # ModCloth rows (optional — pulled from the saved bundle's results dict)
    if MODCLOTH.exists():
        import pickle

        bundle = pickle.loads(MODCLOTH.read_bytes())
        results["modcloth_external_validity"] = bundle["results"]

    B1_OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"  wrote {B1_OUT}")
    print()

    # Markdown table
    print("## B1 — Size Prediction (paper-aligned)")
    print()
    print("### Primary: RTR female upper-body, 6-row ablation")
    print()
    print("| Variant              | Tempering | Exact | Within-1 |  MAE  | VN-Exact | VN-W1 |  k̂  |")
    print("|----------------------|-----------|------:|---------:|------:|---------:|------:|----:|")
    for r in ablation["results"]:
        f, v = r["full_test"], r["vn_subset"]
        ve = f"{v['exact']:.3f}" if v.get("exact") is not None else "—"
        vw = f"{v['within1']:.3f}" if v.get("within1") is not None else "—"
        khat = "—"
        if "PSIS" in r["variant"]:
            khat = f"{ablation.get('psis_k_hat', 0):.2f}"
        print(f"| {r['variant']:<20s} | {r['alpha']:^9s} | {f['exact']:.3f} | {f['within1']:.3f} | {f['mae']:.3f} | {ve:>8s} | {vw:>5s} | {khat:>3s} |")

    if "modcloth_external_validity" in results:
        mc = results["modcloth_external_validity"]
        rows = mc.get("rows_by_feature_set", {})
        print()
        print("### Secondary: ModCloth external-validity")
        print()
        print("| Feature set                                       | Exact | Within-1 |  MAE  |")
        print("|---------------------------------------------------|------:|---------:|------:|")
        for fname, row in rows.items():
            cols = ",".join(row["feature_cols"])
            print(f"| {fname} ({cols})  | {row['exact']:.3f} | {row['within1']:.3f} | {row['mae']:.3f} |")


if __name__ == "__main__":
    main()
