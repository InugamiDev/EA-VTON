"""B3 — Cold-Start Calibration evaluator.

Delegates to research/eval/calibration_eval.py and writes a tidied JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "research/eval/calibration_eval.py"
SRC = ROOT / "research/eval/calibration_results.json"
OUT = Path(__file__).parent / "results.json"


def main() -> None:
    if not RUNNER.exists():
        sys.exit(f"!! {RUNNER} missing — install the research/ tree first")
    print(f"  delegating to {RUNNER}…")
    subprocess.check_call([sys.executable, str(RUNNER)])
    if not SRC.exists():
        sys.exit(f"!! expected {SRC} after run — runner did not produce JSON")
    data = json.loads(SRC.read_text())
    OUT.write_text(json.dumps(data, indent=2, default=float))
    print(f"\n  wrote {OUT}")

    print()
    print("## B3 — Cold-Start Calibration")
    print()
    print("| Strategy        | align@10 δ=0 | δ=0.3 | δ=0.7 | δ=1.0 |  Δ  |")
    print("|-----------------|-------------:|------:|------:|------:|----:|")
    for name, s in data["by_strategy"].items():
        d0, d10 = s["alignment"]["0.0"]["10"], s["alignment"]["1.0"]["10"]
        d3, d7 = s["alignment"]["0.3"]["10"], s["alignment"]["0.7"]["10"]
        print(f"| {name:<15s} | {d0:.3f}        | {d3:.3f} | {d7:.3f} | {d10:.3f} | {d10-d0:+.3f} |")


if __name__ == "__main__":
    main()
