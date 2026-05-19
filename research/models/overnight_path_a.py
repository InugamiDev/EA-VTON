"""Overnight driver for Path A — BodyM vision encoder on Mac (MPS).

Runs an auto-chain so the user can leave it overnight:

  Phase 0 (~5-15 min) — download_bodym_photos.py
                       Idempotent. Skips if 9,500+ masks already on disk.

  Phase 1 (~3-5 min)  — smoke train, --epochs 3 --gender female
                       Validates the pipeline. Aborts the chain on failure.

  Phase 2 (~1-2 h)    — full train, --epochs 25 --gender female
                       Saves best checkpoint to body_vision_encoder.pt.

  Phase 3 (decision)  — read results, compute size-chain within-1 accuracy.
                       If ≥ 0.70 → DONE.
                       If <  0.70 → run Phase 4.

  Phase 4 (~2-4 h)    — fallback variant, --epochs 50 --gender both.
                       More epochs + more data + different schedule.
                       Saves to body_vision_encoder_v2.pt.

Every phase logs to research/eval/overnight_path_a.log so you can `tail -f`.

Output state when done:
    research/models/variants/body_vision_encoder.pt        (phase 2)
    research/models/variants/body_vision_encoder_v2.pt     (phase 4, if reached)
    research/eval/body_vision_encoder_results.json         (phase 2 final eval)
    research/eval/body_vision_encoder_v2_results.json      (phase 4 final eval)
    research/eval/overnight_path_a.log                     (chronological log)
    research/eval/overnight_path_a_summary.json            (which phase met the bar)
"""

# intent: hands-off overnight training to hit ≥70% size accuracy
# status: ready — runs end-to-end without user intervention
# next: review overnight_path_a_summary.json in the morning
# confidence: medium — depends on BodyM's silhouette-to-measurement signal
#             being strong enough; pipeline is correct.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research/datasets/scripts"
MODELS = ROOT / "research/models"
EVAL = ROOT / "research/eval"
VARIANTS = MODELS / "variants"

LOG = EVAL / "overnight_path_a.log"
SUMMARY = EVAL / "overnight_path_a_summary.json"

PY = str(ROOT / ".venv/bin/python3")

TARGET_W1 = 0.70  # size-chain within-1 accuracy bar
PHASE_TIMEOUTS_S = {
    "download": 30 * 60,           # 30 min ceiling on download
    "smoke":    30 * 60,           # 30 min ceiling on 3-epoch smoke
    "full":     4 * 3600,          # 4 hours ceiling on 25-epoch full
    "variant":  6 * 3600,          # 6 hours ceiling on 50-epoch variant
}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], phase: str, timeout: int) -> int:
    log(f"── phase '{phase}' starting: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for chunk in proc.stdout.splitlines():
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(chunk + "\n")
        log(f"── phase '{phase}' exit={proc.returncode} ({time.time()-t0:.0f}s)")
        return proc.returncode
    except subprocess.TimeoutExpired:
        log(f"── phase '{phase}' TIMED OUT after {timeout}s")
        return 124


def n_masks() -> int:
    return sum(
        1 for _ in (ROOT / "research/datasets/raw/bodym").rglob("*.png")
    )


def read_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_within1(results_path: Path) -> float:
    r = read_results(results_path)
    if not r:
        return 0.0
    chain = r.get("final", {}).get("size_chain_testB", {})
    return float(chain.get("within1", 0.0))


def write_summary(record: dict) -> None:
    SUMMARY.write_text(json.dumps(record, indent=2, default=float))
    log(f"── summary written: {SUMMARY}")


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")  # truncate previous run
    log("══ Overnight Path A driver ══")
    log(f"  ROOT = {ROOT}")
    log(f"  PY   = {PY}")
    log(f"  target within-1 ≥ {TARGET_W1}")

    record: dict = {"started_at": time.time(), "phases": []}

    # ── Phase 0 — download ──
    masks_before = n_masks()
    log(f"  masks on disk before phase 0: {masks_before}")
    if masks_before < 9_500:
        rc = run([PY, str(SCRIPTS / "download_bodym_photos.py"), "--workers", "16"],
                 "download", PHASE_TIMEOUTS_S["download"])
        record["phases"].append({"name": "download", "rc": rc, "n_masks_after": n_masks()})
        if n_masks() < 5_000:
            log("!! download produced too few masks; aborting.")
            record["status"] = "aborted_download"
            write_summary(record)
            return
    else:
        log("  enough masks already cached; skipping phase 0")
        record["phases"].append({"name": "download", "rc": "skipped", "n_masks_after": masks_before})

    # ── Phase 1 — smoke train ──
    rc = run(
        [PY, str(MODELS / "train_body_vision_encoder.py"),
         "--epochs", "3", "--batch-size", "32", "--gender", "female"],
        "smoke", PHASE_TIMEOUTS_S["smoke"],
    )
    record["phases"].append({"name": "smoke", "rc": rc})
    if rc != 0:
        log("!! smoke train failed; aborting chain.")
        record["status"] = "smoke_failed"
        write_summary(record)
        return

    # ── Phase 2 — full train ──
    rc = run(
        [PY, str(MODELS / "train_body_vision_encoder.py"),
         "--epochs", "25", "--batch-size", "32", "--gender", "female"],
        "full", PHASE_TIMEOUTS_S["full"],
    )
    full_results = EVAL / "body_vision_encoder_results.json"
    w1_full = get_within1(full_results)
    record["phases"].append({
        "name": "full",
        "rc": rc,
        "size_chain_within1": w1_full,
    })
    log(f"  Phase 2 size-chain within-1 = {w1_full:.3f}")

    if w1_full >= TARGET_W1:
        log(f"  ✓ phase 2 met target ({w1_full:.3f} ≥ {TARGET_W1})")
        record["status"] = "target_met_phase2"
        record["best_within1"] = w1_full
        write_summary(record)
        return

    # ── Phase 4 (variant) — more epochs + more data ──
    log(f"  phase 2 below target ({w1_full:.3f}); running variant")
    # Move phase 2 outputs aside so phase 4 doesn't overwrite
    for p, q in [
        (full_results, EVAL / "body_vision_encoder_phase2_results.json"),
        (VARIANTS / "body_vision_encoder.pt", VARIANTS / "body_vision_encoder_phase2.pt"),
    ]:
        if p.exists():
            shutil.move(p, q)

    rc = run(
        [PY, str(MODELS / "train_body_vision_encoder.py"),
         "--epochs", "50", "--batch-size", "32", "--gender", "both",
         "--lr-backbone", "5e-5", "--lr-head", "5e-4"],
        "variant", PHASE_TIMEOUTS_S["variant"],
    )
    w1_var = get_within1(full_results)
    # Rename to v2 for clarity
    if (VARIANTS / "body_vision_encoder.pt").exists():
        shutil.move(VARIANTS / "body_vision_encoder.pt", VARIANTS / "body_vision_encoder_v2.pt")
    if full_results.exists():
        shutil.move(full_results, EVAL / "body_vision_encoder_v2_results.json")
    record["phases"].append({
        "name": "variant",
        "rc": rc,
        "size_chain_within1": w1_var,
    })
    log(f"  Phase 4 size-chain within-1 = {w1_var:.3f}")

    best = max(w1_full, w1_var)
    record["best_within1"] = best
    if best >= TARGET_W1:
        record["status"] = f"target_met_{'phase2' if w1_full >= w1_var else 'variant'}"
    else:
        record["status"] = "target_not_met"
    write_summary(record)
    log(f"══ DONE. best within-1 = {best:.3f}; status = {record['status']}")


if __name__ == "__main__":
    main()
