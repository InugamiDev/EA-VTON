"""FitView Virtual Try-On Pipeline.

Multi-stage pipeline:
1. Preprocessing  — person photo normalization, garment background removal
2. Inference      — model backend (FASHN VTON v1.5, Replicate, or local composite)
3. Postprocessing — face restoration, skin correction, color correction, confidence scoring

Usage:
    result = await run_pipeline(user_photo_path, garment_image_url, garment_id, config)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/fitview_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _has_torch() -> bool:
    """Check if torch is available for HuggingFace backend."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class PipelineConfig:
    """Configuration for the try-on pipeline."""
    # Inference backend: "replicate", "huggingface", "local_composite", "auto"
    backend: str = "auto"
    # Replicate model version
    replicate_model: str = "cuuupid/idm-vton:c871bb9b046607b680571c6523234de5e884e63f3f725198e457e180f08e5bec"
    # Inference params (30 for Replicate, 20 for local HF on CPU/MPS)
    denoise_steps: int = 20
    seed: int = 42
    # Preprocessing
    auto_crop_person: bool = True
    remove_garment_bg: bool = True
    normalize_person_size: tuple[int, int] = (768, 1024)  # W x H
    # Postprocessing (scoring only — enhancement is garment-masked in Stage 5)
    check_face_preservation: bool = True
    color_correct: bool = False   # disabled: garment composite handles this inside mask
    sharpen_result: bool = False  # disabled: garment composite handles this inside mask
    # HuggingFace backend (FASHN VTON v1.5)
    hf_model_id: str = "fashn-ai/fashn-vton-1.5"
    hf_device: str = "auto"  # auto / cuda / mps / cpu
    enable_cpu_offload: bool = True
    # New postprocessing upgrades
    face_restoration: bool = True
    skin_tone_correction: bool = True
    # Garment-only enhancement (realism upgrades — only modifies garment pixels)
    upscale_to_original: bool = True       # upscale 576x768 → original resolution
    max_output_resolution: int = 4096      # max output dimension (width or height), 0=no limit
    detail_sharpen_strength: float = 1.5   # garment-only sharpening (1.0=off, 2.0=strong)
    lighting_transfer_strength: float = 0.6  # match scene lighting on garment (0=off, 1=full)
    # Garment type for agnostic mask generation
    garment_type: str = "upper_body"  # upper_body / lower_body / full_body


@dataclass
class PipelineStage:
    """A single pipeline stage result."""
    name: str
    status: str = "pending"  # pending, running, completed, skipped, failed
    duration_ms: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    result_path: str
    confidence_score: float
    confidence_label: str
    processing_time_ms: int
    method: str
    stages: list[PipelineStage] = field(default_factory=list)

    @property
    def stage_summary(self) -> dict:
        return {s.name: {"status": s.status, "ms": s.duration_ms} for s in self.stages}


async def run_pipeline(
    user_photo_path: str,
    garment_image_url: str,
    garment_id: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Execute the full try-on pipeline."""
    import os

    if config is None:
        config = PipelineConfig()

    pipeline_start = time.time()
    stages: list[PipelineStage] = []

    # Determine backend
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if config.backend == "auto":
        if replicate_token:
            backend = "replicate"
        elif _has_torch():
            backend = "huggingface"
        else:
            backend = "local_composite"
    else:
        backend = config.backend

    if backend == "replicate" and not replicate_token:
        raise RuntimeError("Replicate backend requested but REPLICATE_API_TOKEN is not set")

    logger.info("Pipeline starting: backend=%s, garment=%s", backend, garment_id)

    # ── Stage 1: Preprocess person photo ─────────────────────────────
    from services.pipeline.preprocessor import preprocess_person, preprocess_garment

    stage = PipelineStage(name="preprocess_person", status="running")
    stage_start = time.time()
    try:
        person_result = await asyncio.to_thread(
            preprocess_person, user_photo_path, config
        )
        stage.status = "completed"
        stage.details = person_result.get("details", {})
    except Exception as e:
        logger.warning("Person preprocessing failed, using original: %s", e)
        person_result = {"path": user_photo_path, "details": {"error": str(e)}}
        stage.status = "failed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stages.append(stage)

    # ── Stage 2: Preprocess garment ──────────────────────────────────
    stage = PipelineStage(name="preprocess_garment", status="running")
    stage_start = time.time()
    try:
        garment_result = await asyncio.to_thread(
            preprocess_garment, garment_image_url, config
        )
        stage.status = "completed"
        stage.details = garment_result.get("details", {})
    except Exception as e:
        logger.warning("Garment preprocessing failed, using original URL: %s", e)
        garment_result = {"path": None, "url": garment_image_url, "mask_path": None}
        stage.status = "failed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stages.append(stage)

    # ── Stage 3: Inference ───────────────────────────────────────────
    stage = PipelineStage(name="inference", status="running")
    stage_start = time.time()
    try:
        if backend == "replicate" and replicate_token:
            from services.pipeline.backends.replicate_vton import run_replicate_inference
            inference_result = await run_replicate_inference(
                person_path=person_result["path"],
                garment_path=garment_result.get("path"),
                garment_url=garment_result.get("url", garment_image_url),
                garment_mask_path=garment_result.get("mask_path"),
                config=config,
                token=replicate_token,
            )
        elif backend == "huggingface":
            from services.pipeline.backends.huggingface_vton import run_huggingface_inference
            # FASHN VTON v1.5 is maskless — no preprocessing masks needed
            inference_result = await run_huggingface_inference(
                person_path=person_result["path"],
                garment_path=garment_result.get("path"),
                garment_url=garment_result.get("url", garment_image_url),
                garment_mask_path=None,
                config=config,
            )
        else:
            from services.pipeline.backends.local_composite import run_local_composite
            inference_result = await run_local_composite(
                person_path=person_result["path"],
                garment_path=garment_result.get("path"),
                garment_url=garment_result.get("url", garment_image_url),
                garment_mask_path=garment_result.get("mask_path"),
                config=config,
            )
        stage.status = "completed"
        stage.details = {"method": inference_result["method"]}
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        stage.status = "failed"
        stage.details = {"error": str(e)}
        raise RuntimeError(f"Inference failed: {e}") from e
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stages.append(stage)

    # ── Stage 4: Postprocess result ──────────────────────────────────
    from services.pipeline.postprocessor import postprocess_result

    stage = PipelineStage(name="postprocess", status="running")
    stage_start = time.time()
    try:
        post_result = await asyncio.to_thread(
            postprocess_result,
            result_path=inference_result["result_path"],
            original_person_path=user_photo_path,
            config=config,
        )
        stage.status = "completed"
        stage.details = post_result.get("details", {})
    except Exception as e:
        logger.warning("Postprocessing failed, using raw result: %s", e)
        post_result = {
            "result_path": inference_result["result_path"],
            "confidence_score": inference_result.get("raw_confidence", 0.5),
            "details": {"error": str(e)},
        }
        stage.status = "failed"
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stages.append(stage)

    # ── Stage 5: Garment-masked composite ────────────────────────────
    # intent: only enhance the garment region, keep original skin/face/bg untouched
    # status: done
    # confidence: high
    stage = PipelineStage(name="garment_composite", status="running")
    stage_start = time.time()
    try:
        from services.pipeline.postprocessing.garment_composite import composite_garment
        from PIL import Image as _PILImage

        result_img = _PILImage.open(post_result["result_path"]).convert("RGB")
        original_img = _PILImage.open(user_photo_path).convert("RGB")

        # Upscale VTON output to original resolution first
        if config.upscale_to_original and result_img.size != original_img.size:
            result_img = result_img.resize(original_img.size, _PILImage.LANCZOS)

        composited_img, composite_details = await asyncio.to_thread(
            composite_garment,
            result_img,
            original_img,
            None,  # auto-detect garment mask via semantic parser
            config.detail_sharpen_strength,
            config.lighting_transfer_strength,
            config.garment_type,  # "upper_body" → "tops", etc.
        )

        # Upscale to 4K if requested (beyond original resolution)
        # intent: produce high-res output for print/display, up to max_output_resolution
        # status: done
        # confidence: high
        if config.max_output_resolution > 0:
            w, h = composited_img.size
            max_dim = max(w, h)
            target = config.max_output_resolution
            if max_dim < target:
                scale = target / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                composited_img = composited_img.resize((new_w, new_h), _PILImage.LANCZOS)
                # Re-sharpen after upscale to recover detail
                from services.pipeline.postprocessing.detail_enhance import frequency_sharpen
                composited_img = frequency_sharpen(composited_img, strength=1.3)
                composite_details["upscaled_to"] = f"{new_w}x{new_h}"
                composite_details["upscale_factor"] = round(scale, 2)

        # Save composited result
        save_path = post_result["result_path"]
        if save_path.endswith(".png"):
            composited_img.save(save_path, "PNG")
        else:
            composited_img.save(save_path, "JPEG", quality=95)
        post_result["result_path"] = save_path

        stage.status = "completed"
        stage.details = composite_details
    except Exception as e:
        logger.warning("Garment composite failed, using raw result: %s", e)
        stage.status = "failed"
        stage.details = {"error": str(e)}
    stage.duration_ms = int((time.time() - stage_start) * 1000)
    stages.append(stage)

    # ── Build final result ───────────────────────────────────────────
    total_ms = int((time.time() - pipeline_start) * 1000)
    confidence = post_result.get("confidence_score", 0.5)

    if confidence >= 0.80:
        confidence_label = "high"
    elif confidence >= 0.60:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    result = PipelineResult(
        result_path=post_result["result_path"],
        confidence_score=round(confidence, 2),
        confidence_label=confidence_label,
        processing_time_ms=total_ms,
        method=backend,
        stages=stages,
    )

    logger.info(
        "Pipeline complete: method=%s confidence=%.2f time=%dms stages=%s",
        backend, confidence, total_ms, result.stage_summary,
    )

    return result
