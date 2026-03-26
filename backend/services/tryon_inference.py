"""Virtual try-on inference — delegates to the pipeline.

This module is the public API consumed by the tryon router.
All heavy lifting is in services.pipeline.
"""

from __future__ import annotations

from services.pipeline import run_pipeline, PipelineConfig, PipelineResult


async def run_tryon(
    user_photo_path: str,
    garment_image_url: str,
    garment_id: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full virtual try-on pipeline.

    Returns a PipelineResult with result_path, confidence, stages, etc.
    """
    return await run_pipeline(
        user_photo_path=user_photo_path,
        garment_image_url=garment_image_url,
        garment_id=garment_id,
        config=config,
    )
