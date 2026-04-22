/**
 * Full pipeline route — photo + height → body measurements + size rec + VTON.
 *
 * intent: single endpoint that orchestrates the complete EA-VTON flow
 * status: done
 * next: add caching, streaming progress updates
 * confidence: high
 *
 * Flow:
 *   1. Extract body measurements from photo (feature-service)
 *   2. Get size recommendation using measurements (recommendation)
 *   3. Run VTON with person photo + selected garment (vton-service)
 *   4. Return combined result
 */

const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const router = express.Router();

const upload = multer({ storage: multer.memoryStorage() });

const FEATURE_SERVICE_URL =
  process.env.FEATURE_SERVICE_URL || "http://localhost:8001";
const RECOMMENDATION_URL =
  process.env.RECOMMENDATION_SERVICE_URL || "http://localhost:8003";
const VTON_SERVICE_URL =
  process.env.VTON_SERVICE_URL || "http://localhost:8002";

// In-memory job store
const pipelineJobs = new Map();

/**
 * POST /api/pipeline — full pipeline with photo + garment.
 *
 * Body (multipart):
 *   person: File — person photo
 *   garment: File — garment image (optional if garment_id provided)
 *   garment_id: string — garment catalog ID (optional if garment file provided)
 *   height_cm: number
 *   weight_kg: number (optional)
 *   population: "universal" | "vietnamese" (default: "vietnamese")
 *   fit_preference: "slim" | "regular" | "relaxed" (default: "regular")
 */
router.post(
  "/",
  upload.fields([
    { name: "person", maxCount: 1 },
    { name: "garment", maxCount: 1 },
  ]),
  async (req, res) => {
    if (!req.files?.person) {
      return res.status(400).json({ detail: "Person photo is required" });
    }

    const heightCm = parseFloat(req.body.height_cm);
    if (!heightCm || heightCm < 100 || heightCm > 250) {
      return res.status(400).json({ detail: "Valid height_cm (100-250) is required" });
    }

    const jobId = uuidv4();
    const job = {
      jobId,
      status: "processing",
      stages: {},
      body_measurements: null,
      size_recommendation: null,
      tryon_result: null,
      error: null,
      createdAt: Date.now(),
      processingTimeMs: null,
    };
    pipelineJobs.set(jobId, job);

    // Run pipeline async
    runPipeline(
      jobId,
      req.files.person[0].buffer,
      req.files.garment?.[0]?.buffer || null,
      req.body.garment_id || null,
      heightCm,
      parseFloat(req.body.weight_kg) || null,
      req.body.population || "vietnamese",
      req.body.fit_preference || "regular"
    ).catch((err) => {
      job.status = "failed";
      job.error = err.message;
      job.processingTimeMs = Date.now() - job.createdAt;
    });

    res.json({ job_id: jobId, status: "processing" });
  }
);

/**
 * GET /api/pipeline/:jobId — poll pipeline job status.
 */
router.get("/:jobId", (req, res) => {
  const job = pipelineJobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ detail: "Pipeline job not found" });
  }
  res.json(job);
});

/**
 * POST /api/pipeline/quick — no photo, just height+weight → size rec.
 * Synchronous, returns immediately.
 */
router.post("/quick", async (req, res) => {
  const { height_cm, weight_kg, population, fit_preference, garment_id } = req.body;

  if (!height_cm) {
    return res.status(400).json({ detail: "height_cm is required" });
  }

  const result = { body_measurements: null, size_recommendation: null };

  try {
    // Step 1: Quick body measurements (no photo)
    const measResp = await fetch(`${FEATURE_SERVICE_URL}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ height_cm, weight_kg }),
    });

    if (measResp.ok) {
      result.body_measurements = await measResp.json();
    }
  } catch {
    // Feature service unavailable — continue without measurements
  }

  try {
    // Step 2: Size recommendation
    let sizeChart = null;
    if (garment_id) {
      const garments = require("../data/garments.json");
      const garment = garments.find((g) => g.id === garment_id);
      if (garment) sizeChart = garment.sizeChart;
    }

    // Try research model first
    const researchResp = await fetch(`${RECOMMENDATION_URL}/recommend-size/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm,
        weight_kg: weight_kg || 0,
        age: 30,
        body_type: "unknown",
        category: "dress",
        population: population || "vietnamese",
      }),
    });

    if (researchResp.ok) {
      result.size_recommendation = await researchResp.json();
    } else {
      // Fall back to rule-based
      const ruleResp = await fetch(`${RECOMMENDATION_URL}/recommend-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          height_cm,
          weight_kg: weight_kg || 55,
          fit_preference: fit_preference || "regular",
          population: population || "vietnamese",
          size_chart: sizeChart,
        }),
      });
      if (ruleResp.ok) {
        result.size_recommendation = await ruleResp.json();
      }
    }
  } catch {
    // Recommendation service unavailable
  }

  res.json(result);
});

// ── Pipeline execution ──

async function runPipeline(
  jobId, personBuf, garmentBuf, garmentId,
  heightCm, weightKg, population, fitPreference
) {
  const job = pipelineJobs.get(jobId);
  const t0 = Date.now();

  // Stage 1: Body measurements
  job.stages.body_measurement = { status: "running", startedAt: Date.now() };
  try {
    const { Blob } = await import("node:buffer");

    const form = new FormData();
    form.append("image", new Blob([personBuf]), "person.jpg");
    form.append("include_pose", "true");
    form.append("include_parsing", "false");
    form.append("include_embedding", "false");
    form.append("height_cm", heightCm.toString());
    if (weightKg) form.append("weight_kg", weightKg.toString());

    const resp = await fetch(`${FEATURE_SERVICE_URL}/extract`, {
      method: "POST",
      body: form,
    });

    if (resp.ok) {
      const data = await resp.json();
      job.body_measurements = data.body_measurements || data;
      job.stages.body_measurement.status = "completed";
    } else {
      job.stages.body_measurement.status = "failed";
    }
  } catch (err) {
    job.stages.body_measurement.status = "failed";
    job.stages.body_measurement.error = err.message;
  }
  job.stages.body_measurement.durationMs = Date.now() - job.stages.body_measurement.startedAt;

  // Stage 2: Size recommendation
  job.stages.size_recommendation = { status: "running", startedAt: Date.now() };
  try {
    // Use research model (trained GBM)
    const resp = await fetch(`${RECOMMENDATION_URL}/recommend-size/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm: heightCm,
        weight_kg: weightKg || (job.body_measurements?.bmi
          ? Math.sqrt(job.body_measurements.bmi * (heightCm / 100) ** 2)
          : 55),
        age: 30,
        body_type: "unknown",
        category: "dress",
        population,
      }),
    });

    if (resp.ok) {
      job.size_recommendation = await resp.json();
      job.stages.size_recommendation.status = "completed";
    } else {
      // Fallback to rule-based
      let sizeChart = null;
      if (garmentId) {
        const garments = require("../data/garments.json");
        const garment = garments.find((g) => g.id === garmentId);
        if (garment) sizeChart = garment.sizeChart;
      }
      const fallResp = await fetch(`${RECOMMENDATION_URL}/recommend-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          height_cm: heightCm,
          weight_kg: weightKg || 55,
          fit_preference: fitPreference,
          population,
          size_chart: sizeChart,
        }),
      });
      if (fallResp.ok) {
        job.size_recommendation = await fallResp.json();
        job.size_recommendation.method = "rule_based_fallback";
        job.stages.size_recommendation.status = "completed";
      } else {
        job.stages.size_recommendation.status = "failed";
      }
    }
  } catch (err) {
    job.stages.size_recommendation.status = "failed";
    job.stages.size_recommendation.error = err.message;
  }
  job.stages.size_recommendation.durationMs = Date.now() - job.stages.size_recommendation.startedAt;

  // Stage 3: VTON (if garment provided)
  if (garmentBuf || garmentId) {
    job.stages.vton = { status: "running", startedAt: Date.now() };
    try {
      const { Blob } = await import("node:buffer");

      const form = new FormData();
      form.append("person_image", new Blob([personBuf], { type: "image/jpeg" }), "person.jpg");

      if (garmentBuf) {
        form.append("garment_image", new Blob([garmentBuf], { type: "image/png" }), "garment.jpg");
      }
      form.append("garment_type", "upper_body");

      const resp = await fetch(`${VTON_SERVICE_URL}/infer`, {
        method: "POST",
        body: form,
      });

      if (resp.ok) {
        const resultBuf = Buffer.from(await resp.arrayBuffer());
        job.tryon_result = {
          image_base64: resultBuf.toString("base64"),
          backend: resp.headers.get("X-Backend") || "unknown",
          confidence: parseFloat(resp.headers.get("X-Confidence") || "0"),
        };
        job.stages.vton.status = "completed";
      } else {
        job.stages.vton.status = "failed";
        job.stages.vton.error = `VTON returned ${resp.status}`;
      }
    } catch (err) {
      job.stages.vton.status = "failed";
      job.stages.vton.error = err.message;
    }
    job.stages.vton.durationMs = Date.now() - job.stages.vton.startedAt;
  }

  job.processingTimeMs = Date.now() - t0;
  job.status = "completed";
}

module.exports = router;
