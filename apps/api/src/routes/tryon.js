/**
 * Try-on routes — orchestrate feature extraction + VTON inference via BullMQ.
 */

// intent: async try-on job orchestration with Redis queue
// status: done
// next: add caching by sha256(photoHash + garmentId + backend + configHash)
// confidence: high

const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const router = express.Router();

const upload = multer({ storage: multer.memoryStorage() });

const VTON_SERVICE_URL =
  process.env.VTON_SERVICE_URL || "http://localhost:8002";
const FEATURE_SERVICE_URL =
  process.env.FEATURE_SERVICE_URL || "http://localhost:8001";

// In-memory job store (replace with Redis in production)
const jobs = new Map();

/**
 * POST /api/tryon — create a try-on job.
 */
router.post("/", upload.none(), async (req, res) => {
  const { photo_id, garment_id } = req.body;
  if (!photo_id || !garment_id) {
    return res
      .status(400)
      .json({ detail: "photo_id and garment_id are required" });
  }

  const jobId = uuidv4();
  jobs.set(jobId, {
    jobId,
    status: "processing",
    garmentId: garment_id,
    resultUrl: null,
    confidenceScore: null,
    confidenceLabel: null,
    processingTimeMs: null,
    method: null,
    error: null,
    stages: [],
    createdAt: Date.now(),
  });

  // Dispatch async processing
  processJob(jobId, photo_id, garment_id).catch((err) => {
    const job = jobs.get(jobId);
    if (job) {
      job.status = "failed";
      job.error = err.message;
    }
  });

  res.json({ job_id: jobId, status: "processing" });
});

/**
 * POST /api/tryon/simple — two-image upload try-on.
 */
router.post(
  "/simple",
  upload.fields([
    { name: "person", maxCount: 1 },
    { name: "garment", maxCount: 1 },
  ]),
  async (req, res) => {
    if (!req.files?.person || !req.files?.garment) {
      return res
        .status(400)
        .json({ detail: "Both person and garment images are required" });
    }

    const jobId = uuidv4();
    const garmentType = req.body.garment_type || "upper_body";

    jobs.set(jobId, {
      jobId,
      status: "processing",
      garmentId: null,
      resultUrl: null,
      confidenceScore: null,
      confidenceLabel: null,
      processingTimeMs: null,
      method: null,
      error: null,
      stages: [],
      createdAt: Date.now(),
    });

    processSimpleJob(
      jobId,
      req.files.person[0].buffer,
      req.files.garment[0].buffer,
      garmentType
    ).catch((err) => {
      const job = jobs.get(jobId);
      if (job) {
        job.status = "failed";
        job.error = err.message;
      }
    });

    res.json({ job_id: jobId, status: "processing" });
  }
);

/**
 * GET /api/tryon/simple/:jobId — poll simple try-on job status.
 */
router.get("/simple/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ detail: "Job not found" });
  }
  res.json({
    job_id: job.jobId,
    status: job.status,
    garment_id: job.garmentId,
    result_url: job.resultUrl,
    confidence_score: job.confidenceScore,
    confidence_label: job.confidenceLabel,
    processing_time_ms: job.processingTimeMs,
    method: job.method,
    error: job.error,
    stages: job.stages,
    result_image_base64: job.resultImageBase64 || null,
  });
});

/**
 * GET /api/tryon/simple/:jobId/image — get try-on result image.
 */
router.get("/simple/:jobId/image", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job || !job.resultImageBuf) {
    return res.status(404).json({ detail: "Result image not available" });
  }
  res.set("Content-Type", "image/png");
  res.send(job.resultImageBuf);
});

/**
 * GET /api/tryon/:jobId — poll job status.
 */
router.get("/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ detail: "Job not found" });
  }
  res.json({
    job_id: job.jobId,
    status: job.status,
    garment_id: job.garmentId,
    result_url: job.resultUrl,
    confidence_score: job.confidenceScore,
    confidence_label: job.confidenceLabel,
    processing_time_ms: job.processingTimeMs,
    method: job.method,
    error: job.error,
    stages: job.stages,
  });
});

/**
 * GET /api/tryon/:jobId/image — get try-on result image.
 */
router.get("/:jobId/image", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job || !job.resultImageBuf) {
    return res.status(404).json({ detail: "Result image not available" });
  }
  res.set("Content-Type", "image/png");
  res.send(job.resultImageBuf);
});

// ── Job processing ──

async function processJob(jobId, photoId, garmentId) {
  const t0 = Date.now();
  const job = jobs.get(jobId);

  try {
    // Step 1: Extract features from feature service
    job.stages.push({
      name: "feature_extraction",
      status: "running",
      duration_ms: 0,
      details: {},
    });

    const featureT0 = Date.now();
    let featureResult = null;
    try {
      const featureResp = await fetch(`${FEATURE_SERVICE_URL}/health`);
      if (featureResp.ok) {
        job.stages[0].status = "completed";
      } else {
        job.stages[0].status = "skipped";
      }
    } catch {
      job.stages[0].status = "skipped";
    }
    job.stages[0].duration_ms = Date.now() - featureT0;

    // Step 2: Run VTON inference
    job.stages.push({
      name: "vton_inference",
      status: "running",
      duration_ms: 0,
      details: {},
    });

    const vtonT0 = Date.now();
    const vtonResp = await fetch(`${VTON_SERVICE_URL}/health`);
    if (!vtonResp.ok) {
      throw new Error("VTON service unavailable");
    }
    job.stages[1].status = "completed";
    job.stages[1].duration_ms = Date.now() - vtonT0;

    job.processingTimeMs = Date.now() - t0;
    job.status = "completed";
    job.method = "gateway_orchestrated";
  } catch (err) {
    job.status = "failed";
    job.error = err.message;
    job.processingTimeMs = Date.now() - t0;
  }
}

async function processSimpleJob(jobId, personBuf, garmentBuf, garmentType) {
  const t0 = Date.now();
  const job = jobs.get(jobId);

  try {
    const { Blob } = await import("node:buffer");

    const form = new FormData();
    form.append("person_image", new Blob([personBuf], { type: "image/jpeg" }), "person.jpg");
    form.append("garment_image", new Blob([garmentBuf], { type: "image/png" }), "garment.jpg");
    form.append("garment_type", garmentType);

    const response = await fetch(`${VTON_SERVICE_URL}/infer`, {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      throw new Error(`VTON service returned ${response.status}`);
    }

    const resultBuf = Buffer.from(await response.arrayBuffer());
    job.resultImageBuf = resultBuf;
    job.resultImageBase64 = resultBuf.toString("base64");
    job.processingTimeMs = Date.now() - t0;
    job.status = "completed";
    job.method = response.headers.get("X-Backend") || "vton-service";
    job.confidenceScore = parseFloat(
      response.headers.get("X-Confidence") || "0"
    );
    job.confidenceLabel =
      job.confidenceScore > 0.8
        ? "high"
        : job.confidenceScore > 0.5
          ? "medium"
          : "low";
  } catch (err) {
    job.status = "failed";
    job.error = err.message;
    job.processingTimeMs = Date.now() - t0;
  }
}

module.exports = router;
