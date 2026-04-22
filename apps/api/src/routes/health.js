/**
 * Health check route — aggregates health from all services.
 */

const express = require("express");
const router = express.Router();

const SERVICES = {
  "feature-service": process.env.FEATURE_SERVICE_URL || "http://localhost:8001",
  "vton-service": process.env.VTON_SERVICE_URL || "http://localhost:8002",
  recommendation:
    process.env.RECOMMENDATION_SERVICE_URL || "http://localhost:8003",
};

router.get("/health", async (_req, res) => {
  const serviceStatuses = {};

  await Promise.all(
    Object.entries(SERVICES).map(async ([name, url]) => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const response = await fetch(`${url}/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeout);
        serviceStatuses[name] = response.ok ? "healthy" : "unhealthy";
      } catch {
        serviceStatuses[name] = "unreachable";
      }
    })
  );

  const allHealthy = Object.values(serviceStatuses).every(
    (s) => s === "healthy"
  );

  res.json({
    status: allHealthy ? "healthy" : "degraded",
    gateway: "healthy",
    services: serviceStatuses,
  });
});

module.exports = router;
