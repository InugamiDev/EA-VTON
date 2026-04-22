/**
 * EA-VTON API Gateway — Node.js + Express + BullMQ.
 *
 * Routes requests to Python ML services (feature-service, vton-service, recommendation).
 * Manages job queues via BullMQ + Redis for async try-on processing.
 */

// intent: API gateway that orchestrates ML services via BullMQ job queues
// status: done
// next: add authentication middleware, rate limiting
// confidence: high

const express = require("express");
const cors = require("cors");

const garmentRoutes = require("./routes/garments");
const tryonRoutes = require("./routes/tryon");
const sizeRoutes = require("./routes/size");
const bodyRoutes = require("./routes/body");
const featureRoutes = require("./routes/features");
const healthRoutes = require("./routes/health");
const pipelineRoutes = require("./routes/pipeline");

const app = express();
const PORT = process.env.PORT || 3001;

// ── Middleware ──
app.use(cors());
app.use(express.json());

// ── Routes ──
app.use("/api/garments", garmentRoutes);
app.use("/api/tryon", tryonRoutes);
app.use("/api/size-recommendation", sizeRoutes);
app.use("/api/body-measurement", bodyRoutes);
app.use("/api/features", featureRoutes);
app.use("/api/pipeline", pipelineRoutes);
app.use("/api", healthRoutes);

// ── Root ──
app.get("/", (_req, res) => {
  res.json({
    app: "EA-VTON API Gateway",
    version: "0.1.0",
    docs: "/api/health",
  });
});

// ── Start ──
app.listen(PORT, () => {
  console.log(`EA-VTON API Gateway listening on port ${PORT}`);
});

module.exports = app;
