/**
 * @ea-vton/types — shared TypeScript types for the EA-VTON monorepo.
 *
 * Consolidated from frontend/src/types/index.ts + backend/models.py.
 */

// ── Garment types ──

export interface GarmentImage {
  id: string;
  url: string;
  type: "display" | "tryon_input" | "flat_lay";
}

export interface SizeChartEntry {
  size: string;
  chestMin: number;
  chestMax: number;
  waistMin?: number;
  waistMax?: number;
  lengthCm?: number;
}

export type GarmentType =
  | "t-shirt"
  | "shirt"
  | "hoodie"
  | "jacket"
  | "sweater"
  | "polo"
  | "vest"
  | "dress"
  | "pants"
  | "skirt";

export interface Garment {
  id: string;
  name: string;
  type: GarmentType;
  brand: string;
  description: string;
  price: number;
  colors: string[];
  sizes: string[];
  images: GarmentImage[];
  sizeChart: SizeChartEntry[];
}

export interface GarmentListResponse {
  garments: Garment[];
  total: number;
  page: number;
  pageSize: number;
}

// ── Try-on types ──

export type JobStatus = "pending" | "processing" | "completed" | "failed";
export type ConfidenceLevel = "high" | "medium" | "low";

export interface PipelineStage {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  durationMs: number;
  details: Record<string, unknown>;
}

export interface TryOnRequest {
  id: string;
  garmentId: string;
  userPhotoUrl: string;
  status: JobStatus;
  createdAt: string;
}

export interface TryOnResult {
  id: string;
  requestId: string;
  resultImageUrl: string;
  confidenceScore: number;
  confidenceLabel: ConfidenceLevel;
  processingTimeMs: number;
}

export interface TryOnJob {
  jobId: string;
  status: JobStatus;
  garmentId: string | null;
  resultUrl: string | null;
  confidenceScore: number | null;
  confidenceLabel: ConfidenceLevel | null;
  processingTimeMs: number | null;
  method: string | null;
  error: string | null;
  stages: PipelineStage[];
}

// ── Size recommendation types ──

export type FitPreference = "slim" | "regular" | "relaxed";
export type Population = "us_women" | "universal" | "vietnamese";

export interface SizeAlternative {
  size: string;
  note: string;
}

export interface SizeRecommendation {
  size: string;
  confidence: ConfidenceLevel;
  explanation: string;
  alternatives: SizeAlternative[];
}

export interface SizeRecommendationResult {
  recommendedSize: string;
  confidence: ConfidenceLevel;
  explanation: string;
  estimatedChestCm: number;
  estimatedWaistCm: number;
  estimatedHipCm: number;
  alternatives: SizeAlternative[];
}

export interface PopulationEstimate {
  population: string;
  estimatedChestCm: number;
  estimatedWaistCm: number;
  estimatedHipCm: number;
  recommendedSize: string;
  confidence: ConfidenceLevel;
  alternatives: SizeAlternative[];
  nearestBodyType?: string;
  nearestBodyTypePct?: number;
}

export interface SizeComparisonResult {
  input: {
    heightCm: number;
    weightKg: number;
    fitPreference: string;
  };
  baseline: PopulationEstimate;
  eavton: PopulationEstimate;
  comparison: {
    sizeChanged: boolean;
    chestDifferenceCm: number;
    explanation: string;
  };
}

// ── Photo & quality types ──

export type QualityStatus = "pass" | "warn" | "fail";

export interface QualityCheck {
  status: QualityStatus;
  value: number | null;
}

export interface QualityAssessment {
  brightness: QualityCheck;
  blur: QualityCheck;
  dimensions: QualityCheck;
  overall: QualityStatus;
}

export interface PhotoUploadResult {
  photoId: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  quality: QualityAssessment;
}

// ── Body measurement types ──

export interface BodyMeasurementResult {
  chestCircumference: number;
  waistCircumference: number;
  hipCircumference: number;
  shoulderWidth: number;
  armLength: number;
  legLength: number;
  torsoLength: number;
  bmi: number;
  method: string;
  landmarkConfidence: number;
  warnings: string[];
  recommendedSize: string;
  sizeConfidence: ConfidenceLevel;
  alternatives: SizeAlternative[];
  populationComparison: SizeComparisonResult | null;
}

// ── User types ──

export interface UserProfile {
  displayName: string;
  heightCm: number | null;
  weightKg: number | null;
  fitPreference: FitPreference;
}

// ── Feature extraction types ──

export interface PoseData {
  keypoints: number[][];
  heatmapsShape: number[];
  heatmapsB64: string;
}

export interface ParsingData {
  shape: number[];
  classesPresent: number[];
  mapB64: string;
}

export interface FeatureExtractionResult {
  pose: PoseData | null;
  parsing: ParsingData | null;
  bodyEmbedding: number[] | null;
  bodyMeasurements: BodyMeasurementResult | null;
  processingTimeMs: number;
}

// ── Video try-on types ──

export interface VideoTryOnJob {
  jobId: string;
  status: JobStatus;
  stage: string | null;
  garmentId: string | null;
  resultUrl: string | null;
  totalFrames: number | null;
  keyframeCount: number | null;
  processingTimeMs: number | null;
  avgFps: number | null;
  error: string | null;
  stages: PipelineStage[];
}

// ── Health check ──

export interface HealthStatus {
  status: string;
  inferenceMode: string;
}

export interface ServiceHealth {
  status: string;
  service: string;
  models?: Record<string, string>;
  backends?: string[];
}
