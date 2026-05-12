/**
 * FitView Backend API client.
 *
 * All try-on, upload, and recommendation calls go through here.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

/* ------------------------------------------------------------------ */
/*  Photo upload                                                       */
/* ------------------------------------------------------------------ */

export interface QualityCheck {
  status: "pass" | "warn" | "fail";
  value: number | null;
}

export interface QualityAssessment {
  brightness: QualityCheck;
  blur: QualityCheck;
  dimensions: QualityCheck;
  overall: "pass" | "warn" | "fail";
}

export interface UploadResult {
  photo_id: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  quality: QualityAssessment;
}

export async function uploadPhoto(file: File | Blob): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/photos/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Participant dataset intake                                         */
/* ------------------------------------------------------------------ */

export interface IntakeSubmissionResult {
  submission_id: string;
  person_id: string;
  image_count: number;
  next_step: string;
}

export interface IntakeStatusResult {
  storage: "restricted_local";
  records: number;
  people: number;
  min_images_per_submission: number;
  max_images_per_submission: number;
  raw_manifest: string;
}

export async function submitParticipantIntake(params: {
  files: File[];
  age_band: string;
  participant_note: string;
  consent_training: boolean;
  consent_research: boolean;
}): Promise<IntakeSubmissionResult> {
  const form = new FormData();
  form.append("gender_label", "female");
  form.append("age_band", params.age_band);
  form.append("participant_note", params.participant_note);
  form.append("consent_training", String(params.consent_training));
  form.append("consent_research", String(params.consent_research));
  for (const file of params.files) {
    form.append("images", file);
  }

  const res = await fetch(`${API_BASE}/api/intake/participant`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Intake submission failed");
  }

  return res.json();
}

export async function getIntakeStatus(): Promise<IntakeStatusResult> {
  const res = await fetch(`${API_BASE}/api/intake/status`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to load intake status");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Try-on job                                                         */
/* ------------------------------------------------------------------ */

export interface PipelineStage {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  duration_ms: number;
  details: Record<string, unknown>;
}

export interface TryOnJob {
  job_id: string;
  status: "processing" | "completed" | "failed";
  garment_id: string | null;
  result_url: string | null;
  confidence_score: number | null;
  confidence_label: "high" | "medium" | "low" | null;
  processing_time_ms: number | null;
  method: string | null;
  error: string | null;
  stages: PipelineStage[];
}

export async function createTryOnJob(
  photoId: string,
  garmentId: string
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append("photo_id", photoId);
  form.append("garment_id", garmentId);

  const res = await fetch(`${API_BASE}/api/tryon`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to create try-on job");
  }

  return res.json();
}

export async function pollTryOnJob(jobId: string): Promise<TryOnJob> {
  const res = await fetch(`${API_BASE}/api/tryon/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get job status");
  }

  return res.json();
}

/**
 * Get the full URL for a try-on result image.
 */
export function getResultImageUrl(jobId: string): string {
  return `${API_BASE}/api/tryon/${jobId}/image`;
}

/**
 * Get the full URL for an uploaded photo.
 */
export function getUploadedPhotoUrl(filename: string): string {
  return `${API_BASE}/uploads/${filename}`;
}

/* ------------------------------------------------------------------ */
/*  Simple try-on (two image uploads)                                  */
/* ------------------------------------------------------------------ */

export async function createSimpleTryOn(
  person: File,
  garment: File,
  garmentType: string = "upper_body"
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append("person", person);
  form.append("garment", garment);
  form.append("garment_type", garmentType);

  const res = await fetch(`${API_BASE}/api/tryon/simple`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to create try-on job");
  }

  return res.json();
}

export async function pollSimpleTryOn(jobId: string): Promise<TryOnJob> {
  const res = await fetch(`${API_BASE}/api/tryon/simple/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get job status");
  }

  return res.json();
}

export function getSimpleResultImageUrl(jobId: string): string {
  return `${API_BASE}/api/tryon/simple/${jobId}/image`;
}

/* ------------------------------------------------------------------ */
/*  Size recommendation                                                */
/* ------------------------------------------------------------------ */

export interface SizeRecommendationResult {
  recommended_size: string;
  confidence: "high" | "medium" | "low";
  explanation: string;
  estimated_chest_cm: number;
  alternatives: { size: string; note: string }[];
}

export async function getSizeRecommendation(params: {
  garment_id: string;
  height_cm: number;
  weight_kg: number;
  fit_preference: "slim" | "regular" | "relaxed";
}): Promise<SizeRecommendationResult> {
  const res = await fetch(`${API_BASE}/api/size-recommendation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get size recommendation");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  EA-VTON Size Comparison                                            */
/* ------------------------------------------------------------------ */

export interface PopulationEstimate {
  population: string;
  estimated_chest_cm: number;
  estimated_waist_cm: number;
  estimated_hip_cm: number;
  recommended_size: string;
  confidence: "high" | "medium" | "low";
  alternatives: { size: string; note: string }[];
  nearest_body_type?: string;
  nearest_body_type_pct?: number;
}

export interface SizeComparisonResult {
  input: {
    height_cm: number;
    weight_kg: number;
    fit_preference: string;
  };
  baseline: PopulationEstimate;
  eavton: PopulationEstimate;
  comparison: {
    size_changed: boolean;
    chest_difference_cm: number;
    explanation: string;
  };
}

export async function compareSizeRecommendation(params: {
  garment_id: string;
  height_cm: number;
  weight_kg: number;
  fit_preference: "slim" | "regular" | "relaxed";
}): Promise<SizeComparisonResult> {
  const res = await fetch(`${API_BASE}/api/size-recommendation/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to compare size recommendations");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Body measurement estimation                                        */
/* ------------------------------------------------------------------ */

export interface BodyMeasurementResult {
  chest_circumference: number;
  waist_circumference: number;
  hip_circumference: number;
  shoulder_width: number;
  arm_length: number;
  leg_length: number;
  torso_length: number;
  bmi: number;
  method: string;
  landmark_confidence: number;
  warnings: string[];
  recommended_size: string;
  size_confidence: "high" | "medium" | "low";
  alternatives: { size: string; note: string }[];
  population_comparison: SizeComparisonResult | null;
}

export interface LiveBodyMeasurementResult {
  chest_circumference: number;
  waist_circumference: number;
  hip_circumference: number;
  shoulder_width: number;
  arm_length: number;
  leg_length: number;
  torso_length: number;
  bmi: number;
  method: string;
  landmark_confidence: number;
  warnings: string[];
}

export interface ResearchSizeRecommendationResult {
  research_model: {
    predicted_size: string | null;
    confidence: "high" | "medium" | "low";
    confidence_score?: number;
    probabilities?: Record<string, number>;
    model: string;
    error?: string;
  };
  rule_based: {
    predicted_size: string | null;
    confidence: "high" | "medium" | "low";
    estimated_chest_cm?: number;
    estimated_waist_cm?: number;
    estimated_hip_cm?: number;
    alternatives?: { size: string; note: string }[];
  };
  input_summary: {
    height_cm: number;
    weight_kg: number;
    bmi: number;
    age: number;
    body_type: string;
    category: string;
    population: string;
  };
}

export async function estimateBodyMeasurements(
  photo: Blob,
  heightCm: number,
  weightKg: number,
  population: string = "us_women",
  fitPreference: string = "regular"
): Promise<BodyMeasurementResult> {
  const form = new FormData();
  form.append("photo", photo, "capture.jpg");
  form.append("height_cm", heightCm.toString());
  form.append("weight_kg", weightKg.toString());
  form.append("population", population);
  form.append("fit_preference", fitPreference);

  const res = await fetch(`${API_BASE}/api/body-measurement`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to estimate body measurements");
  }

  return res.json();
}

export async function getLiveBodyMeasurements(
  photo: Blob,
  heightCm: number,
  weightKg: number
): Promise<LiveBodyMeasurementResult> {
  const form = new FormData();
  form.append("photo", photo, "live-capture.jpg");
  form.append("height_cm", heightCm.toString());
  form.append("weight_kg", weightKg.toString());

  const res = await fetch(`${API_BASE}/api/body-measurement`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to extract live body measurements");
  }

  return res.json();
}

export async function quickBodyMeasurement(params: {
  height_cm: number;
  weight_kg: number;
  population?: string;
  fit_preference?: string;
}): Promise<BodyMeasurementResult> {
  const res = await fetch(`${API_BASE}/api/body-measurement/quick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get quick measurement");
  }

  return res.json();
}

export async function getResearchSizeRecommendation(params: {
  height_cm: number;
  weight_kg: number;
  age?: number;
  body_type?: string;
  category?: string;
  population?: "us_women" | "universal" | "vietnamese";
}): Promise<ResearchSizeRecommendationResult> {
  const res = await fetch(`${API_BASE}/api/size-recommendation/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get research size recommendation");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Visual size recommendation (webcam body ratios → no weight needed)  */
/* ------------------------------------------------------------------ */

export interface VisualSizeResult {
  research_model: {
    predicted_size: string | null;
    confidence: "high" | "medium" | "low";
    confidence_score?: number;
    probabilities?: Record<string, number>;
    model: string;
    error?: string;
  };
  rule_based: {
    predicted_size: string | null;
    confidence: "high" | "medium" | "low";
    estimated_chest_cm?: number;
    estimated_waist_cm?: number;
    estimated_hip_cm?: number;
  };
  weight_estimation: {
    estimated_weight_kg: number;
    estimated_bmi: number;
    method: string;
    note: string;
  };
  body_shape: {
    type: string;
    shoulder_to_hip: number;
    waist_to_hip: number;
    shoulder_to_torso: number;
    torso_to_leg: number;
    arm_to_torso: number;
  };
  input_summary: {
    height_cm: number;
    weight_kg: number;
    bmi: number;
    age: number;
    category: string;
    population: string;
    landmark_confidence: number;
    source: string;
  };
}

export async function getVisualSizeRecommendation(params: {
  height_cm: number;
  shoulder_to_hip: number;
  waist_to_hip: number;
  shoulder_to_torso?: number;
  torso_to_leg?: number;
  arm_to_torso?: number;
  age?: number;
  category?: string;
  population?: "us_women" | "universal" | "vietnamese";
  confidence?: number;
  source?: "world_3d" | "image_2d";
}): Promise<VisualSizeResult> {
  const res = await fetch(`${API_BASE}/api/size-recommendation/visual`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get visual size recommendation");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Upper-body style recommendation                                    */
/* ------------------------------------------------------------------ */

// intent: expose the existing recommendation-service style endpoint to frontend flows
// status: done
// next: use explicit merchandised style attributes when the catalog exposes them
// blockers: none
// confidence: high

export type UpperStyleSize = "XS" | "S" | "M" | "L" | "XL" | "XXL";
export type UpperStyleOccasion = "work" | "casual" | "date" | "formal" | "party";
export type PopulationKey = "us_women" | "universal" | "vietnamese";

export interface UpperStyleRecommendationRequest {
  height_cm: number;
  predicted_size: UpperStyleSize;
  population?: PopulationKey;
  ratios: {
    sh: number;
    wh: number;
    st?: number;
    tl?: number;
    at?: number;
  };
  context: {
    occasion: UpperStyleOccasion;
    sliders: Partial<Record<"bold" | "loose" | "warm" | "cover", number>>;
  };
  user_palette_lab?: number[] | null;
  top_k?: number;
  weights?: Partial<Record<"alpha" | "beta" | "gamma", number>>;
}

export interface UpperStyleRecommendationResult {
  user_state: {
    body_shape?: {
      id: number;
      label: string;
      display_label: string;
      confidence: number;
    };
    body_type_vn?: {
      cluster_id: number;
      label: string;
      label_vi: string;
      confidence: number;
    };
    predicted_size: UpperStyleSize;
    population?: string;
    size_band_midpoints_cm?: Record<string, number>;
  };
  weights: {
    alpha: number;
    beta: number;
    gamma: number;
  };
  items: UpperStyleRecommendationItem[];
}

export interface UpperStyleRecommendationItem {
  item_id: string;
  title: string;
  brand: string;
  image_url: string;
  price_vnd: number;
  category: string;
  garment_parameters?: {
    material_tags?: string[];
    fabric_weight?: string;
    structure?: string;
    closure?: string;
    layering?: string;
    weather_tags?: string[];
  };
  rank: number;
  total_score: number;
  explanation: {
    fit: {
      score: number;
      drape_class: string;
      overall_verdict: string;
      sections?: Array<{
        section: string;
        verdict: string;
        user_cm?: number;
        garment_cm?: number;
      }>;
    };
    flatter: {
      score: number;
      cluster_id: number;
      rules_fired?: Array<{
        attribute_dim: string;
        attribute_value: string;
        polarity: number;
        verdict: string;
      }>;
    };
    match: {
      score: number;
      occasion_ok: boolean;
      color?: {
        score: number;
        delta_e76: number | null;
      };
      sliders?: {
        score: number;
        item_projection: Record<string, number>;
      };
    };
  };
}

export async function getUpperStyleRecommendation(
  params: UpperStyleRecommendationRequest
): Promise<UpperStyleRecommendationResult> {
  const res = await fetch(`${API_BASE}/api/style/upper`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get style recommendations");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Full Pipeline                                                       */
/* ------------------------------------------------------------------ */

export interface PipelineResult {
  job_id: string;
  status: "processing" | "completed" | "failed";
  stages: Record<string, { status: string; durationMs?: number; error?: string }>;
  body_measurements: BodyMeasurementResult | null;
  size_recommendation: SizeRecommendationResult | null;
  tryon_result: {
    image_base64: string;
    backend: string;
    confidence: number;
  } | null;
  error: string | null;
  processingTimeMs: number | null;
}

export async function startPipeline(
  person: Blob,
  heightCm: number,
  options?: {
    garment?: Blob;
    garmentId?: string;
    weightKg?: number;
    population?: string;
    fitPreference?: string;
  }
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append("person", person, "person.jpg");
  form.append("height_cm", heightCm.toString());
  if (options?.garment) form.append("garment", options.garment, "garment.jpg");
  if (options?.garmentId) form.append("garment_id", options.garmentId);
  if (options?.weightKg) form.append("weight_kg", options.weightKg.toString());
  if (options?.population) form.append("population", options.population);
  if (options?.fitPreference) form.append("fit_preference", options.fitPreference);

  const res = await fetch(`${API_BASE}/api/pipeline`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to start pipeline");
  }

  return res.json();
}

export async function pollPipeline(jobId: string): Promise<PipelineResult> {
  const res = await fetch(`${API_BASE}/api/pipeline/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get pipeline status");
  }

  return res.json();
}

export async function quickPipeline(params: {
  height_cm: number;
  weight_kg?: number;
  population?: string;
  fit_preference?: string;
  garment_id?: string;
}): Promise<{
  body_measurements: BodyMeasurementResult | null;
  size_recommendation: SizeRecommendationResult | null;
}> {
  const res = await fetch(`${API_BASE}/api/pipeline/quick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Quick pipeline failed");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Research: Compare All Model Variants                               */
/* ------------------------------------------------------------------ */

export interface ModelComparisonResult {
  models: Record<string, {
    predicted_size: string;
    confidence: number;
    method: string;
  }>;
  recommended: string;
}

export async function compareAllModels(params: {
  height_cm: number;
  weight_kg: number;
  age?: number;
  body_type?: string;
  category?: string;
}): Promise<ModelComparisonResult> {
  const res = await fetch(`${API_BASE}/api/size-recommendation/all-models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Model comparison failed");
  }

  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Health check                                                       */
/* ------------------------------------------------------------------ */

export interface HealthStatus {
  status: string;
  inference_mode: string;
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}
