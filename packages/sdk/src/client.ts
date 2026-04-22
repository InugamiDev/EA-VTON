/**
 * @ea-vton/sdk — unified API client for the EA-VTON platform.
 *
 * Rewritten from frontend/src/lib/api.ts to work as a standalone SDK
 * that can be used by any app in the monorepo.
 */

import type {
  Garment,
  GarmentListResponse,
  TryOnJob,
  SizeRecommendationResult,
  SizeComparisonResult,
  BodyMeasurementResult,
  PhotoUploadResult,
  QualityAssessment,
  HealthStatus,
  FitPreference,
  Population,
  FeatureExtractionResult,
} from "@ea-vton/types";

// ── Client configuration ──

export interface EAVtonClientConfig {
  baseUrl: string;
  timeout?: number;
}

export class EAVtonClient {
  private baseUrl: string;
  private timeout: number;

  constructor(config: EAVtonClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.timeout = config.timeout ?? 30000;
  }

  // ── Photo upload ──

  async uploadPhoto(file: File | Blob): Promise<PhotoUploadResult> {
    const form = new FormData();
    form.append("file", file);
    return this.post<PhotoUploadResult>("/api/photos/upload", form);
  }

  // ── Try-on ──

  async createTryOnJob(
    photoId: string,
    garmentId: string
  ): Promise<{ jobId: string; status: string }> {
    const form = new FormData();
    form.append("photo_id", photoId);
    form.append("garment_id", garmentId);
    return this.post("/api/tryon", form);
  }

  async pollTryOnJob(jobId: string): Promise<TryOnJob> {
    return this.get<TryOnJob>(`/api/tryon/${jobId}`);
  }

  getResultImageUrl(jobId: string): string {
    return `${this.baseUrl}/api/tryon/${jobId}/image`;
  }

  getUploadedPhotoUrl(filename: string): string {
    return `${this.baseUrl}/uploads/${filename}`;
  }

  // ── Simple try-on (two image uploads) ──

  async createSimpleTryOn(
    person: File,
    garment: File,
    garmentType: string = "upper_body"
  ): Promise<{ jobId: string; status: string }> {
    const form = new FormData();
    form.append("person", person);
    form.append("garment", garment);
    form.append("garment_type", garmentType);
    return this.post("/api/tryon/simple", form);
  }

  async pollSimpleTryOn(jobId: string): Promise<TryOnJob> {
    return this.get<TryOnJob>(`/api/tryon/simple/${jobId}`);
  }

  getSimpleResultImageUrl(jobId: string): string {
    return `${this.baseUrl}/api/tryon/simple/${jobId}/image`;
  }

  // ── Size recommendation ──

  async getSizeRecommendation(params: {
    garmentId: string;
    heightCm: number;
    weightKg: number;
    fitPreference: FitPreference;
  }): Promise<SizeRecommendationResult> {
    return this.post<SizeRecommendationResult>("/api/size-recommendation", {
      garment_id: params.garmentId,
      height_cm: params.heightCm,
      weight_kg: params.weightKg,
      fit_preference: params.fitPreference,
    });
  }

  async compareSizeRecommendation(params: {
    garmentId: string;
    heightCm: number;
    weightKg: number;
    fitPreference: FitPreference;
  }): Promise<SizeComparisonResult> {
    return this.post<SizeComparisonResult>("/api/size-recommendation/compare", {
      garment_id: params.garmentId,
      height_cm: params.heightCm,
      weight_kg: params.weightKg,
      fit_preference: params.fitPreference,
    });
  }

  // ── Body measurement ──

  async estimateBodyMeasurements(
    photo: Blob,
    heightCm: number,
    weightKg: number,
    population: Population = "universal",
    fitPreference: FitPreference = "regular"
  ): Promise<BodyMeasurementResult> {
    const form = new FormData();
    form.append("photo", photo, "capture.jpg");
    form.append("height_cm", heightCm.toString());
    form.append("weight_kg", weightKg.toString());
    form.append("population", population);
    form.append("fit_preference", fitPreference);
    return this.post<BodyMeasurementResult>("/api/body-measurement", form);
  }

  async quickBodyMeasurement(params: {
    heightCm: number;
    weightKg: number;
    population?: Population;
    fitPreference?: FitPreference;
  }): Promise<BodyMeasurementResult> {
    return this.post<BodyMeasurementResult>("/api/body-measurement/quick", {
      height_cm: params.heightCm,
      weight_kg: params.weightKg,
      population: params.population,
      fit_preference: params.fitPreference,
    });
  }

  // ── Feature extraction ──

  async extractFeatures(
    image: Blob,
    options?: {
      includePose?: boolean;
      includeParsing?: boolean;
      includeEmbedding?: boolean;
      heightCm?: number;
      weightKg?: number;
    }
  ): Promise<FeatureExtractionResult> {
    const form = new FormData();
    form.append("image", image);
    if (options?.includePose !== undefined)
      form.append("include_pose", String(options.includePose));
    if (options?.includeParsing !== undefined)
      form.append("include_parsing", String(options.includeParsing));
    if (options?.includeEmbedding !== undefined)
      form.append("include_embedding", String(options.includeEmbedding));
    if (options?.heightCm !== undefined)
      form.append("height_cm", String(options.heightCm));
    if (options?.weightKg !== undefined)
      form.append("weight_kg", String(options.weightKg));
    return this.post<FeatureExtractionResult>(
      "/api/features/extract",
      form
    );
  }

  async checkImageQuality(image: Blob): Promise<QualityAssessment> {
    const form = new FormData();
    form.append("image", image);
    return this.post<QualityAssessment>("/api/features/quality-check", form);
  }

  // ── Garments ──

  async listGarments(
    page: number = 1,
    pageSize: number = 20
  ): Promise<GarmentListResponse> {
    return this.get<GarmentListResponse>(
      `/api/garments?page=${page}&page_size=${pageSize}`
    );
  }

  async getGarment(id: string): Promise<Garment> {
    return this.get<Garment>(`/api/garments/${id}`);
  }

  // ── Health ──

  async checkHealth(): Promise<HealthStatus> {
    return this.get<HealthStatus>("/api/health");
  }

  // ── HTTP helpers ──

  private async get<T>(path: string): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new EAVtonError(err.detail || `Request failed: ${res.status}`, res.status);
      }

      return res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async post<T>(
    path: string,
    body: FormData | Record<string, unknown>
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    const isFormData = body instanceof FormData;

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: isFormData ? {} : { "Content-Type": "application/json" },
        body: isFormData ? body : JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new EAVtonError(err.detail || `Request failed: ${res.status}`, res.status);
      }

      return res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

export class EAVtonError extends Error {
  constructor(
    message: string,
    public statusCode: number
  ) {
    super(message);
    this.name = "EAVtonError";
  }
}

/**
 * Create a pre-configured client instance.
 */
export function createClient(
  baseUrl: string = "http://localhost:3001"
): EAVtonClient {
  return new EAVtonClient({ baseUrl });
}
