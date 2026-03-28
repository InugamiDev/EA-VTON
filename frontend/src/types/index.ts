export interface Garment {
  id: string;
  name: string;
  type: "t-shirt" | "shirt" | "hoodie" | "jacket" | "sweater" | "polo" | "vest";
  brand: string;
  description: string;
  price: number;
  colors: string[];
  sizes: string[];
  images: GarmentImage[];
  sizeChart: SizeChartEntry[];
}

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

export interface TryOnRequest {
  id: string;
  garmentId: string;
  userPhotoUrl: string;
  status: "pending" | "processing" | "completed" | "failed";
  createdAt: string;
}

export interface TryOnResult {
  id: string;
  requestId: string;
  resultImageUrl: string;
  confidenceScore: number;
  confidenceLabel: "high" | "medium" | "low";
  processingTimeMs: number;
}

export interface SizeRecommendation {
  size: string;
  confidence: "high" | "medium" | "low";
  explanation: string;
  alternatives: { size: string; note: string }[];
}

export interface UserProfile {
  displayName: string;
  heightCm: number | null;
  weightKg: number | null;
  fitPreference: "slim" | "regular" | "relaxed";
}

export interface PhotoQuality {
  brightness: "pass" | "warn" | "fail";
  blur: "pass" | "warn" | "fail";
  framing: "pass" | "warn" | "fail";
  overall: "pass" | "warn" | "fail";
}
