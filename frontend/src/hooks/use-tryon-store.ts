"use client";

import { useState } from "react";
import type { Garment, PhotoQuality, TryOnResult, SizeRecommendation } from "@/types";

interface TryOnState {
  selectedGarment: Garment | null;
  userPhoto: string | null;
  userPhotoFile: File | null;
  photoQuality: PhotoQuality | null;
  result: TryOnResult | null;
  sizeRecommendation: SizeRecommendation | null;
  jobStatus: "idle" | "uploading" | "processing" | "completed" | "failed";
  errorMessage: string | null;
}

const initialState: TryOnState = {
  selectedGarment: null,
  userPhoto: null,
  userPhotoFile: null,
  photoQuality: null,
  result: null,
  sizeRecommendation: null,
  jobStatus: "idle",
  errorMessage: null,
};

export function useTryOnStore() {
  const [state, setState] = useState<TryOnState>(initialState);

  const setGarment = (garment: Garment) =>
    setState((s) => ({ ...s, selectedGarment: garment }));

  const setUserPhoto = (photo: string, file: File | null) =>
    setState((s) => ({ ...s, userPhoto: photo, userPhotoFile: file }));

  const setPhotoQuality = (quality: PhotoQuality) =>
    setState((s) => ({ ...s, photoQuality: quality }));

  const setJobStatus = (status: TryOnState["jobStatus"], error?: string) =>
    setState((s) => ({
      ...s,
      jobStatus: status,
      errorMessage: error ?? null,
    }));

  const setResult = (result: TryOnResult) =>
    setState((s) => ({ ...s, result, jobStatus: "completed" }));

  const setSizeRecommendation = (rec: SizeRecommendation) =>
    setState((s) => ({ ...s, sizeRecommendation: rec }));

  const reset = () => setState(initialState);

  return {
    ...state,
    setGarment,
    setUserPhoto,
    setPhotoQuality,
    setJobStatus,
    setResult,
    setSizeRecommendation,
    reset,
  };
}
