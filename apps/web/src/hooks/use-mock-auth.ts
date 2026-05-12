"use client";

import { useState, useEffect } from "react";
import type { UserProfile } from "@/types";

const STORAGE_KEY = "fitview_user";

interface MockUser {
  id: string;
  email: string;
  name: string;
  profile: UserProfile;
}

const DEFAULT_USER: MockUser = {
  id: "demo-user-1",
  email: "demo@fitview.app",
  name: "Demo User",
  profile: {
    displayName: "Demo User",
    heightCm: null,
    weightKg: null,
    fitPreference: "regular",
  },
};

export function useMockAuth() {
  const [user, setUser] = useState<MockUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const loadId = window.setTimeout(() => {
      if (cancelled) return;
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setUser(JSON.parse(stored));
      }
      setIsLoading(false);
    }, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(loadId);
    };
  }, []);

  const login = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_USER));
    setUser(DEFAULT_USER);
  };

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  };

  const updateProfile = (profile: Partial<UserProfile>) => {
    if (!user) return;
    const updated = { ...user, profile: { ...user.profile, ...profile } };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    setUser(updated);
  };

  return { user, isLoading, login, logout, updateProfile };
}
