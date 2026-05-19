"use client";

import { useState } from "react";
import { useMockAuth } from "@/hooks/use-mock-auth";
import { cn } from "@/lib/utils";
import { User, Ruler, Scale, Save, Trash2, Clock, LogIn, CheckCircle } from "lucide-react";

export default function ProfilePage() {
  const { user, isLoading, login, logout, updateProfile } = useMockAuth();

  const [height, setHeight] = useState<string>("");
  const [weight, setWeight] = useState<string>("");
  const [fitPreference, setFitPreference] = useState<"slim" | "regular" | "relaxed">("regular");
  const [saved, setSaved] = useState(false);
  const [initialized, setInitialized] = useState(false);

  // Sync form state when user loads
  if (user && !initialized) {
    setHeight(user.profile.heightCm?.toString() ?? "");
    setWeight(user.profile.weightKg?.toString() ?? "");
    setFitPreference(user.profile.fitPreference);
    setInitialized(true);
  }

  if (isLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!user) {
    return (
      <section className="py-16">
        <div className="mx-auto max-w-lg px-6 text-center">
          <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-muted">
            <User className="h-10 w-10 text-muted-foreground" />
          </div>
          <h1 className="mb-3 text-2xl font-bold text-foreground">
            Sign in to view your profile
          </h1>
          <p className="mb-8 text-base text-muted-foreground">
            Access your measurements, fit preferences, and try-on history.
          </p>
          <button
            onClick={login}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-medium text-primary-foreground",
              "transition-all duration-200 hover:opacity-90",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              "motion-reduce:transition-none"
            )}
          >
            <LogIn className="h-5 w-5" />
            Demo Login
          </button>
        </div>
      </section>
    );
  }

  const handleSave = () => {
    updateProfile({
      heightCm: height ? parseFloat(height) : null,
      weightKg: weight ? parseFloat(weight) : null,
      fitPreference,
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleDeleteData = () => {
    if (window.confirm("Are you sure you want to delete all your data? This action cannot be undone.")) {
      alert("Demo: All user data would be permanently deleted.");
      logout();
    }
  };

  const mockHistory = [
    { id: 1, garment: "Classic White Tee", date: "2 days ago", status: "completed" },
    { id: 2, garment: "Denim Jacket", date: "5 days ago", status: "completed" },
    { id: 3, garment: "Oversized Hoodie", date: "1 week ago", status: "completed" },
  ];

  return (
    <section className="py-16">
      <div className="mx-auto max-w-2xl px-6">
        <h1 className="mb-8 text-3xl font-bold text-foreground">Your Profile</h1>

        {/* Profile Info Card */}
        <div className="mb-8 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
              <User className="h-7 w-7 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground">{user.name}</h2>
              <p className="text-base text-muted-foreground">{user.email}</p>
            </div>
          </div>
          <div className="mt-4 flex justify-end">
            <button
              onClick={logout}
              className={cn(
                "rounded-lg border border-border px-6 py-4 text-base font-medium text-foreground",
                "transition-all duration-200 hover:bg-muted",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "motion-reduce:transition-none"
              )}
            >
              Sign Out
            </button>
          </div>
        </div>

        {/* Measurements Form */}
        <div className="mb-8 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <h2 className="mb-6 text-xl font-semibold text-foreground">Measurements</h2>

          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <label
                htmlFor="height"
                className="mb-2 flex items-center gap-2 text-base font-medium text-foreground"
              >
                <Ruler className="h-4 w-4 text-muted-foreground" />
                Height (cm)
              </label>
              <input
                id="height"
                type="number"
                placeholder="e.g. 175"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-border bg-background px-4 py-3 text-base text-foreground placeholder:text-muted-foreground",
                  "transition-all duration-200",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  "motion-reduce:transition-none"
                )}
              />
            </div>

            <div>
              <label
                htmlFor="weight"
                className="mb-2 flex items-center gap-2 text-base font-medium text-foreground"
              >
                <Scale className="h-4 w-4 text-muted-foreground" />
                Weight (kg)
              </label>
              <input
                id="weight"
                type="number"
                placeholder="e.g. 70"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-border bg-background px-4 py-3 text-base text-foreground placeholder:text-muted-foreground",
                  "transition-all duration-200",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  "motion-reduce:transition-none"
                )}
              />
            </div>
          </div>

          <div className="mt-6">
            <label
              htmlFor="fit"
              className="mb-2 block text-base font-medium text-foreground"
            >
              Fit Preference
            </label>
            <select
              id="fit"
              value={fitPreference}
              onChange={(e) => setFitPreference(e.target.value as "slim" | "regular" | "relaxed")}
              className={cn(
                "w-full rounded-lg border border-border bg-background px-4 py-3 text-base text-foreground",
                "transition-all duration-200",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "motion-reduce:transition-none"
              )}
            >
              <option value="slim">Slim</option>
              <option value="regular">Regular</option>
              <option value="relaxed">Relaxed</option>
            </select>
          </div>

          <div className="mt-6 flex items-center gap-4">
            <button
              onClick={handleSave}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-4 text-base font-medium text-primary-foreground",
                "transition-all duration-200 hover:opacity-90",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "motion-reduce:transition-none"
              )}
            >
              <Save className="h-5 w-5" />
              Save Measurements
            </button>
            {saved && (
              <span className="inline-flex items-center gap-1.5 text-base font-medium text-green-600">
                <CheckCircle className="h-5 w-5" />
                Saved successfully
              </span>
            )}
          </div>
        </div>

        {/* Try-On History */}
        <div className="mb-8 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <h2 className="mb-6 flex items-center gap-2 text-xl font-semibold text-foreground">
            <Clock className="h-5 w-5 text-muted-foreground" />
            Try-On History
          </h2>

          <div className="grid gap-4">
            {mockHistory.map((entry) => (
              <div
                key={entry.id}
                className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-6"
              >
                <div>
                  <p className="text-base font-medium text-foreground">{entry.garment}</p>
                  <p className="text-sm text-muted-foreground">{entry.date}</p>
                </div>
                <span className="rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-700">
                  {entry.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Danger Zone */}
        <div className="rounded-2xl border border-destructive/30 bg-card p-6 shadow-sm">
          <h2 className="mb-2 text-xl font-semibold text-destructive">Danger Zone</h2>
          <p className="mb-6 text-base text-muted-foreground">
            Permanently delete all your profile data, measurements, and try-on history.
            This action cannot be undone.
          </p>
          <button
            onClick={handleDeleteData}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg bg-destructive px-6 py-4 text-base font-medium text-white",
              "transition-all duration-200 hover:opacity-90",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              "motion-reduce:transition-none"
            )}
          >
            <Trash2 className="h-5 w-5" />
            Delete All My Data
          </button>
        </div>
      </div>
    </section>
  );
}
