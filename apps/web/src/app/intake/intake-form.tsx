"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, LockKeyhole, UploadCloud } from "lucide-react";
import {
  getIntakeStatus,
  submitParticipantIntake,
  type IntakeStatusResult,
  type IntakeSubmissionResult,
} from "@/lib/api";

const MIN_IMAGES = 10;
const MAX_IMAGES = 20;
const MAX_FILE_MB = 8;
const AGE_BANDS = [
  { value: "", label: "Select age band" },
  { value: "18_24", label: "18-24" },
  { value: "25_34", label: "25-34" },
  { value: "35_44", label: "35-44" },
  { value: "45_54", label: "45-54" },
  { value: "55_64", label: "55-64" },
  { value: "65_plus", label: "65+" },
  { value: "prefer_not_to_say", label: "Prefer not to say" },
];

export function IntakeForm() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [ageBand, setAgeBand] = useState("");
  const [participantNote, setParticipantNote] = useState("");
  const [consentTraining, setConsentTraining] = useState(false);
  const [consentResearch, setConsentResearch] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<IntakeSubmissionResult | null>(null);
  const [status, setStatus] = useState<IntakeStatusResult | null>(null);

  useEffect(() => {
    getIntakeStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  const oversizedFiles = files.filter((file) => file.size > MAX_FILE_MB * 1024 * 1024);
  const imageCountValid = files.length >= MIN_IMAGES && files.length <= MAX_IMAGES;
  const canSubmit = imageCountValid && ageBand && consentTraining && consentResearch && oversizedFiles.length === 0;

  function handleFileChange(nextFiles: FileList | null) {
    const accepted = Array.from(nextFiles || []).filter((file) => /^image\/(jpeg|png|webp)$/.test(file.type));
    setFiles(accepted.slice(0, MAX_IMAGES));
    setResult(null);
    setError(accepted.length ? "" : "Choose JPEG, PNG, or WebP images.");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!canSubmit) {
      setError("Complete consent, age band, and upload 10-20 images under 8MB each.");
      return;
    }

    setIsSubmitting(true);
    try {
      const submission = await submitParticipantIntake({
        files,
        age_band: ageBand,
        participant_note: participantNote,
        consent_training: consentTraining,
        consent_research: consentResearch,
      });
      setResult(submission);
      setFiles([]);
      setParticipantNote("");
      setConsentTraining(false);
      setConsentResearch(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      const nextStatus = await getIntakeStatus().catch(() => null);
      setStatus(nextStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Submission failed");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="rounded-3xl border border-border bg-background p-5 shadow-sm md:p-7">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">
            Local intake
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Participant upload batch</h2>
        </div>
        <div className="rounded-2xl border border-border bg-muted/40 p-3 text-muted-foreground">
          <LockKeyhole className="h-5 w-5" />
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate className="space-y-5">
        <div>
          <label htmlFor="age-band" className="text-sm font-medium">
            Age band
          </label>
          <select
            id="age-band"
            value={ageBand}
            onChange={(event) => setAgeBand(event.target.value)}
            aria-required="true"
            className="mt-2 w-full rounded-2xl border border-border bg-card px-4 py-3 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring"
          >
            {AGE_BANDS.map((band) => (
              <option key={band.value} value={band.value}>
                {band.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="intake-images" className="text-sm font-medium">
            Outfit images
          </label>
          <label
            htmlFor="intake-images"
            className="mt-2 flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-border bg-muted/30 px-4 py-8 text-center transition-colors hover:bg-muted/50"
          >
            <UploadCloud className="h-8 w-8 text-muted-foreground" />
            <span className="mt-3 text-sm font-medium">Choose 10-20 images</span>
            <span className="mt-1 text-xs text-muted-foreground">JPEG, PNG, or WebP. Max {MAX_FILE_MB}MB each.</span>
          </label>
          <input
            ref={fileInputRef}
            id="intake-images"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            onChange={(event) => handleFileChange(event.target.files)}
            aria-describedby="image-count-hint"
            className="sr-only"
          />
          <p id="image-count-hint" className="mt-2 text-xs text-muted-foreground">
            Selected: {files.length}. This batch should represent one consenting participant.
          </p>
          {oversizedFiles.length > 0 ? (
            <p role="alert" className="mt-2 text-xs font-medium text-destructive">
              {oversizedFiles.length} file(s) exceed {MAX_FILE_MB}MB.
            </p>
          ) : null}
        </div>

        <div>
          <label htmlFor="participant-note" className="text-sm font-medium">
            Optional non-identifying note
          </label>
          <textarea
            id="participant-note"
            value={participantNote}
            onChange={(event) => setParticipantNote(event.target.value)}
            rows={3}
            maxLength={500}
            placeholder="Example: mix of casual, work, and layered outfits. Do not include names, contacts, or profile links."
            className="mt-2 w-full resize-none rounded-2xl border border-border bg-card px-4 py-3 text-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>

        <ConsentCheckbox
          id="consent-training"
          checked={consentTraining}
          onChange={setConsentTraining}
          label="I have explicit permission to submit these images for FitView style recommendation training."
        />
        <ConsentCheckbox
          id="consent-research"
          checked={consentResearch}
          onChange={setConsentResearch}
          label="I understand raw images stay restricted and must be face-redacted before annotation or training."
        />

        {error ? (
          <div role="alert" className="flex gap-2 rounded-2xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        ) : null}

        {result ? (
          <div role="status" className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-700 dark:text-emerald-300">
            <div className="flex items-center gap-2 font-medium">
              <CheckCircle2 className="h-4 w-4" />
              Submission saved
            </div>
            <p className="mt-2 text-xs">
              {result.image_count} image(s) stored for {result.person_id}. {result.next_step}
            </p>
          </div>
        ) : null}

        <button
          type="submit"
          disabled={!canSubmit || isSubmitting}
          className="w-full rounded-2xl bg-foreground px-5 py-3 text-sm font-semibold text-background transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isSubmitting ? "Saving restricted intake..." : "Save consented intake batch"}
        </button>
      </form>

      {status ? (
        <div className="mt-6 rounded-2xl border border-border bg-muted/30 p-4 text-xs text-muted-foreground">
          <p className="font-medium text-foreground">Current local intake</p>
          <p className="mt-1">People: {status.people} · Raw records: {status.records}</p>
          <p className="mt-1 break-all">Manifest: {status.raw_manifest}</p>
        </div>
      ) : null}
    </div>
  );
}

function ConsentCheckbox({
  id,
  checked,
  onChange,
  label,
}: {
  id: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <label htmlFor={id} className="flex gap-3 rounded-2xl border border-border bg-card p-4 text-sm leading-6">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-1 h-4 w-4 rounded border-border"
        aria-required="true"
      />
      <span>{label}</span>
    </label>
  );
}
