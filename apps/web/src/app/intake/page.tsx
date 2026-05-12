import type { Metadata } from "next";
import { IntakeForm } from "./intake-form";

export const metadata: Metadata = {
  title: "Dataset Intake — FitView",
  description: "Consented participant image intake for FitView style recommendation research.",
};

export default function IntakePage() {
  return (
    <section className="min-h-[calc(100vh-5rem)] bg-[radial-gradient(circle_at_top_left,hsl(var(--muted))_0,transparent_32rem)] py-10">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 md:grid-cols-[0.9fr_1.1fr] md:px-6">
        <div className="rounded-3xl border border-border bg-card/80 p-7 shadow-sm backdrop-blur md:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">
            First-party collection
          </p>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
            Build the dataset honestly.
          </h1>
          <p className="mt-5 text-base leading-7 text-muted-foreground">
            Public ReID and fashion datasets do not satisfy the required women-focused 10-image-per-person target. This intake path collects only explicit participant submissions into restricted local storage, then waits for face redaction before annotation or training.
          </p>

          <div className="mt-8 grid gap-3 text-sm text-muted-foreground">
            <PolicyCard title="Consent first" body="The upload requires explicit training and internal research consent before files are accepted." />
            <PolicyCard title="Restricted raw storage" body="Raw images are not served from /uploads and are ignored by git under research/datasets/restricted." />
            <PolicyCard title="No training yet" body="Submitted rows remain raw intake records until face boxes are labeled and redaction creates a training manifest." />
          </div>
        </div>

        <IntakeForm />
      </div>
    </section>
  );
}

function PolicyCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-border bg-background/70 p-4">
      <h2 className="font-medium text-foreground">{title}</h2>
      <p className="mt-1 leading-6">{body}</p>
    </div>
  );
}
