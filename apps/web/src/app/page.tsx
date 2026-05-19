import Link from "next/link";
import {
  Camera,
  Shirt,
  Sparkles,
  Ruler,
  ArrowRight,
  Shield,
  Zap,
  Eye,
} from "lucide-react";

// intent: marketing landing — calm monochrome, rounded-full CTAs, DESIGN.md tokens
// status: done
// next: A/B with a journey-style "try the demo right here" hero
// confidence: high

export default function HomePage() {
  return (
    <>
      {/* Hero */}
      <section className="relative py-20 md:py-28">
        <div className="mx-auto max-w-5xl px-6">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs font-medium">
              <Sparkles className="h-3.5 w-3.5" />
              FitView — size & style recommendations
            </div>
            <h1 className="text-balance text-4xl font-bold tracking-tight md:text-6xl">
              Know your size.{" "}
              <span className="text-muted-foreground">See your style.</span>
            </h1>
            <p className="mx-auto mt-6 max-w-xl text-balance text-base text-muted-foreground md:text-lg">
              A camera-only size and style recommender for upper-body tops.
              Photos are processed locally and faces are blurred before any
              label or embedding is computed.
            </p>
            <div className="mt-10 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
              <Link
                href="/live-size"
                className="inline-flex h-12 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <Camera className="h-4 w-4" />
                Start the guided flow
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/catalog"
                className="inline-flex h-12 items-center gap-2 rounded-full border border-border bg-card px-6 text-sm font-semibold transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Browse the catalog
              </Link>
            </div>
            <p className="mt-4 text-[11px] text-muted-foreground">
              No account required. Want the live camera + diagnostics?{" "}
              <Link
                href="/live-size/advanced"
                className="underline underline-offset-4 hover:text-foreground"
              >
                Advanced mode
              </Link>
              .
            </p>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-16 md:py-20">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              How it works
            </span>
            <h2 className="mt-2 text-2xl font-bold tracking-tight md:text-3xl">
              Four steps, one journey
            </h2>
          </div>
          <div className="mt-12 grid gap-6 md:grid-cols-4">
            <StepCard
              step={1}
              icon={<Camera className="h-5 w-5" />}
              title="Capture"
              description="Upload a full-body photo or use the live camera."
            />
            <StepCard
              step={2}
              icon={<Ruler className="h-5 w-5" />}
              title="Size"
              description="We predict a size band from your body proportions + height."
            />
            <StepCard
              step={3}
              icon={<Shirt className="h-5 w-5" />}
              title="Style"
              description="Body-shape-aware ranking from our 208k-item catalog."
            />
            <StepCard
              step={4}
              icon={<Sparkles className="h-5 w-5" />}
              title="Personalize"
              description="Pick favorites; we re-rank to your taste in seconds."
            />
          </div>
        </div>
      </section>

      {/* Why */}
      <section className="border-t border-border bg-muted/30 py-16 md:py-20">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Why FitView
            </span>
            <h2 className="mt-2 text-2xl font-bold tracking-tight md:text-3xl">
              Built for trust
            </h2>
          </div>
          <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={<Shield className="h-4 w-4" />}
              title="Privacy by default"
              description="Faces are mesh-blurred before any embedding is computed. No facial biometrics persist."
            />
            <FeatureCard
              icon={<Ruler className="h-4 w-4" />}
              title="Anthropometry-aware"
              description="Size model tuned for Vietnamese female body proportions, not just US averages."
            />
            <FeatureCard
              icon={<Sparkles className="h-4 w-4" />}
              title="Explainable ranking"
              description="Every recommended item shows its fit / flatter / match score breakdown."
            />
            <FeatureCard
              icon={<Eye className="h-4 w-4" />}
              title="Honest confidence"
              description="Every prediction returns a confidence score — we tell you when we're unsure."
            />
            <FeatureCard
              icon={<Zap className="h-4 w-4" />}
              title="Cold-start in 3 taps"
              description="Pick 3 favorites from a stratified seed grid; recommendations re-rank to your taste."
            />
            <FeatureCard
              icon={<Camera className="h-4 w-4" />}
              title="Open dataset"
              description="208,069 upper-body items across 6 sources, with derived labels under CC BY 4.0."
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 md:py-20">
        <div className="mx-auto max-w-3xl px-6">
          <div className="rounded-3xl border border-border bg-card p-10 text-center shadow-sm md:p-14">
            <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
              Try it now
            </h2>
            <p className="mx-auto mt-3 max-w-md text-sm text-muted-foreground">
              No account. No upload to cloud. Faces blurred client-side.
            </p>
            <Link
              href="/live-size"
              className="mt-8 inline-flex h-12 items-center gap-2 rounded-full bg-foreground px-6 text-sm font-semibold text-background transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Start the guided flow
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

function StepCard({
  step,
  icon,
  title,
  description,
}: {
  step: number;
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-foreground text-background">
          {icon}
        </div>
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Step {step}
        </span>
      </div>
      <h3 className="mt-4 text-base font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-foreground">
        {icon}
      </div>
      <h3 className="mt-4 text-sm font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  );
}
