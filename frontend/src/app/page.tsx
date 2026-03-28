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

export default function HomePage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden py-24 md:py-32">
        <div className="absolute inset-0 bg-gradient-to-b from-accent/50 to-background" />
        <div className="relative mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm font-medium shadow-sm">
              <Sparkles className="h-4 w-4 text-amber-500" />
              AI-Powered Virtual Try-On
            </div>
            <h1 className="text-balance text-4xl font-bold tracking-tight md:text-6xl">
              See how clothes look on you{" "}
              <span className="text-muted-foreground">before you buy</span>
            </h1>
            <p className="mx-auto mt-6 max-w-xl text-lg text-muted-foreground">
              Upload a photo, pick a garment from our catalog, and get an
              AI-generated preview with personalized size recommendations.
            </p>
            <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
              <Link
                href="/try-on"
                className="flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
              >
                Try It Now
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/catalog"
                className="flex items-center gap-2 rounded-xl border border-border bg-card px-8 py-4 text-base font-semibold shadow-sm transition-all duration-200 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
              >
                Browse Catalog
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold tracking-tight">How it works</h2>
            <p className="mt-3 text-muted-foreground">
              Three simple steps to your virtual fitting room
            </p>
          </div>
          <div className="mt-16 grid gap-8 md:grid-cols-3">
            <StepCard
              step={1}
              icon={<Camera className="h-6 w-6" />}
              title="Take or upload a photo"
              description="Use your camera or upload an existing photo. Our quality checker guides you to get the best result."
            />
            <StepCard
              step={2}
              icon={<Shirt className="h-6 w-6" />}
              title="Pick a garment"
              description="Browse our catalog of UNIQLO essentials. Select the item you want to try on virtually."
            />
            <StepCard
              step={3}
              icon={<Eye className="h-6 w-6" />}
              title="See the result"
              description="Get an AI-generated preview of how the garment looks on you, with a size recommendation."
            />
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-border bg-muted/30 py-20">
        <div className="mx-auto max-w-7xl px-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold tracking-tight">
              Built for trust
            </h2>
            <p className="mt-3 text-muted-foreground">
              Honest AI that tells you when it&apos;s uncertain
            </p>
          </div>
          <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={<Sparkles className="h-5 w-5" />}
              title="Confidence labels"
              description="Every result shows a confidence score — we tell you when we're unsure so you can decide."
            />
            <FeatureCard
              icon={<Ruler className="h-5 w-5" />}
              title="Smart sizing"
              description="Size recommendations from your measurements + garment size charts. No magic, just math."
            />
            <FeatureCard
              icon={<Shield className="h-5 w-5" />}
              title="Privacy first"
              description="Your photos are processed and deleted. No facial recognition, no data selling."
            />
            <FeatureCard
              icon={<Zap className="h-5 w-5" />}
              title="Fast results"
              description="Get your virtual try-on in under 60 seconds, with a before/after comparison view."
            />
            <FeatureCard
              icon={<Camera className="h-5 w-5" />}
              title="Photo guidance"
              description="Client-side quality checks help you take a photo that produces the best results."
            />
            <FeatureCard
              icon={<Shirt className="h-5 w-5" />}
              title="Real brands"
              description="Demo catalog with UNIQLO garments — realistic products, real size charts."
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-6">
          <div className="rounded-2xl border border-border bg-card p-12 text-center shadow-sm md:p-16">
            <h2 className="text-3xl font-bold tracking-tight">
              Ready to try it?
            </h2>
            <p className="mx-auto mt-4 max-w-md text-muted-foreground">
              No account required. Take a photo, pick a garment, and see the
              result in seconds.
            </p>
            <Link
              href="/try-on"
              className="mt-8 inline-flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
            >
              Start Virtual Try-On
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
    <div className="group relative rounded-2xl border border-border bg-card p-8 shadow-sm transition-all duration-200 hover:shadow-md">
      <div className="mb-5 flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-all duration-200 group-hover:scale-105">
          {icon}
        </div>
        <span className="text-sm font-semibold text-muted-foreground">
          Step {step}
        </span>
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
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
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm transition-all duration-200 hover:shadow-md">
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-accent">
        {icon}
      </div>
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  );
}
