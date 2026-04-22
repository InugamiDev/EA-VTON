import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy — FitView",
  description: "How FitView handles your data, photos, and personal information.",
};

export default function PrivacyPage() {
  return (
    <section className="py-16">
      <div className="mx-auto max-w-3xl px-6">
        <h1 className="mb-4 text-4xl font-bold text-foreground">Privacy Policy</h1>
        <p className="mb-12 text-base text-muted-foreground">
          Last updated: March 15, 2026
        </p>

        <div className="space-y-12">
          {/* What We Collect */}
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-foreground">
              What We Collect
            </h2>
            <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
              <p>
                FitView collects only the minimum data necessary to provide the
                virtual try-on experience:
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-foreground">Photos you upload</strong> — used
                  solely for generating try-on results. We do not perform facial
                  recognition, biometric analysis, or identity matching of any kind.
                </li>
                <li>
                  <strong className="text-foreground">Body measurements</strong> — height,
                  weight, and fit preference, provided voluntarily to improve size
                  recommendations.
                </li>
                <li>
                  <strong className="text-foreground">Account information</strong> — email
                  address and display name for authentication purposes.
                </li>
                <li>
                  <strong className="text-foreground">Usage data</strong> — anonymous,
                  aggregated statistics such as which garments are tried on most
                  frequently. This data cannot be linked back to individual users.
                </li>
              </ul>
            </div>
          </div>

          {/* How We Use It */}
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-foreground">
              How We Use It
            </h2>
            <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
              <p>Your data is used exclusively for the following purposes:</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-foreground">Generating try-on results</strong>{" "}
                  — your uploaded photo is sent to our processing server, combined with
                  the selected garment image, and returned as a composite result.
                </li>
                <li>
                  <strong className="text-foreground">Size recommendations</strong> — your
                  measurements are compared against garment size charts to suggest the
                  best fit.
                </li>
                <li>
                  <strong className="text-foreground">Service improvement</strong> —
                  anonymous, aggregated statistics help us improve model accuracy and
                  garment coverage.
                </li>
              </ul>
              <p>
                We never sell, share, or license your personal data or photos to third
                parties for advertising, training external AI models, or any purpose
                unrelated to providing this service.
              </p>
            </div>
          </div>

          {/* Storage & Retention */}
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-foreground">
              Storage & Retention
            </h2>
            <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
              <p>
                We follow a minimal-retention approach to protect your privacy:
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-foreground">Uploaded photos</strong> are
                  processed server-side and are not permanently stored by default.
                  Photos are held in temporary encrypted storage only for the duration
                  of processing (typically under 60 seconds) and are then deleted.
                </li>
                <li>
                  <strong className="text-foreground">Try-on results</strong> are cached
                  temporarily (up to 24 hours) so you can revisit them without
                  reprocessing. After that, they are automatically purged.
                </li>
                <li>
                  <strong className="text-foreground">Measurements and preferences</strong>{" "}
                  are stored in your account profile and persist until you choose to
                  delete them.
                </li>
                <li>
                  <strong className="text-foreground">All data is encrypted</strong> at
                  rest and in transit using industry-standard TLS 1.3 and AES-256
                  encryption.
                </li>
              </ul>
            </div>
          </div>

          {/* Your Rights */}
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-foreground">
              Your Rights
            </h2>
            <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
              <p>You have full control over your data at all times:</p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-foreground">Access</strong> — view all data we
                  hold about you from your profile page.
                </li>
                <li>
                  <strong className="text-foreground">Correction</strong> — update your
                  measurements and preferences at any time.
                </li>
                <li>
                  <strong className="text-foreground">Deletion</strong> — permanently
                  delete all your data, including measurements, try-on history, and
                  account information, with a single action from your profile page.
                </li>
                <li>
                  <strong className="text-foreground">Portability</strong> — request an
                  export of your data in a machine-readable format.
                </li>
                <li>
                  <strong className="text-foreground">Opt-out</strong> — you may use the
                  service without providing measurements. Size recommendations will be
                  less accurate but the try-on feature remains fully functional.
                </li>
              </ul>
              <p>
                We honor all data deletion requests within 48 hours. Once deleted,
                data cannot be recovered.
              </p>
            </div>
          </div>

          {/* Contact */}
          <div>
            <h2 className="mb-4 text-2xl font-semibold text-foreground">Contact</h2>
            <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
              <p>
                If you have questions about this privacy policy, want to exercise
                your data rights, or need to report a concern, please reach out:
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <strong className="text-foreground">Email</strong>:{" "}
                  <a
                    href="mailto:privacy@fitview.app"
                    className="text-primary underline underline-offset-4 transition-all duration-200 hover:opacity-80"
                  >
                    privacy@fitview.app
                  </a>
                </li>
                <li>
                  <strong className="text-foreground">Response time</strong>: We aim to
                  respond to all privacy inquiries within 2 business days.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
