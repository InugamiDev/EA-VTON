import Link from "next/link";
import { Shirt } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border bg-muted/50">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="flex flex-col items-center gap-6 md:flex-row md:justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Shirt className="h-4 w-4" />
            </div>
            <span className="font-semibold">FitView</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <Link
              href="/catalog"
              className="transition-colors duration-200 hover:text-foreground"
            >
              Catalog
            </Link>
            <Link
              href="/try-on"
              className="transition-colors duration-200 hover:text-foreground"
            >
              Try On
            </Link>
            <Link
              href="/privacy"
              className="transition-colors duration-200 hover:text-foreground"
            >
              Privacy
            </Link>
          </div>
          <p className="text-sm text-muted-foreground">
            Research Demo &middot; Not a real store
          </p>
        </div>
      </div>
    </footer>
  );
}
