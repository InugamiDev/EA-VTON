"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X, Shirt, LogIn, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMockAuth } from "@/hooks/use-mock-auth";

const NAV_LINKS = [
  { href: "/live-size", label: "Live size" },
  { href: "/catalog", label: "Catalog" },
  { href: "/try-on", label: "Try on" },
  { href: "/data", label: "Data" },
];

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();
  const { user, login, logout } = useMockAuth();

  // intent: replace the full-width bar with a compact neutral navigation pill
  // status: done
  // next: keep link count small unless a page becomes essential to top-level flow
  // blockers: none
  // confidence: high
  return (
    <header className="sticky top-0 z-50 px-3 py-3">
      <nav className="mx-auto flex h-12 max-w-4xl items-center justify-between gap-3 rounded-full border border-border bg-background/90 px-3 shadow-sm backdrop-blur-md md:px-4">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 rounded-full pr-1 text-sm font-semibold tracking-tight focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-foreground text-background">
            <Shirt className="h-4 w-4" />
          </div>
          <span>FitView</span>
        </Link>

        <div className="hidden min-w-0 items-center gap-1 md:flex">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.href}
              href={link.href}
              active={pathname === link.href}
            >
              {link.label}
            </NavLink>
          ))}
        </div>

        <div className="hidden shrink-0 items-center md:flex">
          {user ? (
            <button
              onClick={logout}
              className="flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <LogOut className="h-3.5 w-3.5" />
              Sign Out
            </button>
          ) : (
            <button
              onClick={login}
              className="flex items-center gap-1.5 rounded-full bg-foreground px-3 py-2 text-xs font-medium text-background transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <LogIn className="h-3.5 w-3.5" />
              Demo
            </button>
          )}
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex h-9 w-9 items-center justify-center rounded-full transition-colors hover:bg-accent md:hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-label="Toggle menu"
        >
          {mobileOpen ? (
            <X className="h-5 w-5" />
          ) : (
            <Menu className="h-5 w-5" />
          )}
        </button>
      </nav>

      <div
        className={cn(
          "mx-auto max-w-4xl overflow-hidden transition-all duration-200 md:hidden",
          mobileOpen ? "max-h-96 pt-2" : "max-h-0",
        )}
      >
        <div className="flex flex-col gap-1 rounded-3xl border border-border bg-background/95 p-2 shadow-sm backdrop-blur-md">
          {NAV_LINKS.map((link) => (
            <MobileNavLink
              key={link.href}
              href={link.href}
              active={pathname === link.href}
              onClick={() => setMobileOpen(false)}
            >
              {link.label}
            </MobileNavLink>
          ))}
          <div className="my-2 h-px bg-border" />
          {user ? (
            <button
              onClick={() => {
                logout();
                setMobileOpen(false);
              }}
              className="flex items-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </button>
          ) : (
            <button
              onClick={() => {
                login();
                setMobileOpen(false);
              }}
              className="flex items-center gap-2 rounded-2xl bg-foreground px-4 py-3 text-sm font-medium text-background transition-opacity hover:opacity-90"
            >
              <LogIn className="h-4 w-4" />
              Demo Login
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

function NavLink({
  href,
  children,
  active,
}: {
  href: string;
  children: React.ReactNode;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "rounded-full px-3 py-2 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        active
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </Link>
  );
}

function MobileNavLink({
  href,
  children,
  active,
  onClick,
}: {
  href: string;
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(
        "rounded-2xl px-4 py-3 text-sm font-medium transition-colors",
        active
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </Link>
  );
}
