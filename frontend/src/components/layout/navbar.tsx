"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, Shirt, User, LogIn, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMockAuth } from "@/hooks/use-mock-auth";

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, login, logout } = useMockAuth();

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2.5 font-semibold text-lg tracking-tight">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Shirt className="h-5 w-5" />
          </div>
          <span>FitView</span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden items-center gap-1 md:flex">
          <NavLink href="/catalog">Catalog</NavLink>
          <NavLink href="/closet">Closet</NavLink>
          <NavLink href="/try-on">Try On</NavLink>
          <NavLink href="/profile">Profile</NavLink>
        </div>

        <div className="hidden items-center gap-3 md:flex">
          {user ? (
            <button
              onClick={logout}
              className="flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </button>
          ) : (
            <button
              onClick={login}
              className="flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-all duration-200 hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <LogIn className="h-4 w-4" />
              Demo Login
            </button>
          )}
        </div>

        {/* Mobile toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex h-10 w-10 items-center justify-center rounded-lg transition-all duration-200 hover:bg-accent md:hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {/* Mobile menu */}
      <div
        className={cn(
          "overflow-hidden border-t border-border transition-all duration-300 md:hidden",
          mobileOpen ? "max-h-80" : "max-h-0 border-t-0"
        )}
      >
        <div className="flex flex-col gap-1 px-6 py-4">
          <MobileNavLink href="/catalog" onClick={() => setMobileOpen(false)}>
            Catalog
          </MobileNavLink>
          <MobileNavLink href="/closet" onClick={() => setMobileOpen(false)}>
            Closet
          </MobileNavLink>
          <MobileNavLink href="/try-on" onClick={() => setMobileOpen(false)}>
            Try On
          </MobileNavLink>
          <MobileNavLink href="/profile" onClick={() => setMobileOpen(false)}>
            Profile
          </MobileNavLink>
          <div className="my-2 h-px bg-border" />
          {user ? (
            <button
              onClick={() => {
                logout();
                setMobileOpen(false);
              }}
              className="flex items-center gap-2 rounded-lg px-4 py-3 text-sm font-medium text-muted-foreground transition-all duration-200 hover:bg-accent"
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
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-medium text-primary-foreground transition-all duration-200 hover:opacity-90"
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

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded-lg px-4 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
    >
      {children}
    </Link>
  );
}

function MobileNavLink({
  href,
  children,
  onClick,
}: {
  href: string;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className="rounded-lg px-4 py-3 text-sm font-medium text-muted-foreground transition-all duration-200 hover:bg-accent hover:text-foreground"
    >
      {children}
    </Link>
  );
}
