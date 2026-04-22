"use client";

import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Shirt,
  BarChart3,
  CheckCircle,
  Clock,
  ArrowRight,
} from "lucide-react";

const stats = [
  {
    label: "Total Garments",
    value: "6",
    icon: Shirt,
    color: "text-blue-600 bg-blue-100",
  },
  {
    label: "Total Try-Ons",
    value: "142",
    icon: BarChart3,
    color: "text-purple-600 bg-purple-100",
  },
  {
    label: "Success Rate",
    value: "94.3%",
    icon: CheckCircle,
    color: "text-green-600 bg-green-100",
  },
  {
    label: "Avg Processing Time",
    value: "28s",
    icon: Clock,
    color: "text-orange-600 bg-orange-100",
  },
];

const recentJobs = [
  { id: "job-0041", garment: "Classic White Tee", status: "completed", time: "2m ago" },
  { id: "job-0040", garment: "Denim Jacket", status: "completed", time: "18m ago" },
  { id: "job-0039", garment: "Oversized Hoodie", status: "processing", time: "22m ago" },
  { id: "job-0038", garment: "Oxford Button-Down", status: "completed", time: "1h ago" },
  { id: "job-0037", garment: "Bomber Jacket", status: "failed", time: "2h ago" },
];

function statusBadge(status: string) {
  const map: Record<string, string> = {
    completed: "bg-green-100 text-green-700",
    processing: "bg-yellow-100 text-yellow-700",
    failed: "bg-red-100 text-red-700",
  };
  return (
    <span
      className={cn(
        "rounded-full px-2 py-1 text-xs font-medium",
        map[status] ?? "bg-muted text-muted-foreground"
      )}
    >
      {status}
    </span>
  );
}

export default function AdminPage() {
  return (
    <section className="py-16">
      <div className="mx-auto max-w-5xl px-6">
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Admin Dashboard</h1>
            <p className="mt-1 text-base text-muted-foreground">
              Overview of the FitView virtual try-on system.
            </p>
          </div>
          <div className="flex gap-3">
            <Link
              href="/admin/garments"
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-border px-6 py-4 text-base font-medium text-foreground",
                "transition-all duration-200 hover:bg-muted",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "motion-reduce:transition-none"
              )}
            >
              Garments
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/admin/size-charts"
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-border px-6 py-4 text-base font-medium text-foreground",
                "transition-all duration-200 hover:bg-muted",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "motion-reduce:transition-none"
              )}
            >
              Size Charts
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="mb-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="rounded-xl border border-border bg-card p-6 shadow-sm"
            >
              <div className="mb-3 flex items-center gap-3">
                <div
                  className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-lg",
                    stat.color
                  )}
                >
                  <stat.icon className="h-5 w-5" />
                </div>
                <span className="text-base text-muted-foreground">{stat.label}</span>
              </div>
              <p className="text-3xl font-bold text-foreground">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Recent Jobs Table */}
        <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
          <h2 className="mb-6 text-xl font-semibold text-foreground">
            Recent Try-On Jobs
          </h2>

          {/* Desktop table */}
          <div className="hidden sm:block">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border text-base text-muted-foreground">
                  <th className="pb-3 font-medium">Job ID</th>
                  <th className="pb-3 font-medium">Garment</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {recentJobs.map((job) => (
                  <tr key={job.id}>
                    <td className="py-4 font-mono text-base text-foreground">
                      {job.id}
                    </td>
                    <td className="py-4 text-base text-foreground">{job.garment}</td>
                    <td className="py-4">{statusBadge(job.status)}</td>
                    <td className="py-4 text-right text-base text-muted-foreground">
                      {job.time}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile card list */}
          <div className="space-y-3 sm:hidden">
            {recentJobs.map((job) => (
              <div
                key={job.id}
                className="rounded-lg border border-border bg-muted/50 p-6"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-base text-foreground">{job.id}</span>
                  {statusBadge(job.status)}
                </div>
                <p className="text-base text-foreground">{job.garment}</p>
                <p className="mt-1 text-sm text-muted-foreground">{job.time}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
