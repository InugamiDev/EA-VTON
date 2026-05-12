import type { NextConfig } from "next";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = resolve(webRoot, "../..");

const nextConfig: NextConfig = {
  turbopack: {
    // intent: keep Turbopack rooted where pnpm hoists workspace dependencies
    // status: done
    // next: remove when Next workspace root inference is stable for this monorepo
    // confidence: high
    root: workspaceRoot,
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "image.uniqlo.com",
        pathname: "/UQ/**",
      },
      {
        protocol: "https",
        hostname: "static.nike.com",
        pathname: "/a/images/**",
      },
      {
        protocol: "https",
        hostname: "images.unsplash.com",
      },
    ],
  },
};

export default nextConfig;
