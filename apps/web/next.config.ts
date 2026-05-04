import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Keep Turbopack scoped to the web app package instead of inferring the monorepo root.
    root: process.cwd(),
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
