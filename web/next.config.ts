import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Transpile ESM-only packages
  transpilePackages: ["maplibre-gl", "react-map-gl"],

  // Empty turbopack config opts into Turbopack (Next.js 16 default)
  turbopack: {},
};

export default nextConfig;
