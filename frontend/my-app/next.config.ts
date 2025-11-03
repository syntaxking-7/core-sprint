import type { NextConfig } from "next";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND}/:path*`
      }
    ]
  }
};

export default nextConfig;
