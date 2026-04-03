/** @type {import('next').NextConfig} */
const isExport = process.env.NEXT_EXPORT === 'true';

const nextConfig = {
  reactStrictMode: true,
  // Static export mode for Docker/HF Spaces (NEXT_EXPORT=true disables rewrites)
  ...(isExport ? { output: 'export' } : {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/:path*',
        },
      ];
    },
  }),
};

module.exports = nextConfig;
