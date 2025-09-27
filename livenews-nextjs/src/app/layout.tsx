import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from '@/components/providers/AuthProvider';
import { LocationProvider } from '@/components/providers/LocationProvider';
import { Toaster } from '@/components/ui/toaster';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "வெளிச்சம் - Personalized News Intelligence",
  description: "Get personalized, location-based news with AI-powered insights and summaries",
  keywords: ["news", "AI", "personalized", "location", "current events"],
  openGraph: {
    title: "வெளிச்சம் - Personalized News Intelligence",
    description: "Get personalized, location-based news with AI-powered insights and summaries",
    type: "website",
    url: "https://velicham.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "வெளிச்சம் - Personalized News Intelligence",
    description: "Get personalized, location-based news with AI-powered insights and summaries",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased bg-gradient-to-br from-slate-50 to-blue-50 min-h-screen`}>
        <AuthProvider>
          <LocationProvider>
            {children}
            <Toaster />
          </LocationProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
