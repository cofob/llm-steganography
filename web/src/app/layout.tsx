import { ThemeScript } from "@cofob/design-system-react";
import type { ReactNode } from "react";

import "@cofob/design-system-css/index.css";
import "./globals.css";

import { Providers } from "@/components/providers";

export const metadata = {
  title: "LLM Steganography Lab",
  description: "Encode, decode, chat, and inspect a keyed LLM token channel.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><ThemeScript /></head>
      <body><Providers>{children}</Providers></body>
    </html>
  );
}
