"use client";

import {
  ThemeProvider,
  ToastProvider,
  ToastViewport,
} from "@cofob/design-system-react";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider defaultPreference="system">
      <ToastProvider>
        {children}
        <ToastViewport position="bottom-right" />
      </ToastProvider>
    </ThemeProvider>
  );
}
