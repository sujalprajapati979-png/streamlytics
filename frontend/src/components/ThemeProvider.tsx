'use client';
import { ThemeProvider as NextThemesProvider } from 'next-themes';
import { useEffect } from 'react';
import { useTheme } from 'next-themes';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="dark" enableSystem={false}>
      <BodyClassSync />
      {children}
    </NextThemesProvider>
  );
}

/** Syncs next-themes resolved theme → body.light class for our globals.css */
function BodyClassSync() {
  const { resolvedTheme } = useTheme();
  useEffect(() => {
    if (resolvedTheme === 'light') {
      document.body.classList.add('light');
    } else {
      document.body.classList.remove('light');
    }
  }, [resolvedTheme]);
  return null;
}
