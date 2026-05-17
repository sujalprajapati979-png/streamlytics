'use client';
import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';

export function Header() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  return (
    <div className="flex justify-between items-start mb-5">
      <div className="text-center flex-1">
        <div className="inline-flex items-center gap-3 bg-white/5 backdrop-blur-sm rounded-full px-5 py-2 border border-white/10 mb-4">
          <span className="text-xl">🎬</span>
          <span className="text-sm font-medium tracking-wide">
            STREAMLytics • AI POWERED RECOMMENDATION SYSTEM
          </span>
        </div>
        <div className="flex items-center justify-center gap-4 flex-wrap mb-5">
          <div className="glitch-logo">🎬</div>
          <h1 className="glitch-title">Netflix Intelligence Hub</h1>
        </div>
        <p className="text-gray-300 max-w-2xl mx-auto mt-2 text-lg">
          ...get your personalised recommendations for netflix...
        </p>
      </div>
      <button
        id="themeToggle"
        onClick={toggleTheme}
        className="ml-4 p-3 glass-card rounded-full cursor-pointer"
        aria-label="Toggle theme"
      >
        <i className={mounted ? `fas ${theme === 'dark' ? 'fa-moon' : 'fa-sun'} text-xl` : 'fas fa-moon text-xl'} suppressHydrationWarning />
      </button>
    </div>
  );
}
