'use client';
import { useUIStore } from '@/store';

export function Loader() {
  const loading = useUIStore((s) => s.loading);
  if (!loading) return null;
  return (
    <div className="fixed inset-0 bg-black/70 flex justify-center items-center z-[999]">
      <div className="spinner" />
    </div>
  );
}
