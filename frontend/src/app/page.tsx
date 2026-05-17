'use client';
import { useBootstrap } from '@/hooks/useBootstrap';
import { Header } from '@/components/Header';
import { UserPanel } from '@/components/UserPanel';
import { RecommendationPanel } from '@/components/RecommendationPanel';
import { Loader } from '@/components/Loader';
import { Notifications } from '@/components/Notifications';

export default function HomePage() {
  useBootstrap();

  return (
    <>
      <Loader />
      <Notifications />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
        <Header />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-7">
          <UserPanel />
          <RecommendationPanel />
        </div>

        <footer className="mt-12 text-center text-gray-500 text-xs border-t border-white/10 pt-6">
          Streamlytics AI : by SUJAL PRAJAPATI
        </footer>
      </main>
    </>
  );
}
