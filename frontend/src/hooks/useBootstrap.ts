'use client';
import { useEffect } from 'react';
import { useAppStore, useUIStore } from '@/store';
import type { UserDatabase, CatalogItem } from '@/lib/types';

export function useBootstrap() {
  const { setUsers, setCatalog, setSelectedUserId, setCurrentUser, setRecommendations } = useAppStore();
  const { showNotification, setLoading } = useUIStore();

  useEffect(() => {
    async function load() {
      try {
        const [usersRes, catalogRes] = await Promise.all([
          fetch('/data/users.json'),
          fetch('/data/catalog.json'),
        ]);
        if (!usersRes.ok || !catalogRes.ok) throw new Error('data files missing');
        const users: UserDatabase = await usersRes.json();
        const catalog: CatalogItem[] = await catalogRes.json();
        setUsers(users);
        setCatalog(catalog);

        // Auto-select first user
        const firstId = Object.keys(users)[0];
        if (firstId) setSelectedUserId(firstId);
      } catch {
        showNotification('Using demo data', 'info');
        const demoUsers: UserDatabase = {
          U1: {
            name: 'Demo User',
            cluster: 0,
            age: 30,
            preferred_genre: 'Comedy',
            features: { days_since_login: 1, engagement_score: 5 },
          },
        };
        const demoCatalog: CatalogItem[] = [
          {
            title: 'Stranger Things',
            genre: 'Sci-Fi',
            rating: 'TV-14',
            duration: 50,
            trending: 98,
            ai_description: 'A group of kids encounter supernatural forces.',
            year: 2016,
            type: 'Series',
          },
        ];
        setUsers(demoUsers);
        setCatalog(demoCatalog);
        setSelectedUserId('U1');
      }
    }
    load();
  }, [setUsers, setCatalog, setSelectedUserId, setCurrentUser, setRecommendations, showNotification, setLoading]);
}
