import { create } from 'zustand';
import type { CatalogItem, User, UserDatabase } from '@/lib/types';
import type { MovieSummary } from '@/lib/api';

// ── UI state ──────────────────────────────────────────────────────────────────

interface Notification {
  id: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

interface UIState {
  loading: boolean;
  notifications: Notification[];
  modalTitle: string | null;
  setLoading: (v: boolean) => void;
  showNotification: (msg: string, type?: Notification['type']) => void;
  openModal: (title: string) => void;
  closeModal: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  loading: false,
  notifications: [],
  modalTitle: null,
  setLoading: (v) => set({ loading: v }),
  showNotification: (message, type = 'info') => {
    const id = Math.random().toString(36).slice(2);
    set((s) => ({ notifications: [...s.notifications, { id, message, type }] }));
    setTimeout(
      () => set((s) => ({ notifications: s.notifications.filter((n) => n.id !== id) })),
      3000
    );
  },
  openModal: (title) => set({ modalTitle: title }),
  closeModal: () => set({ modalTitle: null }),
}));

// ── App state ─────────────────────────────────────────────────────────────────

interface AppState {
  // data
  users: UserDatabase;
  catalog: CatalogItem[];
  selectedUserId: string;
  currentUser: User | null;
  recommendations: MovieSummary[];
  cluster: number;
  dominantGenre: string;
  mood: string;
  timeFilter: string;

  // setters
  setUsers: (u: UserDatabase) => void;
  setCatalog: (c: CatalogItem[]) => void;
  setSelectedUserId: (id: string) => void;
  setCurrentUser: (u: User | null) => void;
  setRecommendations: (r: MovieSummary[], cluster: number, genre: string) => void;
  setMood: (m: string) => void;
  setTimeFilter: (t: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  users: {},
  catalog: [],
  selectedUserId: '',
  currentUser: null,
  recommendations: [],
  cluster: 0,
  dominantGenre: '',
  mood: 'all',
  timeFilter: 'any',

  setUsers: (users) => set({ users }),
  setCatalog: (catalog) => set({ catalog }),
  setSelectedUserId: (selectedUserId) => set({ selectedUserId }),
  setCurrentUser: (currentUser) => set({ currentUser }),
  setRecommendations: (recommendations, cluster, dominantGenre) =>
    set({ recommendations, cluster, dominantGenre }),
  setMood: (mood) => set({ mood }),
  setTimeFilter: (timeFilter) => set({ timeFilter }),
}));
