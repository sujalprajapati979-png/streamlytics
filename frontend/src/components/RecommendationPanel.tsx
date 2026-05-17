'use client';
import { useState, useEffect, useCallback, useRef } from 'react';
import { useAppStore, useUIStore } from '@/store';
import type { MovieSummary } from '@/lib/api';
import { searchMovies } from '@/lib/api';
import { MovieModal } from './MovieModal';

export function RecommendationPanel() {
  const { recommendations, catalog } = useAppStore();
  const [view, setView] = useState<'list' | 'swipe'>('list');
  const [swipeIndex, setSwipeIndex] = useState(0);
  const [modalTitle, setModalTitle] = useState<string | null>(null);

  // Trending — top 6 by trending score from catalog
  const trending = [...catalog]
    .sort((a, b) => (b.trending ?? 0) - (a.trending ?? 0))
    .slice(0, 6);

  const handleSwipeLeft = () => {
    if (swipeIndex < recommendations.length) setSwipeIndex((i) => i + 1);
  };
  const handleSwipeRight = () => {
    if (swipeIndex < recommendations.length) setSwipeIndex((i) => i + 1);
  };

  return (
    <>
      <div className="lg:col-span-2 space-y-6">
        {/* ── Recommendations ── */}
        <div className="glass-card p-6">
          <div className="flex flex-wrap justify-between items-center border-b border-white/10 pb-4 mb-5">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <span className="text-xl">🎥</span>
              </div>
              <h2 className="text-xl font-semibold">Recommended for you</h2>
            </div>
            <div className="flex gap-2">
              <button
                id="showListBtn"
                onClick={() => setView('list')}
                className={`px-3 py-1 rounded-lg transition ${view === 'list' ? 'bg-red-600/30' : 'bg-white/10 hover:bg-white/20'}`}
              >
                📋 your List
              </button>
              <button
                id="showSwipeBtn"
                onClick={() => { setView('swipe'); setSwipeIndex(0); }}
                className={`px-3 py-1 rounded-lg transition ${view === 'swipe' ? 'bg-red-600/30' : 'bg-white/10 hover:bg-white/20'}`}
              >
                🔄 like/dislike
              </button>
            </div>
          </div>

          {view === 'list' ? (
            <RecommendationList recs={recommendations} onSelect={setModalTitle} />
          ) : (
            <SwipeView
              items={recommendations}
              index={swipeIndex}
              onLeft={handleSwipeLeft}
              onRight={handleSwipeRight}
            />
          )}
        </div>

        {/* ── Trending ── */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <i className="fas fa-chart-line text-red-500" />
            <h3 className="font-semibold">🔥 Top Trendings</h3>
            <span className="trending-badge text-xs px-2 py-0.5 rounded-full text-white">HOT</span>
          </div>
          <div id="trendingList" className="flex gap-3 overflow-x-auto pb-2">
            {trending.map((t) => (
              <div
                key={t.title}
                className="glass-card p-2 text-center min-w-[110px] cursor-pointer flex-shrink-0"
                onClick={() => setModalTitle(t.title)}
              >
                <div className="font-bold text-sm">{t.title}</div>
                <div className="text-xs">🔥 {t.trending}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Bookmarks + History ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Bookmarks />
          <WatchHistory />
        </div>

        {/* ── Content Explorer ── */}
        <ContentExplorer onSelect={setModalTitle} />

        {/* ── Community Insights ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-card p-5">
            <h3 className="font-semibold flex items-center gap-2">
              <i className="fas fa-users" /> Community Insights
            </h3>
            <div className="space-y-2 text-sm mt-3">
              <p>👥 2,847 active viewers</p>
              <p>⭐ Most loved: &quot;Arcane&quot; (9.1/10)</p>
              <p>💬 &quot;Best thriller this season&quot; trending</p>
              <p>📈 +52% watch time this week</p>
            </div>
          </div>
        </div>
      </div>

      {modalTitle && (
        <MovieModal title={modalTitle} onClose={() => setModalTitle(null)} />
      )}
    </>
  );
}

// ── Recommendation List ───────────────────────────────────────────────────────

function RecommendationList({
  recs,
  onSelect,
}: {
  recs: MovieSummary[];
  onSelect: (title: string) => void;
}) {
  const { showNotification } = useUIStore();
  // Safe localStorage read — empty on server, populated after mount
  const [bookmarks, setBookmarks] = useState<string[]>([]);
  useEffect(() => {
    setBookmarks(JSON.parse(localStorage.getItem('netflix_bookmarks') || '[]'));
  }, []);

  const toggleBookmark = (e: React.MouseEvent, title: string) => {
    e.stopPropagation();
    const next = bookmarks.includes(title)
      ? bookmarks.filter((b) => b !== title)
      : [...bookmarks, title];
    setBookmarks(next);
    localStorage.setItem('netflix_bookmarks', JSON.stringify(next));
    showNotification(`${title} ${next.includes(title) ? 'bookmarked' : 'removed'}`, 'success');
    window.dispatchEvent(new CustomEvent('bookmarks-updated'));
  };

  if (!recs.length) {
    return (
      <div className="text-center py-8 text-gray-400">
        Select a user to get recommendations
      </div>
    );
  }

  return (
    <div id="recommendationList" className="space-y-3">
      {recs.map((rec) => (
        <div
          key={rec.title}
          className="rec-card glass-card p-4 flex justify-between items-center"
          onClick={() => onSelect(rec.title)}
        >
          <div className="flex-1 mr-3">
            <div className="font-bold">{rec.title}</div>
            <div className="text-xs">{rec.listed_in} • {rec.rating}</div>
            <div className="text-xs text-gray-400">{rec.description?.substring(0, 60)}...</div>
          </div>
          <button
            onClick={(e) => toggleBookmark(e, rec.title)}
            aria-label="Bookmark"
            className="text-lg flex-shrink-0"
          >
            <i className={`${bookmarks.includes(rec.title) ? 'fas' : 'far'} fa-bookmark`} />
          </button>
        </div>
      ))}
    </div>
  );
}

// ── Swipe View ────────────────────────────────────────────────────────────────

function SwipeView({
  items,
  index,
  onLeft,
  onRight,
}: {
  items: MovieSummary[];
  index: number;
  onLeft: () => void;
  onRight: () => void;
}) {
  if (!items.length) {
    return <div className="text-center py-12 text-gray-400">Select a user first</div>;
  }
  if (index >= items.length) {
    return <div className="text-center py-12 text-gray-400">✨ No more ✨</div>;
  }
  const item = items[index];
  return (
    <div id="swipeView">
      <div id="swipeStack" className="relative w-full max-w-md mx-auto">
        <div className="swipe-card glass-card p-6 text-center">
          <h3 className="font-bold text-xl">{item.title}</h3>
          <p>{item.listed_in} • {item.rating}</p>
          <p className="text-sm mt-3">{item.description?.substring(0, 100)}...</p>
        </div>
      </div>
      <div className="flex justify-center gap-6 mt-6">
        <button
          id="swipeLeftBtn"
          onClick={onLeft}
          className="w-14 h-14 rounded-full bg-red-600/30 hover:bg-red-600 transition text-2xl"
        >✗</button>
        <button
          id="swipeRightBtn"
          onClick={onRight}
          className="w-14 h-14 rounded-full bg-emerald-600/30 hover:bg-emerald-600 transition text-2xl"
        >✓</button>
      </div>
    </div>
  );
}

// ── Bookmarks ─────────────────────────────────────────────────────────────────

function Bookmarks() {
  const { showNotification } = useUIStore();
  // Start empty; populate from localStorage after hydration
  const [bookmarks, setBookmarks] = useState<string[]>([]);

  useEffect(() => {
    // Initial read
    setBookmarks(JSON.parse(localStorage.getItem('netflix_bookmarks') || '[]'));

    // Keep in sync when RecommendationList toggles bookmarks
    const onUpdate = () => {
      setBookmarks(JSON.parse(localStorage.getItem('netflix_bookmarks') || '[]'));
    };
    window.addEventListener('bookmarks-updated', onUpdate);
    return () => window.removeEventListener('bookmarks-updated', onUpdate);
  }, []);

  const remove = (title: string) => {
    const next = bookmarks.filter((b) => b !== title);
    setBookmarks(next);
    localStorage.setItem('netflix_bookmarks', JSON.stringify(next));
    showNotification(`Removed ${title}`);
  };

  return (
    <div className="glass-card p-5">
      <h3 className="font-semibold flex items-center gap-2 mb-3">
        <i className="fas fa-bookmark text-red-400" /> My Bookmarks
      </h3>
      <div id="bookmarksList" className="text-sm space-y-1">
        {bookmarks.length ? (
          bookmarks.map((b) => (
            <div key={b} className="flex justify-between text-sm">
              <span>📌 {b}</span>
              <button onClick={() => remove(b)}>
                <i className="fas fa-times-circle text-red-400" />
              </button>
            </div>
          ))
        ) : (
          <div className="text-gray-400 text-xs">No bookmarks</div>
        )}
      </div>
    </div>
  );
}

// ── Watch History ─────────────────────────────────────────────────────────────

function WatchHistory() {
  const [history, setHistory] = useState<string[]>([]);
  const [ongoing, setOngoing] = useState<{ title: string; progress: number }[]>([]);

  useEffect(() => {
    setHistory(JSON.parse(localStorage.getItem('netflix_history') || '[]'));
    setOngoing(JSON.parse(localStorage.getItem('netflix_ongoing') || '[]'));
  }, []);

  return (
    <div className="glass-card p-5">
      <h3 className="font-semibold flex items-center gap-2 mb-3">
        <i className="fas fa-history text-blue-400" /> My History
      </h3>
      <div id="historyList" className="text-sm space-y-1">
        {history.length ? (
          history.slice(0, 8).map((h) => (
            <div key={h} className="text-sm">🎬 {h}</div>
          ))
        ) : (
          <div className="text-gray-400 text-xs">No history</div>
        )}
      </div>
      <div className="mt-3 pt-2 border-t border-white/10">
        <h4 className="text-xs font-semibold mb-2">▶️ ongoing</h4>
        <div id="ongoingList" className="text-xs text-gray-300">
          {ongoing.length ? (
            ongoing.map((o) => (
              <div key={o.title}>▶️ {o.title} - {o.progress}%</div>
            ))
          ) : (
            <div className="text-gray-400">No ongoing shows</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Content Explorer ──────────────────────────────────────────────────────────

function ContentExplorer({ onSelect }: { onSelect: (t: string) => void }) {
  const { catalog } = useAppStore();
  const { setLoading } = useUIStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MovieSummary[]>([]);
  const [genre, setGenre] = useState('all');

  // Debounce timer ref — avoids firing on every keystroke
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const genres = [...new Set(catalog.map((c) => c.genre).filter(Boolean))];

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setResults([]); return; }
    setLoading(true);
    const result = await searchMovies(q);
    setLoading(false);
    if (result.success) setResults(result.results);
    else setResults([]);
  }, [setLoading]);

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    // Debounce: wait 400ms after the user stops typing before hitting the API
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(val), 400);
  };

  // Cleanup timer on unmount
  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  return (
    <div className="glass-card p-5">
      <h3 className="font-semibold flex items-center gap-2 mb-3">
        <i className="fas fa-search" /> Explorer Agent
      </h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        <input
          id="searchInput"
          type="text"
          value={query}
          placeholder="Search movies or shows..."
          className="flex-1 bg-gray-900/70 border border-gray-700 rounded-xl px-4 py-2"
          onChange={handleInput}
        />
        <select
          id="genreFilter"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          className="bg-gray-900/70 border border-gray-700 rounded-xl px-3 py-2"
        >
          <option value="all">All Genres</option>
          {genres.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
      </div>
      <div id="explorerResults" className="grid grid-cols-1 gap-3 max-h-96 overflow-y-auto pr-1">
        {results.map((m) => (
          <div
            key={m.title}
            className="detail-card glass-card p-3 cursor-pointer"
            onClick={() => onSelect(m.title)}
          >
            <div className="font-bold">{m.title}</div>
            <div className="info-row"><span>⭐ Rating:</span><span>{m.rating}</span></div>
            <div className="info-row"><span>📅 Year:</span><span>{m.release_year}</span></div>
            <div className="info-row"><span>🎭 Type:</span><span>{m.type}</span></div>
            <div className="text-xs mt-1">{m.description?.substring(0, 100)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
