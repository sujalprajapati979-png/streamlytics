'use client';
import { useState, useEffect } from 'react';
import { useUIStore } from '@/store';
import { getMovieDetails, getSimilarMovies } from '@/lib/api';
import type { MovieDetail, SimilarMovie } from '@/lib/api';

export function MovieModal({ title, onClose }: { title: string; onClose: () => void }) {
  const { setLoading, showNotification } = useUIStore();
  const [movie, setMovie] = useState<MovieDetail | null>(null);
  const [similar, setSimilar] = useState<SimilarMovie[]>([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [nestedTitle, setNestedTitle] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getMovieDetails(title).then((res) => {
      setLoading(false);
      if (res.success) setMovie(res.movie);
      else showNotification(res.error, 'error');
    });
  }, [title]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadSimilar = async () => {
    setLoadingSimilar(true);
    const res = await getSimilarMovies(title);
    setLoadingSimilar(false);
    if (res.success) setSimilar(res.similarMovies);
    else showNotification(res.error, 'error');
  };

  return (
    <div
      className="fixed inset-0 bg-black/90 flex justify-center items-center z-[1000]"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-[#1a1a2e] rounded-xl p-8 max-w-2xl w-11/12 max-h-[80vh] overflow-y-auto relative">
        <button
          className="absolute top-4 right-4 text-2xl cursor-pointer hover:text-red-400 transition"
          onClick={onClose}
          aria-label="Close"
        >
          &times;
        </button>

        {movie ? (
          <>
            <h2 className="text-2xl font-bold mb-4">{movie.title}</h2>
            <div className="space-y-2 text-sm">
              <p><strong>Type:</strong> {movie.type}</p>
              <p><strong>Rating:</strong> {movie.rating}</p>
              <p><strong>Release Year:</strong> {movie.release_year}</p>
              <p><strong>Director:</strong> {movie.director || 'Unknown'}</p>
              <p><strong>Cast:</strong> {movie.cast || 'Unknown'}</p>
              <p><strong>Country:</strong> {movie.country || 'Unknown'}</p>
              <p><strong>Genres:</strong> {movie.listed_in || 'N/A'}</p>
              {movie.movie_duration_minutes && (
                <p><strong>Duration:</strong> {movie.movie_duration_minutes} minutes</p>
              )}
              {movie.tv_show_seasons && (
                <p><strong>Seasons:</strong> {movie.tv_show_seasons}</p>
              )}
              <p><strong>Description:</strong> {movie.description || 'No description'}</p>
            </div>

            <button
              className="mt-4 bg-red-600 hover:bg-red-700 transition px-4 py-2 rounded-lg text-sm font-semibold"
              onClick={loadSimilar}
              disabled={loadingSimilar}
            >
              {loadingSimilar ? 'Loading...' : 'Get Similar Movies'}
            </button>

            {similar.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Similar Movies</h4>
                <div className="similar-grid">
                  {similar.map((m) => (
                    <div
                      key={m.title}
                      className="similar-card"
                      onClick={() => setNestedTitle(m.title)}
                    >
                      <strong className="text-sm">{m.title}</strong>
                      <small className="block text-gray-400">{m.rating} | {m.release_year}</small>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex justify-center py-12">
            <div className="spinner" />
          </div>
        )}
      </div>

      {/* Nested modal for similar movie */}
      {nestedTitle && (
        <MovieModal title={nestedTitle} onClose={() => setNestedTitle(null)} />
      )}
    </div>
  );
}
