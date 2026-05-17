// API client — direct port of the fetch functions from index.html
// Uses NEXT_PUBLIC_API_BASE so it works in both dev (localhost:5000) and prod

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? '';

// ── Types ────────────────────────────────────────────────────────────────────

export interface UserFeaturePayload {
  age: number;
  account_age_months: number;
  monthly_fee: number;
  devices_used: number;
  avg_watch_time_minutes: number;
  watch_sessions_per_week: number;
  binge_watch_sessions: number;
  completion_rate: number;
  rating_given: number;
  content_interactions: number;
  recommendation_click_rate: number;
  days_since_last_login: number;
  // categorical fields for one-hot encoding
  gender?: string;
  country?: string;
  subscription_type?: string;
  payment_method?: string;
  primary_device?: string;
  favorite_genre?: string;
}

export interface MovieSummary {
  title: string;
  type: string;
  rating: string;
  release_year: number;
  listed_in?: string;
  description?: string;
}

export interface MovieDetail {
  title: string;
  type: string;
  director: string;
  cast: string;
  country: string;
  release_year: number;
  rating: string;
  listed_in: string;
  description: string;
  movie_duration_minutes: number | null;
  tv_show_seasons: number | null;
}

export interface SimilarMovie {
  title: string;
  rating: string;
  release_year: number;
}

export interface RecommendationResponse {
  success: true;
  cluster: number;
  genre: string;
  recommendations: MovieSummary[];
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function getRecommendations(
  userData: Partial<UserFeaturePayload>
): Promise<RecommendationResponse | { success: false; error: string }> {
  try {
    const response = await fetch(`${API_BASE}/api/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
    });
    const result = await response.json();
    if (result.success) {
      return {
        success: true,
        cluster: result.user_cluster,
        genre: result.dominant_genre,
        recommendations: result.recommendations,
      };
    }
    return { success: false, error: result.error };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function searchMovies(
  query: string
): Promise<{ success: true; results: MovieSummary[] } | { success: false; error: string }> {
  try {
    const response = await fetch(
      `${API_BASE}/api/search?q=${encodeURIComponent(query)}`
    );
    const result = await response.json();
    if (result.success) return { success: true, results: result.results };
    return { success: false, error: result.error };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function getMovieDetails(
  title: string
): Promise<{ success: true; movie: MovieDetail } | { success: false; error: string }> {
  try {
    const response = await fetch(
      `${API_BASE}/api/movie/${encodeURIComponent(title)}`
    );
    const result = await response.json();
    if (result.success) return { success: true, movie: result.movie };
    return { success: false, error: result.error };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function getSimilarMovies(
  title: string
): Promise<
  { success: true; similarMovies: SimilarMovie[] } | { success: false; error: string }
> {
  try {
    const response = await fetch(
      `${API_BASE}/api/movie-similar/${encodeURIComponent(title)}`
    );
    const result = await response.json();
    if (result.success)
      return { success: true, similarMovies: result.similar_movies };
    return { success: false, error: result.error };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}
