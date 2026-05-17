'use client';
import { useAppStore, useUIStore } from '@/store';
import { getRecommendations } from '@/lib/api';

const clusterDefinitions: Record<number, { name: string; genre: string; archetype: string }> = {
  0: { name: 'Comedy Enthusiasts', genre: 'Comedy', archetype: '🎉 High binge' },
  1: { name: 'Thrill Seekers', genre: 'Thriller', archetype: '🔪 Intense' },
  2: { name: 'Drama Devotees', genre: 'Drama', archetype: '🎭 Character-driven' },
  3: { name: "Sci-Fi Explorers", genre: 'Sci-Fi', archetype: '🚀 Futuristic' },
  4: { name: 'Documentary Buffs', genre: 'Documentary', archetype: '📽️ Knowledge' },
};

export function UserPanel() {
  const { users, selectedUserId, setSelectedUserId, setCurrentUser, setRecommendations, mood, timeFilter } =
    useAppStore();
  const { setLoading, showNotification } = useUIStore();

  const analyzeUser = async () => {
    const user = users[selectedUserId];
    if (!user) { showNotification('Select a valid user', 'error'); return; }
    setCurrentUser(user);
    setLoading(true);
    try {
      const result = await getRecommendations({
        age: user.age,
        account_age_months: user.features?.account_age_months ?? 12,
        monthly_fee: user.features?.monthly_fee ?? 12.99,
        devices_used: user.features?.devices_used ?? 2,
        avg_watch_time_minutes: user.features?.avg_watch_time ?? 120,
        watch_sessions_per_week: user.features?.watch_sessions_per_week ?? 10,
        binge_watch_sessions: user.features?.binge_watch_sessions ?? 5,
        completion_rate: user.features?.completion_rate ?? 50,
        rating_given: user.features?.rating_given ?? 3,
        content_interactions: user.features?.content_interactions ?? 15,
        recommendation_click_rate: user.features?.recommendation_click_rate ?? 5,
        days_since_last_login: user.features?.days_since_login ?? 5,
        // categorical one-hot fields
        gender: user.gender,
        country: user.country,
        subscription_type: user.subscription_type,
        payment_method: user.payment_method,
        primary_device: user.primary_device,
        favorite_genre: user.preferred_genre,
      });
      if (result.success) {
        setRecommendations(result.recommendations, result.cluster, result.genre);
        showNotification(`Analyzed ${user.name}`, 'success');
      } else {
        showNotification(result.error, 'error');
      }
    } finally {
      setLoading(false);
    }
  };

  const currentUser = users[selectedUserId];
  const clusterInfo = clusterDefinitions[currentUser?.cluster ?? 0] ?? clusterDefinitions[0];
  const features = currentUser?.features ?? {};

  return (
    <div className="lg:col-span-1 space-y-6">
      {/* ── User Select ── */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 border-b border-white/10 pb-4 mb-5">
          <div className="w-10 h-10 rounded-full bg-red-600/20 flex items-center justify-center">
            <span className="text-xl">👤</span>
          </div>
          <h2 className="text-xl font-semibold">User Information</h2>
        </div>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-300 block mb-2">🎯 Select User</label>
            <select
              id="userIdSelect"
              value={selectedUserId}
              onChange={(e) => setSelectedUserId(e.target.value)}
              className="w-full bg-gray-900/70 border border-gray-700 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-red-500 transition"
            >
              <option value="">Select a user...</option>
              {Object.entries(users).map(([uid, u]) => (
                <option key={uid} value={uid}>
                  {uid} – {u.name} ({u.preferred_genre})
                </option>
              ))}
            </select>
          </div>
          <button
            id="analyzeBtn"
            onClick={analyzeUser}
            className="w-full bg-gradient-to-r from-red-700 to-red-800 hover:from-red-600 hover:to-red-700 transition-all rounded-xl py-3 font-semibold shadow-lg flex items-center justify-center gap-2 text-white"
          >
            <span>🔍</span> GET USER INFORMATION
          </button>
        </div>
      </div>

      {/* ── User Profile ── */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg">🧩</span>
          <span className="font-medium">User Profile</span>
        </div>
        <div id="clusterInfo" className="space-y-3 text-sm">
          {currentUser ? (
            <>
              <div className={`cluster-badge cluster-${currentUser.cluster} mb-2`}>
                {clusterInfo.name}
              </div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <span>👤 Name:</span><span>{currentUser.name}</span>
                <span>🎂 Age:</span><span>{currentUser.age}</span>
                <span>🎭 Preferred:</span><span>{currentUser.preferred_genre}</span>
                <span>⭐ Engagement:</span><span>{features.engagement_score ?? 'N/A'}/5</span>
                <span>📅 Days since login:</span><span>{features.days_since_login ?? 'N/A'}</span>
              </div>
              <div className="mt-2 text-xs">{clusterInfo.archetype}</div>
            </>
          ) : (
            <p className="text-gray-400">Select a user to view complete profile</p>
          )}
        </div>
      </div>

      {/* ── Mood Filter ── */}
      <MoodFilter />

      {/* ── Seasonal Spotlight ── */}
      <SeasonalSpotlight />
    </div>
  );
}

function MoodFilter() {
  const { mood, timeFilter, setMood, setTimeFilter } = useAppStore();
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 border-b border-white/10 pb-4 mb-5">
        <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center">
          <span className="text-xl">🎭</span>
        </div>
        <h2 className="text-xl font-semibold">whats your Mood!!</h2>
      </div>
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-300 block mb-2">😊 my Mood</label>
          <select
            id="moodSelect"
            value={mood}
            onChange={(e) => setMood(e.target.value)}
            className="w-full bg-gray-900/70 border border-gray-700 rounded-xl px-4 py-2"
          >
            <option value="all">All moods</option>
            <option value="funny">😂 Funny</option>
            <option value="emotional">😢 Emotional</option>
            <option value="mindbending">🧠 Mind-bending</option>
            <option value="scary">👻 Scary</option>
            <option value="romantic">💕 Romantic</option>
            <option value="action">⚡ Action</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-300 block mb-2">⏰ Time duration</label>
          <select
            id="timeSelect"
            value={timeFilter}
            onChange={(e) => setTimeFilter(e.target.value)}
            className="w-full bg-gray-900/70 border border-gray-700 rounded-xl px-4 py-2"
          >
            <option value="any">Any duration</option>
            <option value="short">📺 Short (&lt;30 min)</option>
            <option value="medium">🍿 Medium (30–60 min)</option>
            <option value="long">🎬 Long (&gt;60 min)</option>
          </select>
        </div>
      </div>
    </div>
  );
}

function SeasonalSpotlight() {
  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🌸</span>
        <span className="font-medium">Seasonal Spotlight</span>
        <span className="trending-badge text-xs px-2 py-0.5 rounded-full text-white ml-2">LIVE</span>
      </div>
      <div className="text-sm text-gray-300">
        🎬 Spring Season: Romantic Comedies &amp; Drama blooms
      </div>
    </div>
  );
}
