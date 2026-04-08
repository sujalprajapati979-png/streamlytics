from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
_model_load_attempted = False
_active_model_dir = None

# Global variables for loaded models
tfidf = None
cosine_sim = None
indices = None
df_titles = None
rf_model = None
lr_model = None
scaler = None
kmeans = None
cluster_genre_preferences = None
config = None
df_user_features = None


def _resolve_model_dir():
    """Resolve model directory from env or common local paths."""
    env_model_dir = os.getenv('MODEL_DIR', '').strip()
    candidates = []

    if env_model_dir:
        env_path = Path(env_model_dir)
        candidates.append(env_path if env_path.is_absolute() else (BASE_DIR / env_path))

    candidates.extend([
        BASE_DIR / 'saved_models',
        BASE_DIR / 'models',
    ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[-1]


def _ensure_df_titles_schema():
    """Ensure df_titles has the columns required by API routes."""
    global df_titles

    if df_titles is None:
        return

    required_defaults = {
        'title': '',
        'type': 'Movie',
        'rating': 'NR',
        'listed_in': 'Unknown',
        'description': '',
        'director': 'Unknown',
        'cast': 'Unknown',
        'country': 'Unknown',
        'release_year': np.nan,
        'movie_duration_minutes': np.nan,
        'tv_show_seasons': np.nan,
    }

    for col, default_val in required_defaults.items():
        if col not in df_titles.columns:
            df_titles[col] = default_val

    df_titles['title'] = df_titles['title'].fillna('').astype(str)
    df_titles['type'] = df_titles['type'].fillna('Movie').astype(str)
    df_titles['rating'] = df_titles['rating'].fillna('NR').astype(str)
    df_titles['listed_in'] = df_titles['listed_in'].fillna('Unknown').astype(str)
    df_titles['description'] = df_titles['description'].fillna('').astype(str)
    df_titles['director'] = df_titles['director'].fillna('Unknown').astype(str)
    df_titles['cast'] = df_titles['cast'].fillna('Unknown').astype(str)
    df_titles['country'] = df_titles['country'].fillna('Unknown').astype(str)
    df_titles['release_year'] = pd.to_numeric(df_titles['release_year'], errors='coerce')
    df_titles['movie_duration_minutes'] = pd.to_numeric(df_titles['movie_duration_minutes'], errors='coerce')
    df_titles['tv_show_seasons'] = pd.to_numeric(df_titles['tv_show_seasons'], errors='coerce')


def _load_titles_from_catalog():
    """Fallback title loading from data/catalog.json for local demo mode."""
    global df_titles

    catalog_path = DATA_DIR / 'catalog.json'
    if not catalog_path.exists():
        print(f"[load_models] Missing catalog fallback file: {catalog_path}")
        return False

    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)

        rows = []
        for item in catalog:
            duration = item.get('duration')
            seasons = item.get('seasons')
            release_year = item.get('year')

            try:
                duration = int(duration) if duration is not None else np.nan
            except (TypeError, ValueError):
                duration = np.nan

            try:
                seasons = int(seasons) if seasons is not None else np.nan
            except (TypeError, ValueError):
                seasons = np.nan

            try:
                release_year = int(release_year) if release_year is not None else np.nan
            except (TypeError, ValueError):
                release_year = np.nan

            rows.append({
                'title': item.get('title', ''),
                'type': item.get('type', 'Movie'),
                'rating': item.get('rating', 'NR'),
                'listed_in': item.get('genre', 'Unknown'),
                'description': item.get('ai_description', ''),
                'director': item.get('director', 'Unknown'),
                'cast': item.get('cast', 'Unknown'),
                'country': item.get('country', 'Unknown'),
                'release_year': release_year,
                'movie_duration_minutes': duration,
                'tv_show_seasons': seasons,
            })

        df_titles = pd.DataFrame(rows)
        _ensure_df_titles_schema()
        print(f"[load_models] Loaded fallback catalog with {len(df_titles)} records.")
        return not df_titles.empty
    except Exception as exc:
        print(f"[load_models] Failed to load fallback catalog: {exc}")
        return False


def ensure_runtime_data():
    """Ensure runtime data is available before serving model-dependent routes."""
    if df_titles is not None:
        return True
    return load_models()

def load_models():
    """Best-effort load for runtime model artifacts and title data."""
    global tfidf, cosine_sim, indices, df_titles, rf_model, lr_model
    global scaler, kmeans, cluster_genre_preferences, config, df_user_features
    global _model_load_attempted, _active_model_dir

    if _model_load_attempted:
        return df_titles is not None

    _model_load_attempted = True
    _active_model_dir = _resolve_model_dir()
    print(f"Loading runtime assets from: {_active_model_dir}")

    def _load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _safe_load(label, path, loader):
        if not path.exists():
            print(f"[load_models] Missing {label}: {path}")
            return None
        try:
            loaded_val = loader(path)
            print(f"[load_models] Loaded {label}")
            return loaded_val
        except Exception as exc:
            print(f"[load_models] Failed {label}: {exc}")
            return None

    tfidf = _safe_load('tfidf_vectorizer', _active_model_dir / 'tfidf_vectorizer.pkl', _load_pickle)
    cosine_sim = _safe_load('cosine_similarity_matrix', _active_model_dir / 'cosine_sim.joblib', joblib.load)
    indices = _safe_load('title_indices', _active_model_dir / 'indices.pkl', _load_pickle)

    loaded_titles = _safe_load('titles_dataframe', _active_model_dir / 'df_titles.pkl', pd.read_pickle)
    if loaded_titles is not None:
        df_titles = loaded_titles

    rf_model = _safe_load('random_forest_churn_model', _active_model_dir / 'random_forest_churn_model.joblib', joblib.load)
    lr_model = _safe_load('logistic_regression_churn_model', _active_model_dir / 'logistic_regression_churn_model.joblib', joblib.load)
    scaler = _safe_load('feature_scaler', _active_model_dir / 'feature_scaler.joblib', joblib.load)
    kmeans = _safe_load('kmeans_model', _active_model_dir / 'kmeans_model.joblib', joblib.load)
    cluster_genre_preferences = _safe_load(
        'cluster_genre_preferences',
        _active_model_dir / 'cluster_genre_preferences.csv',
        lambda p: pd.read_csv(p, index_col=0)
    )
    config = _safe_load('model_config', _active_model_dir / 'model_config.pkl', _load_pickle)
    df_user_features = _safe_load(
        'user_features_with_clusters',
        _active_model_dir / 'user_features_with_clusters.csv',
        lambda p: pd.read_csv(p, index_col=0)
    )

    if df_titles is None:
        _load_titles_from_catalog()
    else:
        _ensure_df_titles_schema()

    if indices is None and df_titles is not None and 'title' in df_titles.columns:
        indices = pd.Series(df_titles.index, index=df_titles['title']).drop_duplicates()
        print('[load_models] Built title indices from df_titles.')

    if df_titles is not None:
        print('Runtime data initialized successfully.')
        return True

    print('Runtime initialization incomplete: title data unavailable.')
    return False

# Helper functions
def get_age_appropriate_ratings(age):
    """Get allowed ratings based on user age"""
    if age < 13:
        return ['TV-Y', 'TV-G', 'G']
    elif age < 14:
        return ['TV-Y', 'TV-G', 'TV-PG', 'G', 'PG']
    elif age < 17:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'G', 'PG', 'PG-13']
    else:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'G', 'PG', 'PG-13', 'R', 'NC-17']

def get_recommendations(title, n_recommendations=10):
    """Get content-based recommendations for a title"""
    if not ensure_runtime_data():
        return []

    if 'title' not in df_titles.columns:
        return []

    if indices is None or cosine_sim is None:
        fallback_titles = df_titles['title'].dropna().astype(str).tolist()
        return [t for t in fallback_titles if t != title][:n_recommendations]

    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return df_titles['title'].iloc[movie_indices].tolist()

def get_user_cluster(user_features):
    """Predict user cluster based on features"""
    if scaler is None or kmeans is None:
        return 0

    scaled_features = scaler.transform([user_features])
    cluster = kmeans.predict(scaled_features)[0]
    return cluster

def get_cluster_dominant_genre(cluster):
    """Get dominant genre for a cluster"""
    if cluster_genre_preferences is None:
        return 'Drama'

    genre_cols = [col for col in cluster_genre_preferences.columns if col.startswith('favorite_genre_')]
    if not genre_cols:
        return 'Drama'

    if cluster not in cluster_genre_preferences.index:
        cluster = cluster_genre_preferences.index[0]

    cluster_prefs = cluster_genre_preferences.loc[cluster, genre_cols]
    dominant = cluster_prefs.idxmax().replace('favorite_genre_', '')
    return dominant

def get_cluster_enhanced_recommendations(user_features, age, n_recommendations=10):
    """Get personalized recommendations using cluster information"""
    if not ensure_runtime_data():
        return [], 0, 'Drama'

    cluster = get_user_cluster(user_features)
    dominant_genre = get_cluster_dominant_genre(cluster)
    
    # Find movies in the dominant genre
    genre_movies = df_titles[
        df_titles['listed_in'].str.contains(dominant_genre, case=False, na=False) &
        (df_titles['type'] == 'Movie')
    ]
    
    if genre_movies.empty:
        # Fallback to popular movie
        seed_title = 'Spider-Man: Into the Spider-Verse'
    else:
        # Pick a movie with good duration
        if 'movie_duration_minutes' in genre_movies.columns:
            seed_title = genre_movies.nlargest(5, 'movie_duration_minutes')['title'].iloc[0]
        else:
            seed_title = genre_movies['title'].iloc[0]
    
    # Get recommendations
    recommendations = get_recommendations(seed_title, n_recommendations * 2)
    
    # Filter by age appropriateness
    allowed_ratings = get_age_appropriate_ratings(age)
    filtered = df_titles[
        df_titles['title'].isin(recommendations) & 
        df_titles['rating'].isin(allowed_ratings)
    ]['title'].tolist()
    
    # Prioritize genre-aligned content
    genre_aligned = df_titles[
        df_titles['title'].isin(filtered) &
        df_titles['listed_in'].str.contains(dominant_genre, case=False, na=False)
    ]['title'].tolist()
    
    other = [t for t in filtered if t not in genre_aligned]
    final = (genre_aligned + other)[:n_recommendations]
    
    return final, cluster, dominant_genre

def predict_churn(user_features):
    """Predict churn probability using both models"""
    if rf_model is None and lr_model is None:
        raise ValueError('Churn models are not available.')

    features_array = np.array(user_features).reshape(1, -1)

    rf_prob = float(rf_model.predict_proba(features_array)[0][1]) if rf_model is not None else None
    lr_prob = float(lr_model.predict_proba(features_array)[0][1]) if lr_model is not None else None

    if rf_prob is not None and lr_prob is not None:
        ensemble_prob = (rf_prob + lr_prob) / 2
    else:
        ensemble_prob = rf_prob if rf_prob is not None else lr_prob

    ensemble_pred = 1 if ensemble_prob > 0.5 else 0
    
    return {
        'random_forest_probability': rf_prob,
        'logistic_regression_probability': lr_prob,
        'ensemble_probability': float(ensemble_prob),
        'ensemble_prediction': int(ensemble_pred),
        'risk_level': 'High' if ensemble_prob > 0.7 else 'Medium' if ensemble_prob > 0.3 else 'Low'
    }


@app.before_request
def initialize_runtime_once():
    """Initialize runtime assets lazily on the first incoming request."""
    if not _model_load_attempted:
        load_models()


def _runtime_component_status():
    """Return model/component readiness flags for operational visibility."""
    return {
        'tfidf': tfidf is not None,
        'cosine_sim': cosine_sim is not None,
        'indices': indices is not None,
        'df_titles': df_titles is not None,
        'rf_model': rf_model is not None,
        'lr_model': lr_model is not None,
        'scaler': scaler is not None,
        'kmeans': kmeans is not None,
        'cluster_genre_preferences': cluster_genre_preferences is not None,
        'config': config is not None,
        'df_user_features': df_user_features is not None,
    }

# Routes
@app.route('/health', methods=['GET'])
def health():
    """Basic liveness endpoint for deployment checks."""
    return jsonify({
        'status': 'ok',
        'service': 'streamlytics',
        'model_load_attempted': _model_load_attempted,
        'runtime_data_ready': df_titles is not None
    })


@app.route('/health/models', methods=['GET'])
def health_models():
    """Detailed runtime/model readiness endpoint."""
    component_status = _runtime_component_status()
    return jsonify({
        'status': 'ready' if component_status['df_titles'] else 'degraded',
        'model_dir': str(_active_model_dir or _resolve_model_dir()),
        'model_load_attempted': _model_load_attempted,
        'components': component_status,
    })


@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/data/<path:filename>', methods=['GET'])
def serve_data_files(filename):
    """Serve data JSON files required by the frontend."""
    allowed_files = {'users.json', 'catalog.json'}
    if filename not in allowed_files:
        abort(404)

    return send_from_directory(DATA_DIR, filename)

@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    """Get personalized recommendations based on user input"""
    try:
        if not ensure_runtime_data():
            return jsonify({'success': False, 'error': 'Runtime data is not available'}), 503

        data = request.json
        
        # Extract user features from request
        user_features = [
            float(data.get('age', 30)),
            float(data.get('account_age_months', 12)),
            float(data.get('monthly_fee', 12.99)),
            float(data.get('devices_used', 2)),
            float(data.get('avg_watch_time_minutes', 120)),
            float(data.get('watch_sessions_per_week', 10)),
            float(data.get('binge_watch_sessions', 5)),
            float(data.get('completion_rate', 50)),
            float(data.get('rating_given', 3.0)),
            float(data.get('content_interactions', 15)),
            float(data.get('recommendation_click_rate', 5)),
            float(data.get('days_since_last_login', 5)),
        ]
        
        # Add one-hot encoded features (simplified for demo)
        # In production, you'd need all 42 features
        # For demo, we pad with zeros
        while len(user_features) < 42:
            user_features.append(0)
        
        age = float(data.get('age', 30))
        
        recommendations, cluster, genre = get_cluster_enhanced_recommendations(
            user_features, age, n_recommendations=10
        )
        
        # Get movie details for each recommendation
        movie_details = []
        for title in recommendations:
            movie_data = df_titles[df_titles['title'] == title]
            if not movie_data.empty:
                movie_details.append({
                    'title': title,
                    'type': movie_data['type'].iloc[0],
                    'rating': movie_data['rating'].iloc[0],
                    'description': movie_data['description'].iloc[0][:200] + '...' if len(movie_data['description'].iloc[0]) > 200 else movie_data['description'].iloc[0],
                    'release_year': int(movie_data['release_year'].iloc[0]),
                    'listed_in': movie_data['listed_in'].iloc[0]
                })
        
        return jsonify({
            'success': True,
            'user_cluster': int(cluster),
            'dominant_genre': genre,
            'recommendations': movie_details
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn_api():
    """Predict churn probability for a user"""
    try:
        load_models()
        if rf_model is None and lr_model is None:
            return jsonify({'success': False, 'error': 'Churn models are not available'}), 503

        data = request.json
        
        # Build feature vector (simplified for demo)
        user_features = [
            float(data.get('age', 30)),
            float(data.get('account_age_months', 12)),
            float(data.get('monthly_fee', 12.99)),
            float(data.get('devices_used', 2)),
            float(data.get('avg_watch_time_minutes', 120)),
            float(data.get('watch_sessions_per_week', 10)),
            float(data.get('binge_watch_sessions', 5)),
            float(data.get('completion_rate', 50)),
            float(data.get('rating_given', 3.0)),
            float(data.get('content_interactions', 15)),
            float(data.get('recommendation_click_rate', 5)),
            float(data.get('days_since_last_login', 5)),
        ]
        
        while len(user_features) < 42:
            user_features.append(0)
        
        prediction = predict_churn(user_features)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by title"""
    if not ensure_runtime_data():
        return jsonify({'success': False, 'error': 'Runtime data is not available'}), 503

    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'success': True, 'results': []})
    
    results = df_titles[df_titles['title'].str.lower().str.contains(query, na=False)]
    results = results.head(20)[['title', 'type', 'release_year', 'rating']].to_dict('records')
    
    return jsonify({'success': True, 'results': results})

@app.route('/api/movie/<title>', methods=['GET'])
def get_movie_details(title):
    """Get detailed information about a specific movie"""
    if not ensure_runtime_data():
        return jsonify({'success': False, 'error': 'Runtime data is not available'}), 503

    movie = df_titles[df_titles['title'] == title]
    if movie.empty:
        return jsonify({'success': False, 'error': 'Movie not found'}), 404
    
    movie = movie.iloc[0]
    return jsonify({
        'success': True,
        'movie': {
            'title': movie['title'],
            'type': movie['type'],
            'director': movie['director'],
            'cast': movie['cast'],
            'country': movie['country'],
            'release_year': int(movie['release_year']),
            'rating': movie['rating'],
            'listed_in': movie['listed_in'],
            'description': movie['description'],
            'movie_duration_minutes': int(movie['movie_duration_minutes']) if pd.notna(movie['movie_duration_minutes']) else None,
            'tv_show_seasons': int(movie['tv_show_seasons']) if pd.notna(movie['tv_show_seasons']) else None
        }
    })

@app.route('/api/movie-similar/<title>', methods=['GET'])
def get_similar_movies(title):
    """Get similar movies to a given title"""
    try:
        if not ensure_runtime_data():
            return jsonify({'success': False, 'error': 'Runtime data is not available'}), 503

        similar = get_recommendations(title, n_recommendations=10)
        
        movie_details = []
        for t in similar:
            movie_data = df_titles[df_titles['title'] == t]
            if not movie_data.empty:
                movie_details.append({
                    'title': t,
                    'rating': movie_data['rating'].iloc[0],
                    'release_year': int(movie_data['release_year'].iloc[0])
                })
        
        return jsonify({'success': True, 'similar_movies': movie_details})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
