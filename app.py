from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

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

def load_models():
    """Load all saved models and components"""
    global tfidf, cosine_sim, indices, df_titles, rf_model, lr_model
    global scaler, kmeans, cluster_genre_preferences, config, df_user_features
    
    print("Loading models...")
    
    # Load recommendation system components
    with open('saved_models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    cosine_sim = joblib.load('saved_models/cosine_sim.joblib')
    
    with open('saved_models/indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    
    df_titles = pd.read_pickle('saved_models/df_titles.pkl')
    
    # Load churn prediction models
    rf_model = joblib.load('saved_models/random_forest_churn_model.joblib')
    lr_model = joblib.load('saved_models/logistic_regression_churn_model.joblib')
    
    # Load scaler and clustering
    scaler = joblib.load('saved_models/feature_scaler.joblib')
    kmeans = joblib.load('saved_models/kmeans_model.joblib')
    
    # Load cluster preferences
    cluster_genre_preferences = pd.read_csv('saved_models/cluster_genre_preferences.csv', index_col=0)
    
    # Load config
    with open('saved_models/model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Load user features
    df_user_features = pd.read_csv('saved_models/user_features_with_clusters.csv', index_col=0)
    
    print("All models loaded successfully!")

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
    scaled_features = scaler.transform([user_features])
    cluster = kmeans.predict(scaled_features)[0]
    return cluster

def get_cluster_dominant_genre(cluster):
    """Get dominant genre for a cluster"""
    genre_cols = [col for col in cluster_genre_preferences.columns if col.startswith('favorite_genre_')]
    cluster_prefs = cluster_genre_preferences.loc[cluster, genre_cols]
    dominant = cluster_prefs.idxmax().replace('favorite_genre_', '')
    return dominant

def get_cluster_enhanced_recommendations(user_features, age, n_recommendations=10):
    """Get personalized recommendations using cluster information"""
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
        seed_title = genre_movies.nlargest(5, 'movie_duration_minutes')['title'].iloc[0]
    
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
    features_array = np.array(user_features).reshape(1, -1)
    
    rf_prob = rf_model.predict_proba(features_array)[0][1]
    lr_prob = lr_model.predict_proba(features_array)[0][1]
    
    # Ensemble prediction (average)
    ensemble_prob = (rf_prob + lr_prob) / 2
    ensemble_pred = 1 if ensemble_prob > 0.5 else 0
    
    return {
        'random_forest_probability': float(rf_prob),
        'logistic_regression_probability': float(lr_prob),
        'ensemble_probability': float(ensemble_prob),
        'ensemble_prediction': int(ensemble_pred),
        'risk_level': 'High' if ensemble_prob > 0.7 else 'Medium' if ensemble_prob > 0.3 else 'Low'
    }

# Routes
@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    """Get personalized recommendations based on user input"""
    try:
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
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'success': True, 'results': []})
    
    results = df_titles[df_titles['title'].str.lower().str.contains(query, na=False)]
    results = results.head(20)[['title', 'type', 'release_year', 'rating']].to_dict('records')
    
    return jsonify({'success': True, 'results': results})

@app.route('/api/movie/<title>', methods=['GET'])
def get_movie_details(title):
    """Get detailed information about a specific movie"""
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
