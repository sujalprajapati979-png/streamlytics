# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages

# ============================================
# LOAD ALL SAVED MODELS
# ============================================

print("Loading models...")

# Load recommendation system components
tfidf = pickle.load(open('saved_models/tfidf_vectorizer.pkl', 'rb'))
cosine_sim = joblib.load('saved_models/cosine_sim.joblib')
indices = pickle.load(open('saved_models/indices.pkl', 'rb'))
df_titles = pd.read_pickle('saved_models/df_titles.pkl')

# Load churn prediction models
rf_churn_model = joblib.load('saved_models/random_forest_churn_model.joblib')
lr_churn_model = joblib.load('saved_models/logistic_regression_churn_model.joblib')

# Load clustering model
kmeans = joblib.load('saved_models/kmeans_model.joblib')
scaler = joblib.load('saved_models/feature_scaler.joblib')

# Load configuration
with open('saved_models/model_config.pkl', 'rb') as f:
    config = pickle.load(f)

feature_columns = config['feature_columns']
genre_cols = config['genre_cols']

# Load cluster genre preferences
cluster_genre_preferences = pd.read_csv('saved_models/cluster_genre_preferences.csv', index_col=0)

print("All models loaded successfully!")

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_age_appropriate_ratings(age):
    if age < 13:
        return ['TV-Y', 'TV-G', 'G']
    elif age < 14:
        return ['TV-Y', 'TV-G', 'TV-PG', 'G', 'PG']
    elif age < 17:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'G', 'PG', 'PG-13']
    else:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'G', 'PG', 'PG-13', 'R', 'NC-17']

def get_recommendations(title, num_recommendations=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_titles['title'].iloc[movie_indices].tolist()

def get_cluster_recommendations(user_data, num_recommendations=10):
    """
    Get recommendations based on user's cluster
    user_data: dict with keys like 'age', 'cluster' (or features to predict cluster)
    """
    # If cluster not provided, predict it
    if 'cluster' not in user_data:
        # Scale user features and predict cluster
        user_features = pd.DataFrame([user_data])[feature_columns]
        user_features_scaled = scaler.transform(user_features)
        cluster = kmeans.predict(user_features_scaled)[0]
    else:
        cluster = user_data['cluster']
    
    # Get dominant genre for the cluster
    dominant_genre = cluster_genre_preferences.loc[cluster].idxmax()
    dominant_genre = dominant_genre.replace('favorite_genre_', '')
    
    # Find movies in that genre
    genre_movies = df_titles[df_titles['listed_in'].str.contains(dominant_genre, case=False, na=False)]
    
    if not genre_movies.empty:
        # Pick the longest movie as seed
        movies = genre_movies[genre_movies['type'] == 'Movie'].sort_values('movie_duration_minutes', ascending=False)
        if not movies.empty:
            seed_title = movies.iloc[0]['title']
        else:
            seed_title = genre_movies.iloc[0]['title']
        
        recommendations = get_recommendations(seed_title, num_recommendations)
        
        # Filter by age appropriateness
        age_appropriate = get_age_appropriate_ratings(user_data.get('age', 18))
        filtered = [r for r in recommendations if r in df_titles[df_titles['rating'].isin(age_appropriate)]['title'].values]
        
        return filtered[:num_recommendations]
    
    return get_recommendations('Spider-Man: Into the Spider-Verse', num_recommendations)[:num_recommendations]

def predict_churn(user_features):
    """Predict churn probability for a user"""
    # Ensure all required features are present
    df = pd.DataFrame([user_features])[feature_columns]
    probability = rf_churn_model.predict_proba(df)[0][1]
    prediction = rf_churn_model.predict(df)[0]
    return {
        'churn_probability': float(probability),
        'will_churn': bool(prediction)
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': True})

@app.route('/api/recommendations/title', methods=['POST'])
def get_recommendations_by_title():
    """Get recommendations based on a movie title"""
    data = request.json
    title = data.get('title')
    num_recs = data.get('num_recommendations', 10)
    
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    
    recommendations = get_recommendations(title, num_recs)
    return jsonify({
        'seed_title': title,
        'recommendations': recommendations
    })

@app.route('/api/recommendations/cluster', methods=['POST'])
def get_recommendations_by_cluster():
    """Get recommendations based on user's cluster and age"""
    user_data = request.json
    num_recs = user_data.get('num_recommendations', 10)
    
    recommendations = get_cluster_recommendations(user_data, num_recs)
    
    return jsonify({
        'user_info': user_data,
        'recommendations': recommendations
    })

@app.route('/api/recommendations/user', methods=['POST'])
def get_full_user_recommendations():
    """Get personalized recommendations using all user data"""
    user_data = request.json
    
    # Scale features and predict cluster
    user_features = pd.DataFrame([user_data])[feature_columns]
    user_features_scaled = scaler.transform(user_features)
    cluster = kmeans.predict(user_features_scaled)[0]
    
    # Add cluster to user data
    user_data['cluster'] = int(cluster)
    
    # Get recommendations
    recommendations = get_cluster_recommendations(user_data)
    
    # Get churn prediction
    churn_prediction = predict_churn(user_data)
    
    return jsonify({
        'user_cluster': int(cluster),
        'recommendations': recommendations,
        'churn_analysis': churn_prediction
    })

@app.route('/api/predict/churn', methods=['POST'])
def predict_user_churn():
    """Predict if a user will churn"""
    user_features = request.json
    result = predict_churn(user_features)
    return jsonify(result)

@app.route('/api/titles/search', methods=['GET'])
def search_titles():
    """Search for titles by name"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'results': []})
    
    matches = df_titles[df_titles['title'].str.lower().str.contains(query, na=False)]
    results = matches[['title', 'type', 'release_year', 'rating']].head(20).to_dict('records')
    
    return jsonify({'results': results})

@app.route('/api/titles/popular', methods=['GET'])
def get_popular_titles():
    """Get a list of popular titles to use as seeds"""
    # Get some diverse popular movies
    popular_movies = df_titles[df_titles['type'] == 'Movie'].head(50)['title'].tolist()
    return jsonify({'popular_titles': popular_movies})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
