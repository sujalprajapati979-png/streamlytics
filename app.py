from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load models at startup
print("Loading models...")
tfidf = pickle.load(open('saved_models/tfidf_vectorizer.pkl', 'rb'))
cosine_sim = joblib.load('saved_models/cosine_sim.joblib')
indices = pickle.load(open('saved_models/indices.pkl', 'rb'))
df_titles = pd.read_pickle('saved_models/df_titles.pkl')
rf_model = joblib.load('saved_models/random_forest_churn_model.joblib')
scaler = joblib.load('saved_models/feature_scaler.joblib')
kmeans = joblib.load('saved_models/kmeans_model.joblib')

with open('saved_models/model_config.json', 'r') as f:
    config = json.load(f)

print("Models loaded successfully!")

# Recommendation function
def get_recommendations(title, n=10):
    if title not in indices.index:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_titles['title'].iloc[movie_indices].tolist()

def get_age_appropriate_ratings(age):
    if age < 13:
        return ['TV-Y', 'TV-G', 'G']
    elif age < 14:
        return ['TV-Y', 'TV-G', 'TV-PG', 'G', 'PG']
    elif age < 17:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'G', 'PG', 'PG-13']
    else:
        return ['TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'G', 'PG', 'PG-13', 'R', 'NC-17']

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    data = request.json
    title = data.get('title')
    age = data.get('age', 25)
    
    recs = get_recommendations(title, n=10)
    
    # Age filtering
    age_ratings = get_age_appropriate_ratings(age)
    filtered = df_titles[df_titles['title'].isin(recs) & df_titles['rating'].isin(age_ratings)]
    
    if len(filtered) > 0:
        recs = filtered['title'].tolist()
    
    return jsonify({
        'success': True,
        'recommendations': recs[:10],
        'seed_title': title
    })

@app.route('/api/churn-predict', methods=['POST'])
def churn_predict():
    data = request.json
    # Create feature vector from input
    # This should match your training features
    features = np.array([[
        data.get('age', 0),
        data.get('account_age_months', 0),
        data.get('monthly_fee', 0),
        data.get('devices_used', 0),
        data.get('avg_watch_time_minutes', 0),
        data.get('watch_sessions_per_week', 0),
        data.get('binge_watch_sessions', 0),
        data.get('completion_rate', 0),
        data.get('rating_given', 0),
        data.get('content_interactions', 0),
        data.get('recommendation_click_rate', 0),
        data.get('days_since_last_login', 0)
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    prediction = rf_model.predict(features_scaled)[0]
    probability = rf_model.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'success': True,
        'churn_prediction': int(prediction),
        'churn_probability': float(probability)
    })

@app.route('/api/search-titles', methods=['GET'])
def search_titles():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify({'titles': []})
    
    matches = df_titles[df_titles['title'].str.contains(query, case=False)]
    return jsonify({
        'titles': matches['title'].head(20).tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)