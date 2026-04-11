<p align="center">
  <h1 align="center">🎬 Streamlytics</h1>
  <p align="center">
    <strong>Hybrid Content-Based Recommendation & Sentiment Analytics Engine</strong>
  </p>
  <p align="center">
    An intelligent content recommendation and sentiment analytics platform that combines collaborative filtering, content-based algorithms, and advanced NLP to deliver personalized suggestions and real-time user sentiment analysis.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
  <a href="https://streamlytics-fei3.onrender.com/"><img src="https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GTU-Internship%202026-purple" alt="GTU Internship">
  <img src="https://img.shields.io/badge/Microsoft%20Elevate-Program-blue" alt="Microsoft Elevate">
  <img src="https://img.shields.io/badge/Ahmedabad%20Institute%20of%20Technology-CE%20Dept-brightgreen" alt="AIT">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Notebooks](#-notebooks)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## Overview

**Streamlytics** is a full-stack machine learning application that powers intelligent content recommendations and user behavior analytics. Built with a Flask backend and trained ML models, it serves personalized content suggestions through a REST API and an interactive web interface.

The platform is designed for digital media/streaming services and addresses core challenges like the **cold-start problem**, **user churn prediction**, and **preference-aware content discovery** — all from a single, deployable application.

### 🔗 [Try the Live Demo →](https://streamlytics-ggfx.onrender.com/)

---

## ✨ Key Features

### 🎯 Hybrid Recommendation Engine
- **Content-Based Filtering** using TF-IDF vectorization and cosine similarity on content metadata (genre, description, cast, director).
- **Cluster-Enhanced Recommendations** that leverage K-Means user segmentation to surface genre-aligned content.
- **Age-Appropriate Filtering** ensuring recommendations respect content ratings (TV-Y through NC-17) based on user age.

### 📊 User Churn Prediction
- **Dual-Model Ensemble** combining Random Forest and Logistic Regression classifiers.
- **42-Feature Vector** including demographics, watch behavior, engagement metrics, and subscription data.
- **Risk Stratification** with Low / Medium / High churn risk levels.

### 🧠 User Segmentation & Clustering
- **K-Means Clustering** (k=4) to identify distinct user behavioral segments.
- **Genre Preference Mapping** per cluster to understand dominant content preferences.
- **Personalized Seed Selection** for the recommendation engine based on cluster profiles.

### 🔍 Content Search & Discovery
- Real-time title search with fuzzy matching.
- Detailed movie/show metadata (director, cast, country, release year, duration, genre).
- Similar content discovery via the cosine similarity matrix.

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | Python 3.11+, Flask, Gunicorn |
| **ML / Data Science** | Scikit-Learn, Pandas, NumPy, SciPy, Imbalanced-Learn |
| **NLP** | TF-IDF Vectorizer (Scikit-Learn) |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 templates) |
| **Serialization** | Pickle, Joblib |
| **Deployment** | Render (PaaS), Gunicorn WSGI |
| **Notebooks** | Google Colab / Jupyter |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
streamlytics/
├── app.py                          # Flask application — routes, model loading, API logic
├── requirements.txt                # Pinned Python dependencies
├── render.yaml                     # Render deployment configuration
├── project structure.txt           # Quick reference for repository layout
│
├── models/                         # Serialized ML model artifacts
│   ├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
│   ├── indices.pkl                 # Title-to-index mapping
│   ├── df_titles.pkl               # Processed titles DataFrame
│   ├── kmeans_model.joblib         # K-Means user clustering model (k=4)
│   ├── feature_scaler.joblib       # StandardScaler for feature normalization
│   ├── random_forest_churn_model.joblib
│   ├── logistic_regression_churn_model.joblib
│   ├── cluster_genre_preferences.csv
│   ├── model_config.json           # Feature columns & model hyperparameters
│   ├── model_config.pkl            # Pickled model configuration
│   └── user_features_with_clusters.csv
│
├── data/                           # Runtime JSON data served to the frontend
│   ├── catalog.json                # Content catalog (fallback title source)
│   └── users.json                  # Sample user data
│
├── datasets/                       # Raw & processed datasets for training
│   ├── raw/
│   │   ├── netflix_titles.csv
│   │   ├── netflix_user_behavior_dataset.csv
│   │   └── merged_emotional_attachment_survey.csv
│   ├── df_attachment_survey_cleaned.csv
│   ├── df_netflix_titles_processed.csv
│   └── df_netflix_user_behavior_encoded.csv
│
├── colab/                          # Google Colab / Jupyter notebooks
│   ├── streamlytics analysis.ipynb # EDA, visualization & insights
│   └── streamlytics model.ipynb    # Model training & evaluation
│
└── templates/
    └── index.html                  # Main web interface (Jinja2 template)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (recommended; 3.8+ minimum)
- **pip** package manager
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sujalprajapati979-png/streamlytics.git
cd streamlytics

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

The application will start on **http://localhost:5000**. Open the URL in your browser to access the interactive dashboard.

> **Note:** On first startup, the app lazily loads all model artifacts from the `models/` directory. If any model file is missing, it gracefully degrades — the recommendation engine falls back to the `data/catalog.json` catalog.

---

## 💡 Usage

### Web Interface

Navigate to `http://localhost:5000` after starting the server. The dashboard provides:

- **Content Recommendations** — Enter your profile details (age, watch habits, preferences) to get personalized suggestions.
- **Movie Search** — Search the catalog by title.
- **Churn Risk Analysis** — Predict your churn probability based on engagement metrics.
- **Content Details** — View detailed metadata and find similar titles.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models/` | Path to the directory containing serialized model artifacts. |
| `FLASK_ENV` | `production` | Flask environment (`development` for debug mode). |
| `PORT` | `5000` | Port for the web server (used by Gunicorn on Render). |

---

## 📡 API Reference

All endpoints return JSON. The base URL is `http://localhost:5000`.

### Health & Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Basic liveness check. Returns service status. |
| `GET` | `/health/models` | Detailed model readiness per component. |

### Content Discovery

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/search?q=<query>` | Search titles by name (returns up to 20 results). |
| `GET` | `/api/movie/<title>` | Get full metadata for a specific title. |
| `GET` | `/api/movie-similar/<title>` | Get 10 similar titles based on cosine similarity. |

### Personalization

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/recommendations` | Get cluster-enhanced, age-filtered recommendations. |
| `POST` | `/api/predict-churn` | Predict user churn probability (ensemble model). |

<details>
<summary><strong>POST /api/recommendations — Request Body</strong></summary>

```json
{
  "age": 25,
  "account_age_months": 18,
  "monthly_fee": 14.99,
  "devices_used": 3,
  "avg_watch_time_minutes": 150,
  "watch_sessions_per_week": 12,
  "binge_watch_sessions": 6,
  "completion_rate": 70,
  "rating_given": 4.0,
  "content_interactions": 20,
  "recommendation_click_rate": 8,
  "days_since_last_login": 2
}
```

**Response:**

```json
{
  "success": true,
  "user_cluster": 2,
  "dominant_genre": "Drama",
  "recommendations": [
    {
      "title": "Example Movie",
      "type": "Movie",
      "rating": "PG-13",
      "description": "...",
      "release_year": 2023,
      "listed_in": "Dramas, International Movies"
    }
  ]
}
```

</details>

<details>
<summary><strong>POST /api/predict-churn — Request Body</strong></summary>

```json
{
  "age": 30,
  "account_age_months": 24,
  "monthly_fee": 12.99,
  "devices_used": 2,
  "avg_watch_time_minutes": 90,
  "watch_sessions_per_week": 5,
  "binge_watch_sessions": 2,
  "completion_rate": 40,
  "rating_given": 2.5,
  "content_interactions": 8,
  "recommendation_click_rate": 3,
  "days_since_last_login": 15
}
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "random_forest_probability": 0.72,
    "logistic_regression_probability": 0.68,
    "ensemble_probability": 0.70,
    "ensemble_prediction": 1,
    "risk_level": "High"
  }
}
```

</details>

---

## 📓 Notebooks

The `colab/` directory contains the Jupyter/Colab notebooks used for data exploration and model training:

| Notebook | Purpose |
|---|---|
| `streamlytics analysis.ipynb` | Exploratory data analysis — data cleaning, visualization, statistical insights on Netflix titles and user behavior. |
| `streamlytics model.ipynb` | Model training & evaluation — TF-IDF + cosine similarity, K-Means clustering, Random Forest & Logistic Regression churn models. |

> **Tip:** Open these notebooks in [Google Colab](https://colab.research.google.com/) for a zero-setup experience with GPU support.

---

## 🌐 Deployment

Streamlytics is configured for one-click deployment on **[Render](https://render.com)**:

1. Fork this repository.
2. Create a new **Web Service** on Render.
3. Connect your forked repo and Render will auto-detect `render.yaml`.
4. The app will build and deploy automatically.

**Production configuration** (from `render.yaml`):
- **Runtime:** Python 3.11
- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT`

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository.
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and ensure they work locally.
4. **Commit** with a clear, descriptive message:
   ```bash
   git commit -m "feat: add genre-based filtering to recommendations"
   ```
5. **Push** and open a **Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Ideas for Contribution

- Add a collaborative filtering module using user-item interaction matrices.
- Integrate a deep learning recommendation model (e.g., neural collaborative filtering).
- Build a sentiment analysis pipeline for user reviews.
- Add unit tests for the API routes.
- Improve the frontend dashboard (charts, visualizations).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **GTU (Gujarat Technological University)** — Summer Internship 2026
- **Microsoft Elevate Program**
- **Ahmedabad Institute of Technology** — Computer Engineering Department
- **Datasets:** [Netflix Titles](https://www.kaggle.com/datasets/shivamb/netflix-shows) & Netflix User Behavior Dataset

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/sujalprajapati979-png">Sujal Prajapati</a>
</p>
