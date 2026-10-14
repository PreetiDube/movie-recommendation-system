# Movie Recommendation System

A content-based movie recommendation system built with Python,
scikit-learn, and Streamlit.

## 🚀 Live Demo

Try the application here: [Movie Recommendation System](https://movie-recommendation-system-by-preeti-dube.streamlit.app/)


## Features

- Search and select a movie by title
- Recommend similar movies from a catalog of 141,687 movies
- Genre similarity: 65%
- Director similarity: 15%
- Language similarity: 10%
- Smoothed critic-review sentiment: up to 10% when available
- Configurable number of recommendations

## Technology Stack

- Python
- Pandas and NumPy
- SciPy sparse matrices
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
movie-recommendation-system/
├── app.py
├── requirements.txt
├── library_versions.json
├── model/
│   └── recommendation_model.joblib
├── notebooks/
│   └── 01_preprocess_original_rotten_tomatoes.ipynb
└── reports/
```

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Model Scoring

The recommender combines genre, director, and language similarity
with smoothed critic-review sentiment when available. Movies with
missing sentiment values do not receive a sentiment boost.

## Evaluation

The exported model and recommendation scoring were tested in Colab.
Evaluation metrics should be interpreted as offline proxies, not
guarantees of individual user satisfaction.

## Data and Attribution

The project uses Rotten Tomatoes movie and review data.
Review the original dataset's terms, license, and redistribution
permissions before publishing data or derived artifacts.

## Limitations

- Recommendations depend on the selected movie's metadata.
- Critic sentiment is unavailable for some movies.
- The recommendation score is a ranking score, not a probability.
