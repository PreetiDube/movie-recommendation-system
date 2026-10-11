
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "recommendation_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

movies_model = bundle["movies"]
id_to_index = bundle["id_to_index"]
genre_matrix = bundle["genre_matrix"]
director_matrix = bundle["director_matrix"]
language_matrix = bundle["language_matrix"]

st.title("🎬 Movie Recommendation System")
st.write(
    "Discover movies based on genre, director, language, "
    "and available critic-review sentiment."
)

@st.cache_data
def get_movie_options(titles):
    return titles


movie_titles = movies_model["title"].fillna("").astype(str)
unique_titles = sorted(
    movie_titles[movie_titles.str.strip() != ""].unique()
)

search_text = st.text_input(
    "Search for a movie",
    placeholder="Type at least 3 letters of a movie title...",
    key="movie_search"
)


if len(search_text.strip()) >= 3:
    matching_titles = [
        title for title in unique_titles
        if search_text.strip().casefold() in title.casefold()
    ][:100]

    if matching_titles:
        selected_title = st.selectbox(
            "Choose a movie from the search results",
            options=matching_titles,
            index=None,
            placeholder="Select the exact movie...",
            key="movie_selection"
        )
    else:
        selected_title = None
        st.info("No matching movies found. Try another title.")
else:
    selected_title = None



top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

def recommend(movie_title, top_n=10):
    matches = movies_model[
        movies_model["title"].fillna("").astype(str) == movie_title
    ]

    if matches.empty:
        return None, None

    # If multiple records share a title, use the first matching record.
    movie_id = str(matches.iloc[0]["id"])
    movie_index = id_to_index.get(movie_id)

    if movie_index is None:
        return None, None

    genre_scores = cosine_similarity(
        genre_matrix[movie_index], genre_matrix
    ).ravel()

    director_scores = cosine_similarity(
        director_matrix[movie_index], director_matrix
    ).ravel()

    language_scores = cosine_similarity(
        language_matrix[movie_index], language_matrix
    ).ravel()

    base_scores = (
        0.65 * genre_scores
        + 0.15 * director_scores
        + 0.10 * language_scores
    )

    sentiment = movies_model["smoothed_sentiment"].to_numpy(dtype=float)
    has_sentiment = ~np.isnan(sentiment)

    final_scores = base_scores.copy()
    final_scores[has_sentiment] += 0.10 * sentiment[has_sentiment]
    final_scores[movie_index] = -1

    ranking = pd.DataFrame({
        "index": np.arange(len(movies_model)),
        "score": final_scores,
        "id": movies_model["id"].astype(str).to_numpy()
    })

    ranking = ranking[ranking["index"] != movie_index]
    ranking = ranking.sort_values(
    ["score", "id"],
    ascending=[False, True],
    kind="mergesort"
        )
    top_indices = ranking.head(top_n)["index"].to_numpy()

result = movies_model.iloc[top_indices][[
    "title", "genre", "director",
    "originalLanguage", "reviewCount", "smoothed_sentiment"
]].copy()

    result["Recommendation Score"] = [
        round(float(final_scores[i]), 4) for i in top_indices
    ]

    result = result.rename(columns={
        "title": "Movie",
        "genre": "Genre",
        "director": "Director",
        "originalLanguage": "Language",
        "reviewCount": "Review Count",
        "smoothed_sentiment": "Critic Sentiment"
    })

    return movie_id, result.reset_index(drop=True)

if st.button("Recommend Movies", type="primary"):
    if not selected_title:
        st.warning("Please select a movie first.")
    else:
        with st.spinner("Finding similar movies..."):
            movie_id, recommendations = recommend(selected_title, top_n)

        if recommendations is None:
            st.error("Could not generate recommendations for this movie.")
        else:
            st.subheader(f"Movies similar to {selected_title}")
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

st.caption(
    "Recommendations are generated using content similarity "
    "and available smoothed critic-review sentiment."
)
