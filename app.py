
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

movies_model = bundle["movies"].reset_index(drop=True)
id_to_index = bundle["id_to_index"]
genre_matrix = bundle["genre_matrix"]
director_matrix = bundle["director_matrix"]
language_matrix = bundle["language_matrix"]

# Validate the saved model before recommending movies.
n_movies = len(movies_model)

if not (
    genre_matrix.shape[0]
    == director_matrix.shape[0]
    == language_matrix.shape[0]
    == n_movies
):
    st.error(
        "The saved movie data and feature matrices have different "
        "row counts. Please check how recommendation_model.joblib "
        "was created."
    )
    st.stop()

required_columns = {
    "id",
    "title",
    "genre",
    "director",
    "originalLanguage",
    "reviewCount",
    "smoothed_sentiment",
}

missing_columns = required_columns - set(movies_model.columns)

if missing_columns:
    st.error(f"Missing columns in the saved model: {sorted(missing_columns)}")
    st.stop()


st.markdown(
    """
    <div style="text-align:center; padding: 25px 0 20px 0;">
        <p style="color:#E50914; font-size:14px; letter-spacing:4px; font-weight:600;">
            YOUR NEXT FAVORITE FILM
        </p>
        <h1 style="font-size:42px; margin:0; color:#FFFFFF;">
            🎬 Movie <span style="color:#E50914;">Discovery</span>
        </h1>
        <p style="color:#B8B8C2; font-size:17px; margin-top:12px;">
            Discover your next favorite movie through intelligent recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Build the searchable movie list using both title and ID.
movie_options = movies_model[["id", "title"]].copy()
movie_options["id"] = movie_options["id"].astype(str)
movie_options["title"] = movie_options["title"].fillna("").astype(str)

movie_options = movie_options[
    (movie_options["title"].str.strip() != "")
    & (movie_options["id"].str.strip() != "")
].drop_duplicates(subset=["id"])

movie_options["display"] = movie_options["title"]

search_text = st.text_input(
    "🔎 FIND YOUR MOVIE",
    placeholder="Type at least 3 letters of a movie title...",
    key="movie_search"
)

selected_movie_id = None

if len(search_text.strip()) >= 3:
    matching_movies = movie_options[
        movie_options["title"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False
        )
    ].sort_values(["title", "id"]).head(100)

    if not matching_movies.empty:
        display_to_id = dict(
            zip(matching_movies["display"], matching_movies["id"])
        )

        selected_display = st.selectbox(
            "Choose a movie from the search results",
            options=list(display_to_id.keys()),
            index=None,
            placeholder="Select the exact movie...",
            key="movie_selection"
        )

        if selected_display is not None:
            selected_movie_id = display_to_id[selected_display]
    else:
        st.info("No matching movies found. Try another title.")

else:
    st.caption("Search by movie title to discover similar films.")


top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


def recommend(movie_id, top_n=10):
    movie_id = str(movie_id)

    matches = movies_model.index[
        movies_model["id"].astype(str) == movie_id
    ].tolist()

    if not matches:
        return None

    movie_index = matches[0]

    # Confirm the ID-to-row mapping before calculating similarities.
    if str(movies_model.iloc[movie_index]["id"]) != movie_id:
        return None

    # Calculate similarity against every movie.
    genre_scores = cosine_similarity(
        genre_matrix[movie_index], genre_matrix
    ).ravel()

    director_scores = cosine_similarity(
        director_matrix[movie_index], director_matrix
    ).ravel()

    language_scores = cosine_similarity(
        language_matrix[movie_index], language_matrix
    ).ravel()

    # Preserve the original scoring formula.
    base_scores = (
        0.65 * genre_scores
        + 0.15 * director_scores
        + 0.10 * language_scores
    )

    sentiment = movies_model["smoothed_sentiment"].to_numpy(dtype=float)
    has_sentiment = np.isfinite(sentiment)

    final_scores = base_scores.copy()
    final_scores[has_sentiment] += (
        0.10 * sentiment[has_sentiment]
    )

    # Never recommend the selected movie itself.
    final_scores[movie_index] = -1

    ranking = pd.DataFrame({
        "index": np.arange(n_movies),
        "score": final_scores,
        "id": movies_model["id"].astype(str).to_numpy()
    })

    ranking = ranking[
        (ranking["index"] != movie_index)
        & (ranking["id"] != movie_id)
    ]

    ranking = ranking.sort_values(
        ["score", "id"],
        ascending=[False, True],
        kind="mergesort"
    )

    top_indices = ranking.head(top_n)["index"].to_numpy()

    result = movies_model.iloc[top_indices][[
        "title",
        "genre",
        "director",
        "originalLanguage",
        "reviewCount",
        "smoothed_sentiment"
    ]].copy()

    result["Recommendation Score"] = [
        round(float(final_scores[i]), 4)
        for i in top_indices
    ]

    result = result.rename(columns={
        "title": "Movie",
        "genre": "Genre",
        "director": "Director",
        "originalLanguage": "Language",
        "reviewCount": "Review Count",
        "smoothed_sentiment": "Critic Sentiment"
    })

    result["Critic Sentiment"] = result["Critic Sentiment"].round(4)

    return result.reset_index(drop=True)


if st.button("Recommend Movies", type="primary"):
    if selected_movie_id is None:
        st.warning("Please search for and select a movie first.")
    else:
        selected_rows = movies_model[
            movies_model["id"].astype(str) == selected_movie_id
        ]

        selected_title = selected_rows.iloc[0]["title"]

        with st.spinner("Finding similar movies..."):
            recommendations = recommend(selected_movie_id, top_n)

        if recommendations is None:
            st.error("Could not generate recommendations for this movie.")
        elif recommendations.empty:
            st.warning("No recommendations are available.")
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
