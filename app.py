
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Discovery",
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

n_movies = len(movies_model)

if not (
    genre_matrix.shape[0]
    == director_matrix.shape[0]
    == language_matrix.shape[0]
    == n_movies
):
    st.error(
        "The saved movie data and feature matrices have different "
        "row counts. Please check recommendation_model.joblib."
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


# -------------------- VISUAL DESIGN --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1016;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #E50914, #FF414D);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 44px;
        padding: 0.5rem 1.5rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #FF414D;
        border: none;
        color: white;
    }

    [data-testid="stMetric"] {
        background: #191C25;
        border: 1px solid #303442;
        border-radius: 10px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- HERO HEADER --------------------

st.markdown(
    """
    <div style="text-align:center; padding:22px 0 28px 0;">
        <p style="color:#FF414D; font-size:13px;
                  letter-spacing:4px; font-weight:700;">
            YOUR NEXT FAVORITE FILM
        </p>
        <h1 style="font-size:44px; margin:0; color:#FFFFFF;">
            🎬 Movie <span style="color:#FF414D;">Discovery</span>
        </h1>
        <p style="color:#B8BBC7; font-size:17px; margin-top:12px;">
            Discover films you'll love with intelligent recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------- MOVIE SEARCH OPTIONS --------------------

movie_options = movies_model[["id", "title"]].copy()
movie_options["id"] = movie_options["id"].astype(str)
movie_options["title"] = movie_options["title"].fillna("").astype(str)

movie_options = movie_options[
    (movie_options["title"].str.strip() != "")
    & (movie_options["id"].str.strip() != "")
].drop_duplicates(subset=["id"]).copy()

# Keep ordinary titles clean, but disambiguate duplicate titles.
duplicate_titles = movie_options["title"].duplicated(keep=False)

movie_options["display"] = movie_options["title"]

movie_options.loc[duplicate_titles, "display"] = (
    movie_options.loc[duplicate_titles, "title"]
    + " · "
    + movie_options.loc[duplicate_titles, "id"]
)


# -------------------- SEARCH PANEL --------------------

selected_movie_id = None

with st.container(border=True):
    st.markdown("### 🔎 Find Your Movie")
    st.caption(
        "Search by title, choose your movie, and customize your recommendations."
    )

    search_text = st.text_input(
        "Movie title",
        placeholder="Type at least 3 letters of a movie title...",
        key="movie_search",
        label_visibility="collapsed"
    )

    matching_movies = movie_options.iloc[0:0].copy()

    if len(search_text.strip()) >= 3:
        matching_movies = movie_options[
            movie_options["title"].str.contains(
                search_text.strip(),
                case=False,
                regex=False,
                na=False
            )
        ].sort_values(["title", "id"]).head(100)

        if matching_movies.empty:
            st.info("No matching movies found. Try another title.")

    else:
        st.caption("Enter at least 3 letters to search for a movie.")

    if not matching_movies.empty:
        display_to_id = dict(
            zip(matching_movies["display"], matching_movies["id"])
        )

        selected_display = st.selectbox(
            "Choose a movie",
            options=list(display_to_id.keys()),
            index=None,
            placeholder="Select the exact movie...",
            key="movie_selection"
        )

        if selected_display is not None:
            selected_movie_id = display_to_id[selected_display]

    st.divider()

    st.markdown("**How many recommendations would you like?**")

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        label_visibility="collapsed"
    )

    generate_recommendations = st.button(
        "🎬 Recommend Movies",
        type="primary",
        use_container_width=True
    )


# -------------------- RECOMMENDATION ENGINE --------------------

def recommend(movie_id, top_n=10):
    movie_id = str(movie_id)

    matches = movies_model.index[
        movies_model["id"].astype(str) == movie_id
    ].tolist()

    if not matches:
        return None

    movie_index = matches[0]

    if str(movies_model.iloc[movie_index]["id"]) != movie_id:
        return None

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

    sentiment = pd.to_numeric(
        movies_model["smoothed_sentiment"],
        errors="coerce"
    ).to_numpy(dtype=float)

    has_sentiment = np.isfinite(sentiment)

    final_scores = base_scores.copy()

    final_scores[has_sentiment] += (
        0.10 * sentiment[has_sentiment]
    )

    # Exclude the selected movie by row and ID.
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

    result["Critic Sentiment"] = pd.to_numeric(
        result["Critic Sentiment"],
        errors="coerce"
    ).round(4)

    return result.reset_index(drop=True)


# -------------------- MOVIE RECOMMENDATION CARDS --------------------

if generate_recommendations:
    if selected_movie_id is None:
        st.warning("Please search for and select a movie first.")

    else:
        selected_rows = movies_model[
            movies_model["id"].astype(str) == selected_movie_id
        ]

        if selected_rows.empty:
            st.error("The selected movie could not be found.")
        else:
            selected_title = str(selected_rows.iloc[0]["title"])

            with st.spinner("Finding movies you'll love..."):
                recommendations = recommend(selected_movie_id, top_n)

            if recommendations is None:
                st.error("Could not generate recommendations for this movie.")

            elif recommendations.empty:
                st.warning("No recommendations are available.")

            else:
                st.markdown("---")
                st.markdown(
                    f"## 🍿 Movies similar to **{selected_title}**"
                )
                st.caption(
                    f"Showing {len(recommendations)} recommendations "
                    "ranked by your model."
                )

                # Summary metrics
                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Movies found",
                    len(recommendations)
                )

                metric2.metric(
                    "Selected movie",
                    selected_title
                )

                st.markdown("### Your recommendations")

                # Two cards per row.
                for start in range(0, len(recommendations), 2):
                    columns = st.columns(2)

                    for offset, column in enumerate(columns):
                        row_index = start + offset

                        if row_index >= len(recommendations):
                            break

                        movie = recommendations.iloc[row_index]

                        with column:
                            with st.container(border=True):
                                st.markdown(
                                    f"**#{row_index + 1} · "
                                    f"{movie['Movie']}**"
                                )

                                st.markdown(
                                    f"<span style='color:#FF414D;"
                                    f"font-size:13px;font-weight:600;'>"
                                    f"SIMILARITY SCORE: "
                                    f"{movie['Recommendation Score']:.4f}"
                                    f"</span>",
                                    unsafe_allow_html=True
                                )

                                genre = movie["Genre"]
                                director = movie["Director"]
                                language = movie["Language"]

                                st.caption(
                                    f"🎭 Genre: {genre if pd.notna(genre) and str(genre).strip() else 'Not available'}"
                                )

                                st.caption(
                                    f"🎬 Director: {director if pd.notna(director) and str(director).strip() else 'Not available'}"
                                )

                                st.caption(
                                    f"🌐 Language: {language if pd.notna(language) and str(language).strip() else 'Not available'}"
                                )

                                review_count = pd.to_numeric(
                                    pd.Series([movie["Review Count"]]),
                                    errors="coerce"
                                ).iloc[0]

                                sentiment_value = movie["Critic Sentiment"]

                                info1, info2 = st.columns(2)

                                info1.metric(
                                    "Reviews",
                                    (
                                        f"{review_count:,.0f}"
                                        if pd.notna(review_count)
                                        else "N/A"
                                    )
                                )

                                info2.metric(
                                    "Critic sentiment",
                                    (
                                        f"{sentiment_value:.3f}"
                                        if pd.notna(sentiment_value)
                                        else "N/A"
                                    )
                                )

                with st.expander("View all recommendations in a table"):
                    st.dataframe(
                        recommendations,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Recommendation Score": st.column_config.NumberColumn(
                                "Recommendation Score",
                                format="%.4f"
                            ),
                            "Critic Sentiment": st.column_config.NumberColumn(
                                "Critic Sentiment",
                                format="%.4f"
                            )
                        }
                    )


# -------------------- FOOTER --------------------

st.markdown("---")

st.caption(
    "Movie Discovery · Recommendations use genre, director, language, "
    "and available smoothed critic-review sentiment."
)
