import os
import base64
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

try:
    import requests
except ImportError:
    requests = None

# -----------------------------------------------------------------------------
# Page setup and assets
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "recommendation_model.joblib"
HERO_IMAGE_PATH = BASE_DIR / "movie_hero.webp"


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def get_tmdb_api_key():
    """Read TMDB_API_KEY from Streamlit Secrets or the environment."""
    key = os.getenv("TMDB_API_KEY", "").strip()
    try:
        key = (st.secrets.get("TMDB_API_KEY", "") or key).strip()
    except Exception:
        # Streamlit raises when no secrets are configured; posters remain optional.
        pass
    return key


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_search(title: str, api_key: str):
    """Return poster and backdrop URLs for a title; fail safely if unavailable."""
    if not api_key or requests is None or not str(title).strip():
        return None, None
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": str(title).strip(),
                "include_adult": "false",
            },
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        for item in results:
            poster_path = item.get("poster_path")
            backdrop_path = item.get("backdrop_path")
            if poster_path or backdrop_path:
                poster = (
                    "https://image.tmdb.org/t/p/w500" + poster_path
                    if poster_path else None
                )
                backdrop = (
                    "https://image.tmdb.org/t/p/original" + backdrop_path
                    if backdrop_path else None
                )
                return poster, backdrop
    except Exception:
        pass
    return None, None


def safe_text(value, fallback="Not available"):
    if pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


if not MODEL_PATH.exists():
    st.error("The file recommendation_model.joblib was not found. Keep it in the same GitHub folder as app.py.")
    st.stop()

try:
    bundle = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load recommendation_model.joblib: {exc}")
    st.stop()

required_bundle_keys = {"movies", "genre_matrix", "director_matrix", "language_matrix"}
missing_bundle_keys = required_bundle_keys - set(bundle.keys())
if missing_bundle_keys:
    st.error(f"The saved model is missing required items: {sorted(missing_bundle_keys)}")
    st.stop()

movies = bundle["movies"].reset_index(drop=True).copy()
genre_matrix = bundle["genre_matrix"]
director_matrix = bundle["director_matrix"]
language_matrix = bundle["language_matrix"]
n_movies = len(movies)

required_columns = {
    "id", "title", "genre", "director", "originalLanguage",
    "reviewCount", "smoothed_sentiment",
}
missing_columns = required_columns - set(movies.columns)
if missing_columns:
    st.error(f"Missing columns in the saved model: {sorted(missing_columns)}")
    st.stop()

if not (
    genre_matrix.shape[0]
    == director_matrix.shape[0]
    == language_matrix.shape[0]
    == n_movies
):
    st.error("Movie data and feature matrices have different row counts. Check recommendation_model.joblib.")
    st.stop()

# Normalize IDs and titles so lookups and duplicate-title handling are consistent.
movies["id"] = movies["id"].astype(str)
movies["title"] = movies["title"].fillna("").astype(str)

# -----------------------------------------------------------------------------
# Visual styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #101b2a 0%, #080b11 52%, #07090d 100%);
        color: #f5f7fb;
    }
    .block-container { max-width: 1600px; padding-top: .7rem; padding-bottom: 2.5rem; }
    header[data-testid="stHeader"] { background: rgba(7,9,13,.92); }
    #MainMenu, footer { visibility: hidden; }
    nav.site-nav {
        display:flex; align-items:center; justify-content:space-between; gap:18px;
        padding:10px 4px 16px; margin-bottom:16px; border-bottom:1px solid #26303d;
    }
    .site-brand { color:#fff; font-size:clamp(17px,2vw,23px); font-weight:900; white-space:nowrap; }
    .site-brand span { color:#ff4059; }
    .site-links { display:flex; flex-wrap:wrap; align-items:center; gap:22px; }
    .site-links a { color:#e6eaf0 !important; text-decoration:none !important; font-size:13px; font-weight:700; }
    .site-links a:hover { color:#ff4059 !important; }
    .hero {
        position:relative; isolation:isolate; overflow:hidden; min-height:320px;
        padding:38px 34px; border:1px solid #202b39; border-radius:16px;
        margin:0 0 26px; background-color:#0a111b;
        background-image:linear-gradient(90deg,rgba(5,8,13,.98) 0%,rgba(5,8,13,.82) 38%,rgba(5,8,13,.20) 100%);
        background-size:cover; background-position:center 48%; display:flex; align-items:center;
        box-shadow:0 18px 50px rgba(0,0,0,.25);
    }
    .hero-content { max-width:760px; position:relative; z-index:1; }
    .hero-kicker { display:inline-block; color:#fff; background:rgba(255,48,75,.12); border:1px solid rgba(255,64,89,.65); border-radius:5px; padding:7px 10px; font-size:10px; letter-spacing:2px; font-weight:900; }
    .hero-title { color:#fff; font-size:clamp(32px,4vw,55px); font-weight:950; line-height:1.05; margin:14px 0 12px; }
    .hero-title span { color:#ff4059; }
    .hero-sub { color:#d3dae5; font-size:15px; line-height:1.65; max-width:650px; }
    .section-title { margin:25px 0 5px; font-size:25px; font-weight:850; color:#f7f8fb; }
    .section-subtitle { color:#9da8b8; font-size:13px; margin-bottom:10px; }
    .panel { padding:18px 18px 8px; border:1px solid #273343; border-radius:13px; background:linear-gradient(145deg,rgba(18,29,43,.92),rgba(10,15,23,.96)); margin:12px 0 24px; }
    .movie-card { height:100%; padding:10px; border:1px solid #283342; border-radius:12px; background:linear-gradient(145deg,#121b27,#0b1018); transition:transform .18s ease,border-color .18s ease; }
    .movie-card:hover { transform:translateY(-3px); border-color:#ff4059; }
    .poster { width:100%; height:205px; object-fit:cover; display:block; border-radius:7px; background:#1b2532; }
    .poster-placeholder { width:100%; height:205px; display:flex; align-items:center; justify-content:center; text-align:center; padding:12px; border-radius:7px; color:#b8c3d2; background:linear-gradient(145deg,#1d2a3a,#111722); font-weight:800; }
    .movie-name { color:#fff; font-size:14px; font-weight:850; line-height:1.3; margin:10px 0 7px; min-height:35px; }
    .score { display:inline-block; border:1px solid #8f2534; background:rgba(255,48,75,.12); color:#ff7a8b; border-radius:20px; padding:4px 8px; font-size:11px; font-weight:850; }
    .meta { color:#bdc6d3; font-size:11px; line-height:1.5; margin-top:7px; }
    .popular-strip { display:flex; gap:14px; overflow-x:auto; padding:8px 2px 17px; scroll-snap-type:x mandatory; scrollbar-width:thin; scrollbar-color:#485466 #111722; }
    .popular-card { flex:0 0 170px; width:170px; scroll-snap-align:start; padding:8px; border:1px solid #283342; border-radius:12px; background:#101722; }
    .popular-poster { width:100%; height:235px; object-fit:cover; display:block; border-radius:7px; background:#202a38; }
    .popular-title { color:#f5f7fb; font-size:13px; line-height:1.3; font-weight:800; min-height:34px; margin:9px 0 7px; }
    div.stButton > button[kind="primary"] { min-height:46px; border:0; border-radius:8px; font-weight:850; color:#fff; background:linear-gradient(100deg,#f21d3a,#ff4059); }
    div.stButton > button[kind="primary"]:hover { color:#fff; background:linear-gradient(100deg,#d91530,#ed304b); border:0; }
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:#172231; color:#f7f8fb; border-radius:8px; border-color:#435166; }
    div[data-testid="stSlider"] [data-testid="stTickBar"] { background:#394454; }
    div[data-testid="stDataFrame"] { border:1px solid #293646; border-radius:10px; overflow:hidden; }
    .footer { color:#7f8b9c; font-size:12px; text-align:center; margin-top:36px; padding:22px 10px; border-top:1px solid #25303d; }
    @media(max-width:760px) {
        nav.site-nav { align-items:flex-start; flex-direction:column; }
        .site-links { gap:14px; }
        .hero { min-height:290px; padding:26px 20px; }
        .popular-card { flex-basis:145px; width:145px; }
        .popular-poster { height:205px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation
st.markdown(
    """
    <nav class="site-nav" id="home">
      <div class="site-brand">🎬 Movie <span>Recommendation</span> System</div>
      <div class="site-links">
        <a href="#home">⌂ Home</a>
        <a href="#discover">⌕ Discover</a>
        <a href="#popular">▦ Popular Movies</a>
        <a href="#about">ⓘ About</a>
      </div>
    </nav>
    """,
    unsafe_allow_html=True,
)

# Hero: use the local supplied artwork first; otherwise use TMDB backdrop if configured.
api_key = get_tmdb_api_key()
hero_background = "linear-gradient(110deg,#080d14 0%,#111c2a 62%,#29101a 100%)"
if HERO_IMAGE_PATH.exists():
    try:
        encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")
        hero_background = (
            "linear-gradient(90deg,rgba(5,8,13,.97) 0%,rgba(5,8,13,.79) 38%,rgba(5,8,13,.12) 100%),"
            f"url('data:image/webp;base64,{encoded}')"
        )
    except OSError:
        pass
elif api_key:
    _, backdrop = tmdb_search("cinema", api_key)
    if backdrop:
        hero_background = (
            "linear-gradient(90deg,rgba(5,8,13,.97) 0%,rgba(5,8,13,.79) 38%,rgba(5,8,13,.12) 100%),"
            f"url('{escape(backdrop, quote=True)}')"
        )

st.markdown(
    f"""
    <section class="hero" style="background-image:{hero_background};">
      <div class="hero-content">
        <div class="hero-kicker">▶ YOUR NEXT FAVOURITE FILM</div>
        <div class="hero-title">Movie<br><span>Recommendation</span><br>System</div>
        <div class="hero-sub">Discover films you'll love with recommendations powered by genres, directors, languages, and available critic-review sentiment.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Popular Movies carousel - shown before search so visitors can browse first.
# -----------------------------------------------------------------------------
st.markdown('<div id="popular"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🍿 Popular Movies</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Browse the most-reviewed movies in this dataset.</div>', unsafe_allow_html=True)

popular = movies.copy()
popular["_reviews"] = pd.to_numeric(popular["reviewCount"], errors="coerce").fillna(0)
popular = (
    popular.sort_values(["_reviews", "title"], ascending=[False, True], kind="mergesort")
    .drop_duplicates(subset=["id"])
    .head(12)
)
popular_html = []
for _, item in popular.iterrows():
    title = safe_text(item["title"], "Untitled movie")
    poster, _ = tmdb_search(title, api_key) if api_key else (None, None)
    safe_title = escape(title)
    if poster:
        image_html = f'<img class="popular-poster" src="{escape(poster, quote=True)}" alt="Poster for {safe_title}" loading="lazy">'
    else:
        image_html = f'<div class="poster-placeholder" style="height:235px">🎬<br>{safe_title}</div>'
    reviews = int(max(0, item["_reviews"]))
    popular_html.append(
        f'<article class="popular-card">{image_html}'
        f'<div class="popular-title">{safe_title}</div>'
        f'<span class="score">Reviews · {reviews}</span></article>'
    )
st.markdown('<div class="popular-strip">' + ''.join(popular_html) + '</div>', unsafe_allow_html=True)
if not api_key:
    st.caption("To display real movie posters, add a TMDB_API_KEY in Streamlit app settings → Secrets. The site remains usable without it.")

# -----------------------------------------------------------------------------
# Search and recommendation controls
# -----------------------------------------------------------------------------
st.markdown('<div id="discover"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔎 Find Your Movie</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Search for a film, select the exact match, and customize your recommendations.</div>', unsafe_allow_html=True)

options = movies[["id", "title"]].copy()
options["id"] = options["id"].astype(str)
options["title"] = options["title"].fillna("").astype(str)
options = options[options["title"].str.strip().ne("") & options["id"].str.strip().ne("")].drop_duplicates("id")
# Keep movie titles clean in the dropdown.
options["display"] = options["title"]

# Keep only one search option for each exact movie title.
options = (
    options.sort_values(["title", "id"], kind="mergesort")
    .drop_duplicates(subset=["title"], keep="first")
)

with st.container(border=True):
    search_col, select_col = st.columns([1, 1])
    with search_col:
        search = st.text_input("Search for a movie", placeholder="Type at least 3 letters, e.g. Toy Story", key="movie_search")
    selected_id = None
    with select_col:
        st.markdown("**🎬 Choose a movie from the search results**")
        if len(search.strip()) >= 3:
            found = options[options["title"].str.contains(search.strip(), case=False, regex=False, na=False)].head(100)
            if not found.empty:
                choices = dict(zip(found["display"], found["id"]))
                picked = st.selectbox("Choose a movie", list(choices.keys()), index=None, placeholder="Select the exact movie...", label_visibility="collapsed", key="movie_selection")
                if picked is not None:
                    selected_id = choices[picked]
            else:
                st.info("No matching movies found. Try another title.")
        else:
            st.selectbox("Choose a movie", [], disabled=True, placeholder="Search above to see results", label_visibility="collapsed", key="movie_selection_empty")

    slider_col, button_col = st.columns([1.4, 1])
    with slider_col:
        top_n = st.slider("🎚️ Number of recommendations", min_value=4, max_value=12, value=8)
    with button_col:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        run = st.button("🎁 Recommend Movies", type="primary", use_container_width=True)



def recommend(movie_id: str, top_n: int) -> pd.DataFrame | None:
    """Rank similar movies, excluding the selected movie and duplicate titles."""
    movie_id = str(movie_id)

    matched_rows = movies.index[
        movies["id"].astype(str).eq(movie_id)
    ].tolist()

    if not matched_rows:
        return None

    movie_index = matched_rows[0]
    selected_title_key = " ".join(
        str(movies.iloc[movie_index]["title"]).casefold().split()
    )

    # Calculate content similarity.
    genre_scores = cosine_similarity(
        genre_matrix[movie_index], genre_matrix
    ).ravel()

    director_scores = cosine_similarity(
        director_matrix[movie_index], director_matrix
    ).ravel()

    language_scores = cosine_similarity(
        language_matrix[movie_index], language_matrix
    ).ravel()

    scores = (
        0.65 * genre_scores
        + 0.15 * director_scores
        + 0.10 * language_scores
    )

    sentiment = pd.to_numeric(
        movies["smoothed_sentiment"], errors="coerce"
    ).to_numpy(dtype=float)

    valid_sentiment = np.isfinite(sentiment)
    scores[valid_sentiment] += 0.10 * sentiment[valid_sentiment]

    # Normalize titles so duplicate title variants are excluded too.
    title_keys = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
        .to_numpy()
    )

    ranking = pd.DataFrame({
        "row": np.arange(len(movies)),
        "score": scores,
        "id": movies["id"].astype(str).to_numpy(),
        "title_key": title_keys,
    })

    # Exclude the exact selected row, its ID, and identical titles.
    ranking = ranking[
        (ranking["row"] != movie_index)
        & (ranking["id"] != movie_id)
        & (ranking["title_key"] != selected_title_key)
    ]

    ranking = (
        ranking
        .drop_duplicates(subset=["id"])
        .sort_values(
            ["score", "id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(int(top_n))
    )

    selected_rows = ranking["row"].to_numpy(dtype=int)

    result = movies.iloc[selected_rows][[
        "id",
        "title",
        "genre",
        "director",
        "originalLanguage",
        "reviewCount",
        "smoothed_sentiment",
    ]].copy()

    result["Recommendation Score"] = [
        round(float(scores[row]), 4) for row in selected_rows
    ]

    result = result.rename(columns={
        "title": "Movie",
        "genre": "Genre",
        "director": "Director",
        "originalLanguage": "Language",
        "reviewCount": "Review Count",
        "smoothed_sentiment": "Critic Sentiment",
    })

    result["Critic Sentiment"] = pd.to_numeric(
        result["Critic Sentiment"], errors="coerce"
    ).round(4)

    result["Review Count"] = pd.to_numeric(
        result["Review Count"], errors="coerce"
    )

    return result.reset_index(drop=True)



if run:
    if selected_id is None:
        st.warning("Please search for a movie and select it from the results first.")
    else:
        selected_matches = movies[movies["id"].astype(str).eq(str(selected_id))]
        if selected_matches.empty:
            st.error("The selected movie could not be found. Please search again.")
        else:
            selected_movie = selected_matches.iloc[0]
            selected_title = safe_text(selected_movie["title"], "Selected movie")
            with st.spinner("Finding your next favourite films..."):
                recommendations = recommend(selected_id, top_n)

            if recommendations is None or recommendations.empty:
                st.warning("No recommendations are available for this selection.")
            else:
                st.markdown("---")
                st.markdown(
                    f"<div class='section-title'>✨ Because you liked <span style='color:#ff4059'>{escape(selected_title)}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="section-subtitle">Here are some similar movies you might enjoy.</div>', unsafe_allow_html=True)

                # Attractive recommendation cards in a responsive row/grid.
                for start in range(0, len(recommendations), 4):
                    chunk = recommendations.iloc[start:start + 4]
                    columns = st.columns(4)
                    for col, (_, rec) in zip(columns, chunk.iterrows()):
                        with col:
                            title = safe_text(rec["Movie"], "Untitled movie")
                            poster, _ = tmdb_search(title, api_key) if api_key else (None, None)
                            if poster:
                                st.image(poster, use_container_width=True)
                            else:
                                st.markdown(
                                    f"<div class='poster-placeholder'>🎬<br>{escape(title)}</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**{escape(title)}**")
                            st.markdown(f"<span class='score'>Match · {float(rec['Recommendation Score']):.4f}</span>", unsafe_allow_html=True)
                            st.markdown(f"<div class='meta'>🎭 Genre: {escape(safe_text(rec['Genre']))}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='meta'>🎬 Director: {escape(safe_text(rec['Director']))}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='meta'>🌐 Language: {escape(safe_text(rec['Language']))}</div>", unsafe_allow_html=True)
                            sentiment_value = rec["Critic Sentiment"]
                            sentiment_display = f"{float(sentiment_value):.4f}" if pd.notna(sentiment_value) else "N/A"
                            review_value = rec["Review Count"]
                            review_display = f"{int(review_value)}" if pd.notna(review_value) else "N/A"
                            st.markdown(f"<div class='meta'>Reviews: {review_display} · Sentiment: {sentiment_display}</div>", unsafe_allow_html=True)
                            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                st.markdown('<div class="section-title">☷ All Recommended Movies</div>', unsafe_allow_html=True)
                table = recommendations.copy()
                table.insert(0, "Rank", np.arange(1, len(table) + 1))
                table = table.drop(columns=["id"]).rename(columns={"Recommendation Score": "Match Score"})
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("#", width="small"),
                        "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0.0, max_value=1.0, format="%.4f"),
                        "Critic Sentiment": st.column_config.NumberColumn("Critic Sentiment", format="%.4f"),
                        "Review Count": st.column_config.NumberColumn("Review Count", format="%d"),
                    },
                )

st.markdown(
    "<div class='footer' id='about'>Movie Recommendation System · Recommendations use content similarity and available smoothed critic-review sentiment. Movie poster images are provided by TMDB when configured.</div>",
    unsafe_allow_html=True,
)
