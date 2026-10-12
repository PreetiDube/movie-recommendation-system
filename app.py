from html import escape
import os
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

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "recommendation_model.joblib"

# ---------------------------- Data and model ----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, api_key):
    """Return a TMDB poster URL when a valid API key is configured."""
    if not api_key or requests is None or not str(title).strip():
        return None
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": str(title)},
            timeout=5,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + results[0]["poster_path"]
    except Exception:
        return None
    return None


def get_tmdb_key():
    try:
        return st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


try:
    bundle = load_model()
except Exception as exc:
    st.error(f"Could not load the recommendation model. Details: {exc}")
    st.info("Keep recommendation_model.joblib in the same folder as app.py.")
    st.stop()

movies = bundle["movies"].reset_index(drop=True)
genre_matrix = bundle["genre_matrix"]
director_matrix = bundle["director_matrix"]
language_matrix = bundle["language_matrix"]
n_movies = len(movies)

if not (genre_matrix.shape[0] == director_matrix.shape[0] == language_matrix.shape[0] == n_movies):
    st.error("Movie data and feature matrices have different row counts. Check recommendation_model.joblib.")
    st.stop()

required_columns = {
    "id", "title", "genre", "director", "originalLanguage", "reviewCount", "smoothed_sentiment"
}
missing_columns = required_columns - set(movies.columns)
if missing_columns:
    st.error(f"Missing model columns: {sorted(missing_columns)}")
    st.stop()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "active_movie_id" not in st.session_state:
    st.session_state.active_movie_id = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "recommendation_title" not in st.session_state:
    st.session_state.recommendation_title = ""

# ---------------------------- Styling ----------------------------
st.markdown(
    """
    <style>
    :root { --cm-red:#E50914; --cm-black:#080808; --cm-panel:#141414; --cm-muted:#b3b3b3; }
    html, body, [data-testid="stAppViewContainer"] { background:#080808; color:#fff; }
    [data-testid="stHeader"] { background:rgba(8,8,8,.94); }
    [data-testid="stToolbar"] { right:1rem; }
    .block-container { max-width:1600px; padding-top:1rem; padding-bottom:2rem; }
    #MainMenu, footer { visibility:hidden; }
    [data-testid="stSidebar"] { background:#101010; }
    .cm-nav { display:flex; align-items:center; gap:28px; padding:6px 0 20px; border-bottom:1px solid #242424; margin-bottom:16px; }
    .cm-logo { font-size:28px; font-weight:900; letter-spacing:-1.2px; white-space:nowrap; }
    .cm-logo span { color:var(--cm-red); }
    .cm-nav-label { color:#ddd; font-size:14px; font-weight:600; }
    .cm-nav-tag { margin-left:auto; color:#aaa; font-size:12px; letter-spacing:1px; text-transform:uppercase; }
    .cm-hero { position:relative; min-height:330px; padding:44px 42px; border-radius:5px; overflow:hidden; margin:8px 0 30px;
       background:linear-gradient(90deg,rgba(8,8,8,.98) 0%,rgba(8,8,8,.88) 38%,rgba(8,8,8,.30) 100%),
       radial-gradient(ellipse at 78% 35%,#56242b 0%,#241417 28%,#111 62%,#080808 100%); }
    .cm-kicker { color:#f0444d; font-size:12px; letter-spacing:4px; font-weight:800; text-transform:uppercase; }
    .cm-hero h1 { font-size:clamp(36px,5vw,66px); line-height:1.02; margin:14px 0; color:#fff; font-weight:900; letter-spacing:-1.8px; }
    .cm-hero p { max-width:610px; color:#d0d0d0; font-size:16px; line-height:1.7; }
    .cm-meta { color:#ddd; font-size:13px; margin:14px 0 20px; }
    .cm-section-title { font-size:23px; font-weight:800; margin:28px 0 5px; color:#fff; }
    .cm-section-sub { color:#aaa; font-size:13px; margin-bottom:16px; }
    .cm-panel { background:#121212; border:1px solid #292929; border-radius:8px; padding:20px; margin:12px 0 24px; }
    .cm-card { background:#151515; border-radius:6px; padding:9px 9px 13px; margin-bottom:10px; transition:transform .2s ease, box-shadow .2s ease; height:100%; }
    .cm-card:hover { transform:scale(1.035); box-shadow:0 10px 28px rgba(0,0,0,.65); }
    .cm-poster { width:100%; height:220px; object-fit:cover; border-radius:4px; background:linear-gradient(145deg,#32151a,#171717 60%,#282828); display:flex; align-items:center; justify-content:center; color:#e50914; font-size:44px; }
    .cm-movie-title { font-size:14px; line-height:1.35; font-weight:800; margin:10px 0 6px; color:#fff; overflow-wrap:anywhere; }
    .cm-card-meta { font-size:11px; line-height:1.55; color:#b8b8b8; overflow-wrap:anywhere; }
    .cm-score { display:inline-block; background:#251114; border:1px solid #6e2027; color:#ff7b83; border-radius:3px; padding:3px 6px; font-size:11px; font-weight:800; margin:4px 0; }
    .cm-footer { color:#777; font-size:12px; text-align:center; padding:28px 0 10px; border-top:1px solid #222; margin-top:35px; }
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input { background:#202020; color:#fff; border-color:#393939; border-radius:4px; }
    div[data-testid="stSlider"] [data-baseweb="slider"] div { color:#fff; }
    div.stButton > button { border-radius:4px; border:1px solid #444; background:#222; color:#fff; font-weight:700; min-height:40px; }
    div.stButton > button:hover { border-color:#fff; color:#fff; }
    div.stButton > button[kind="primary"] { background:#E50914; border-color:#E50914; color:#fff; }
    div.stButton > button[kind="primary"]:hover { background:#b80710; border-color:#b80710; }
    div[data-testid="stDataFrame"] { border:1px solid #292929; border-radius:5px; }
    [data-testid="stMetric"] { background:#151515; padding:12px; border-radius:5px; }
    @media (max-width: 700px) { .cm-hero { padding:28px 22px; min-height:280px; } .cm-nav { gap:12px; flex-wrap:wrap; } .cm-nav-tag { display:none; } .cm-poster { height:175px; } }
    </style>
    <div class="cm-nav">
      <div class="cm-logo">Cine<span>Match</span></div>
      <div class="cm-nav-label">Home</div>
      <div class="cm-nav-label">Discover</div>
      <div class="cm-nav-label">Recommendations</div>
      <div class="cm-nav-tag">Your personal movie universe</div>
    </div>
    <section class="cm-hero">
      <div class="cm-kicker">Your next favourite film starts here</div>
      <h1>Stories worth<br>staying in for.</h1>
      <p>Discover your next favourite movie with intelligent recommendations shaped by genre, director, language, and available critic-review sentiment.</p>
      <div class="cm-meta">PERSONALIZED DISCOVERY &nbsp; • &nbsp; CONTENT-BASED AI</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Search and controls ----------------------------
options = movies[["id", "title"]].copy()
options["id"] = options["id"].astype(str)
options["title"] = options["title"].fillna("").astype(str)
options = options[options["title"].str.strip().ne("") & options["id"].str.strip().ne("")].drop_duplicates("id")
duplicates = options["title"].duplicated(keep=False)
options["display"] = options["title"]
options.loc[duplicates, "display"] = options.loc[duplicates, "title"] + " · ID " + options.loc[duplicates, "id"]
options = options.sort_values(["title", "id"])

st.markdown('<div class="cm-section-title">Find your movie</div>', unsafe_allow_html=True)
st.markdown('<div class="cm-section-sub">Search the catalogue, choose a title, and let CineMatch find films with similar characteristics.</div>', unsafe_allow_html=True)
with st.container():
    search_col, select_col = st.columns([1, 1.25], gap="large")
    with search_col:
        search_text = st.text_input("Search movie titles", placeholder="Try a title, e.g. Toy Story", key="movie_search")
    with select_col:
        selected_id = None
        if len(search_text.strip()) >= 2:
            matches = options[options["title"].str.contains(search_text.strip(), case=False, regex=False, na=False)].head(100)
            if not matches.empty:
                choice_map = dict(zip(matches["display"], matches["id"]))
                selected_display = st.selectbox("Select exact movie", list(choice_map), index=None, placeholder="Choose a movie…", key="movie_choice")
                if selected_display is not None:
                    selected_id = choice_map[selected_display]
            else:
                st.info("No matching titles found. Try another search.")
        else:
            st.selectbox("Select exact movie", [], disabled=True, placeholder="Type at least 2 letters above", key="movie_choice_empty")

settings_col, button_col = st.columns([1, 1], gap="large")
with settings_col:
    top_n = st.slider("Movies to recommend", min_value=4, max_value=12, value=8)
with button_col:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    recommend_clicked = st.button("▶  Find similar movies", type="primary", use_container_width=True)

# ---------------------------- Recommendation engine (preserved) ----------------------------
def recommend(movie_id, top_n=8):
    movie_id = str(movie_id)
    matches = movies.index[movies["id"].astype(str) == movie_id].tolist()
    if not matches:
        return None
    movie_index = matches[0]
    genre_scores = cosine_similarity(genre_matrix[movie_index], genre_matrix).ravel()
    director_scores = cosine_similarity(director_matrix[movie_index], director_matrix).ravel()
    language_scores = cosine_similarity(language_matrix[movie_index], language_matrix).ravel()
    # Preserve the original project's scoring weights and ranking behavior.
    base_scores = 0.65 * genre_scores + 0.15 * director_scores + 0.10 * language_scores
    sentiment = pd.to_numeric(movies["smoothed_sentiment"], errors="coerce").to_numpy(dtype=float)
    has_sentiment = np.isfinite(sentiment)
    final_scores = base_scores.copy()
    final_scores[has_sentiment] += 0.10 * sentiment[has_sentiment]
    final_scores[movie_index] = -1
    ranking = pd.DataFrame({
        "index": np.arange(n_movies),
        "score": final_scores,
        "id": movies["id"].astype(str).to_numpy(),
    })
    ranking = ranking[(ranking["index"] != movie_index) & (ranking["id"] != movie_id)]
    ranking = ranking.sort_values(["score", "id"], ascending=[False, True], kind="mergesort")
    top_indices = ranking.head(top_n)["index"].to_numpy()
    result = movies.iloc[top_indices][["id", "title", "genre", "director", "originalLanguage", "reviewCount", "smoothed_sentiment"]].copy()
    result["Recommendation Score"] = [round(float(final_scores[i]), 4) for i in top_indices]
    result = result.rename(columns={
        "title": "Movie", "genre": "Genre", "director": "Director",
        "originalLanguage": "Language", "reviewCount": "Review Count",
        "smoothed_sentiment": "Critic Sentiment",
    })
    result["Critic Sentiment"] = pd.to_numeric(result["Critic Sentiment"], errors="coerce").round(4)
    return result.reset_index(drop=True)

if recommend_clicked:
    if selected_id is None:
        st.warning("Search for a movie and select the exact title first.")
    else:
        selected_row = movies[movies["id"].astype(str) == selected_id].iloc[0]
        selected_title = str(selected_row["title"])
        with st.spinner("Finding movies you might love…"):
            results = recommend(selected_id, top_n)
        if results is None or results.empty:
            st.warning("No recommendations are available for this selection.")
        else:
            st.session_state.active_movie_id = selected_id
            st.session_state.recommendations = results
            st.session_state.recommendation_title = selected_title

# ---------------------------- Recommendation gallery ----------------------------
recs = st.session_state.recommendations
if recs is not None and not recs.empty:
    title = escape(str(st.session_state.recommendation_title))
    st.markdown(f'<div class="cm-section-title">Because you searched for <span style="color:#E50914">{title}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="cm-section-sub">A curated set of similar titles, ranked by the existing recommendation model.</div>', unsafe_allow_html=True)
    api_key = get_tmdb_key()
    if not api_key:
        st.caption("Poster artwork is optional. Add TMDB_API_KEY to Streamlit secrets to enable live poster lookup.")

    # Streamlit columns keep the cards responsive while presenting a poster-gallery layout.
    for start in range(0, len(recs), 4):
        row = recs.iloc[start:start + 4]
        cols = st.columns(4, gap="medium")
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                movie_title = str(movie["Movie"])
                poster_url = get_poster(movie_title, api_key)
                if poster_url:
                    st.markdown(f'<div class="cm-card"><img class="cm-poster" src="{escape(poster_url, quote=True)}" alt="Poster for {escape(movie_title, quote=True)}"><div class="cm-movie-title">{escape(movie_title)}</div><span class="cm-score">Match score · {float(movie["Recommendation Score"]):.3f}</span><div class="cm-card-meta">Genre · {escape(str(movie["Genre"])) if pd.notna(movie["Genre"]) else "Unavailable"}</div><div class="cm-card-meta">Director · {escape(str(movie["Director"])) if pd.notna(movie["Director"]) else "Unavailable"}</div><div class="cm-card-meta">Language · {escape(str(movie["Language"])) if pd.notna(movie["Language"]) else "Unavailable"}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="cm-card"><div class="cm-poster">🎬</div><div class="cm-movie-title">{escape(movie_title)}</div><span class="cm-score">Match score · {float(movie["Recommendation Score"]):.3f}</span><div class="cm-card-meta">Genre · {escape(str(movie["Genre"])) if pd.notna(movie["Genre"]) else "Unavailable"}</div><div class="cm-card-meta">Director · {escape(str(movie["Director"])) if pd.notna(movie["Director"]) else "Unavailable"}</div><div class="cm-card-meta">Language · {escape(str(movie["Language"])) if pd.notna(movie["Language"]) else "Unavailable"}</div></div>', unsafe_allow_html=True)
                movie_id = str(movie["id"])
                if movie_id in st.session_state.watchlist:
                    if st.button("✓ In My List", key=f"remove_{movie_id}", use_container_width=True):
                        st.session_state.watchlist.remove(movie_id)
                        st.rerun()
                else:
                    if st.button("＋ My List", key=f"add_{movie_id}", use_container_width=True):
                        st.session_state.watchlist.append(movie_id)
                        st.rerun()

    with st.expander("View recommendation data"):
        table = recs.copy()
        table.insert(0, "Rank", np.arange(1, len(table) + 1))
        table = table.drop(columns=["id"])
        st.dataframe(table, use_container_width=True, hide_index=True)

# ---------------------------- My List ----------------------------
st.markdown('<div class="cm-section-title">My List</div>', unsafe_allow_html=True)
st.markdown('<div class="cm-section-sub">Saved for this browser session. Your list is not linked to a permanent account.</div>', unsafe_allow_html=True)
if st.session_state.watchlist:
    saved = movies[movies["id"].astype(str).isin(st.session_state.watchlist)]
    list_cols = st.columns(4, gap="medium")
    for col, (_, movie) in zip(list_cols, saved.iterrows()):
        with col:
            poster_url = get_poster(str(movie["title"]), get_tmdb_key())
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown('<div class="cm-poster">🎬</div>', unsafe_allow_html=True)
            st.markdown(f"**{escape(str(movie['title']))}**")
            if st.button("Remove", key=f"watch_remove_{movie['id']}"):
                st.session_state.watchlist.remove(str(movie["id"]))
                st.rerun()
else:
    st.caption("Your list is empty. Use the + My List buttons on recommendations to save movies here.")

st.markdown('<div class="cm-footer">CineMatch · A movie discovery project powered by content similarity and available critic-review sentiment.</div>', unsafe_allow_html=True)
