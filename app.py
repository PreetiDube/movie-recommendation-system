import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import base64
from html import escape
import textwrap
from sklearn.metrics.pairwise import cosine_similarity

try:
    import requests
except ImportError:
    requests = None

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
MODEL_PATH = Path(__file__).parent / "recommendation_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def get_tmdb_api_key():
    """Read TMDB key from Streamlit Secrets or an environment variable.

    For Streamlit Community Cloud, add TMDB_API_KEY to App settings > Secrets.
    For local runs, you can also set the TMDB_API_KEY environment variable.
    """
    key = os.getenv("TMDB_API_KEY", "").strip()
    try:
        key = (st.secrets.get("TMDB_API_KEY", "") or key).strip()
    except Exception:
        # Streamlit raises when no secrets.toml / Cloud Secrets are configured.
        pass
    return key


@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, api_key):
    """Return a TMDB poster URL; return None safely when unavailable."""
    if not api_key or requests is None or not str(title).strip():
        return None
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": str(title).strip(), "include_adult": "false"},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        # Prefer the first result that actually has a poster, rather than
        # returning a placeholder when the first title match has no artwork.
        for result in results:
            poster_path = result.get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        return None
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_backdrop(title, api_key):
    """Return a TMDB backdrop URL when one is available."""
    if not api_key or requests is None or not str(title).strip():
        return None
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": str(title).strip(), "include_adult": "false"},
            timeout=8,
        )
        response.raise_for_status()
        for result in response.json().get("results", []):
            backdrop_path = result.get("backdrop_path")
            if backdrop_path:
                return "https://image.tmdb.org/t/p/original" + backdrop_path
    except Exception:
        return None
    return None

bundle = load_model()
movies = bundle["movies"].reset_index(drop=True)
genre_matrix = bundle["genre_matrix"]
director_matrix = bundle["director_matrix"]
language_matrix = bundle["language_matrix"]
n = len(movies)

if not (genre_matrix.shape[0] == director_matrix.shape[0] == language_matrix.shape[0] == n):
    st.error("Movie data and feature matrices have different row counts. Check recommendation_model.joblib.")
    st.stop()

required = {"id", "title", "genre", "director", "originalLanguage", "reviewCount", "smoothed_sentiment"}
missing = required - set(movies.columns)
if missing:
    st.error(f"Missing model columns: {sorted(missing)}")
    st.stop()

st.markdown("""
<style>
.stApp {
    background: #080808;
    color: #ffffff;
}
.block-container {
    max-width: 1600px;
    padding-top: .7rem;
    padding-bottom: 2rem;
}

/* Navigation */
nav.site-nav#home {
    position: relative;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 10px 4px 16px;
    margin: 0 0 8px;
    border-bottom: 1px solid #242424;
    background: #080808 !important;
}
nav.site-nav#home .site-brand {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
    font-size: 20px;
    font-weight: 900;
    white-space: pre-line;
    text-shadow: none !important;
}
nav.site-nav#home .site-brand span {
    color: #e50914 !important;
    -webkit-text-fill-color: #e50914 !important;
}
nav.site-nav#home .site-links {
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}
nav.site-nav#home .site-links a,
nav.site-nav#home .site-links a:link,
nav.site-nav#home .site-links a:visited {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: 700 !important;
    text-shadow: none !important;
    transition: color .18s ease;
}
nav.site-nav#home .site-links a:hover,
nav.site-nav#home .site-links a:focus-visible,
nav.site-nav#home .site-links a.active {
    color: #e50914 !important;
    -webkit-text-fill-color: #e50914 !important;
}
nav.site-nav#home .site-links a:focus-visible {
    outline: 2px solid #e50914;
    outline-offset: 4px;
    border-radius: 2px;
}

.section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 20px 0 2px;
}
.section-heading h3 { margin: 0; }

header[data-testid="stHeader"] {
    background: rgba(8, 8, 8, .92);
}
section[data-testid="stSidebar"] { background: #111111; }

.hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    min-height: 410px;
    padding: 44px 38px 40px;
    border-radius: 8px;
    margin: 0 0 24px;
    background-color: #080808;
    background-image:
        linear-gradient(90deg, rgba(8,8,8,.98) 0%, rgba(8,8,8,.80) 34%, rgba(8,8,8,.25) 68%, rgba(8,8,8,.08) 100%),
        linear-gradient(0deg, rgba(8,8,8,.35), transparent 55%);
    background-size: cover;
    background-position: center 48%;
    display: flex;
    align-items: center;
    box-shadow: 0 18px 48px rgba(0,0,0,.28);
}
.hero-content { max-width: 780px; position: relative; z-index: 1; }
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #ffffff;
    background: rgba(229,9,20,.18);
    border: 1px solid rgba(229,9,20,.55);
    border-radius: 3px;
    padding: 6px 9px;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 800;
    text-transform: uppercase;
}
.hero-title {
    font-size: clamp(32px, 4vw, 56px);
    font-weight: 900;
    line-height: 1.08;
    margin: 10px 0 14px;
    max-width: 850px;
    text-shadow: 0 2px 20px rgba(0,0,0,.6);
    color: #ffffff;
}
.hero-title span { color: #e50914 !important; }
.hero-sub {
    color: #e0e0e0;
    font-size: 16px;
    max-width: 700px;
    line-height: 1.65;
    text-shadow: 0 1px 10px #000;
}

.panel {
    padding: 20px;
    border: 1px solid #292929;
    border-radius: 10px;
    background: #141414;
    margin: 10px 0 22px;
}
.movie-card {
    border: 1px solid #292929;
    border-radius: 8px;
    background: #151515;
    padding: 10px;
    min-height: 100%;
    margin-bottom: 5px;
    transition: transform .2s ease, border-color .2s ease;
}
.movie-card:hover { transform: scale(1.025); border-color: #666; }
.poster {
    height: 250px;
    border-radius: 5px;
    background: linear-gradient(145deg, #242424, #101010);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: #777;
    letter-spacing: 2px;
    text-align: center;
    margin-bottom: 12px;
    overflow: hidden;
}
.poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-missing {
    background: linear-gradient(145deg, #242424, #101010);
    color: #777;
    font-size: 12px;
    letter-spacing: 2px;
    text-align: center;
}
.movie-name { font-size: 15px; font-weight: 800; margin-bottom: 8px; min-height: 36px; line-height: 1.25; }
.score {
    display: inline-block;
    border: 1px solid #7c1119;
    background: #260a0d;
    color: #ff727a;
    border-radius: 4px;
    padding: 4px 7px;
    font-size: 11px;
    font-weight: 800;
}
.meta { color: #b3b3b3; font-size: 11px; line-height: 1.45; margin-top: 6px; }
.carousel {
    display: flex;
    gap: 14px;
    overflow-x: auto;
    padding: 10px 3px 18px;
    scroll-snap-type: x mandatory;
    scrollbar-width: thin;
    scrollbar-color: #555 #151515;
}
.carousel-card {
    flex: 0 0 190px;
    max-width: 190px;
    scroll-snap-align: start;
    border: 1px solid #292929;
    border-radius: 7px;
    background: #151515;
    padding: 8px;
    transition: transform .2s ease, border-color .2s ease;
}
.carousel-card:hover { transform: scale(1.035); border-color: #666; }
.carousel-poster {
    width: 100%;
    height: 265px;
    object-fit: cover;
    border-radius: 4px;
    display: block;
    background: #242424;
}
.carousel-title { font-size: 14px; font-weight: 800; margin: 9px 0 7px; line-height: 1.3; min-height: 36px; color: #ffffff; }
.carousel-meta { font-size: 11px; line-height: 1.45; color: #b3b3b3; margin-top: 5px; }

div.stButton > button[kind="primary"] {
    border: 0;
    border-radius: 5px;
    min-height: 46px;
    font-weight: 800;
    background: #e50914;
    color: #ffffff;
}
div.stButton > button[kind="primary"]:hover { background: #b80710; color: #ffffff; }
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #222222;
    color: #ffffff;
    border-radius: 5px;
    border-color: #444444;
}
div[data-testid="stDataFrame"] { border: 1px solid #292929; border-radius: 8px; overflow: hidden; }
.footer { color: #777777; font-size: 12px; text-align: center; margin-top: 35px; padding: 20px; border-top: 1px solid #222222; }
h2, h3 { color: #ffffff; }

@media (max-width: 700px) {
    nav.site-nav#home { align-items: flex-start; flex-direction: column; }
    nav.site-nav#home .site-links { gap: 16px; }
    nav.site-nav#home .site-brand { font-size: 17px; }
    .hero { min-height: 350px; padding: 28px 22px; }
    .hero-title { font-size: clamp(30px, 8vw, 44px); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<nav class="site-nav" id="home">
  <div class="site-brand">🎬 Movie <span>Recommendation</span> System</div>
  <div class="site-links">
    <a class="active" href="#home">⌂ Home</a>
    <a href="#discover">⌕ Discover</a>
    <a href="#popular">▣ Popular Movies</a>
    <a href="#about">ⓘ About</a>
  </div>
</nav>
""", unsafe_allow_html=True)

# Keep the approved cinematic hero artwork available even without a TMDB API key.
# Upload movie_hero.webp into the same GitHub folder as app.py.
hero_image_path = Path(__file__).parent / "movie_hero.webp"
hero_api_key = get_tmdb_api_key()
hero_backdrop = get_backdrop("Interstellar", hero_api_key) if hero_api_key else None

if hero_image_path.exists():
    hero_image_b64 = base64.b64encode(hero_image_path.read_bytes()).decode("ascii")
    hero_style = (
        "background-image: linear-gradient(90deg, "
        "rgba(8,8,8,0.96) 0%, rgba(8,8,8,0.78) 32%, "
        "rgba(8,8,8,0.24) 68%, rgba(8,8,8,0.06) 100%), "
        f"url('data:image/webp;base64,{hero_image_b64}');"
    )
elif hero_backdrop:
    hero_style = (
        "background-image: linear-gradient(90deg, "
        "rgba(8,8,8,0.96) 0%, rgba(8,8,8,0.65) 45%, "
        "rgba(8,8,8,0.12) 100%), "
        f"url('{escape(hero_backdrop, quote=True)}');"
    )
else:
    hero_style = (
        "background-image: linear-gradient(90deg, #080808 0%, "
        "#101010 58%, #241014 100%);"
    )

st.markdown(f"""
<div class="hero" id="home-hero" style="{hero_style}">
 <div class="hero-content">
   <div class="hero-kicker">▶ YOUR NEXT FAVOURITE FILM</div>
   <div class="hero-title">Movie<br><span>Recommendation</span><br>System</div>
   <div class="hero-sub">Discover your next favourite film with personalized recommendations powered by movie genres, directors, languages, and critic-review sentiment.</div>
 </div>
</div>
""", unsafe_allow_html=True)

options = movies[["id", "title"]].copy()
options["id"] = options["id"].astype(str)
options["title"] = options["title"].fillna("").astype(str)
options = options[options["title"].str.strip().ne("") & options["id"].str.strip().ne("")].drop_duplicates("id")
dupes = options["title"].duplicated(keep=False)
options["display"] = options["title"]
options.loc[dupes, "display"] = options.loc[dupes, "title"] + " · " + options.loc[dupes, "id"]
options = options.sort_values(["title", "id"])

st.markdown('<div id="discover"></div>', unsafe_allow_html=True)
st.markdown("### 🔎 Find Your Movie")
st.caption("Search for a film, choose the exact match, and customize your recommendations.")
c1, c2 = st.columns([1, 1])
with c1:
    search = st.text_input("Search for a movie", placeholder="Type at least 3 letters, e.g. Toy Story")
selected_id = None
with c2:
    st.markdown("**🎬 Choose a movie from the search results**")
    if len(search.strip()) >= 3:
        found = options[options["title"].str.contains(search.strip(), case=False, regex=False, na=False)].head(100)
        if not found.empty:
            choices = dict(zip(found["display"], found["id"]))
            picked = st.selectbox("Choose a movie", list(choices), index=None, placeholder="Select the exact movie...", label_visibility="collapsed")
            if picked is not None:
                selected_id = choices[picked]
        else:
            st.info("No matching movies found. Try another title.")
    else:
        st.selectbox("Choose a movie", [], disabled=True, placeholder="Search above to see results", label_visibility="collapsed")
c3, c4 = st.columns([1.5, 1])
with c3:
    top_n = st.slider("🎚️ Number of recommendations", 4, 12, 8)
with c4:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    run = st.button("🎁 Recommend Movies", type="primary", use_container_width=True)
# Popular row uses review counts from the existing dataset.
st.markdown('<div id="popular"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading"><h3>🍿 Popular Movies</h3></div>', unsafe_allow_html=True)
popular_source = movies.copy()
popular_source["_review_count_numeric"] = pd.to_numeric(popular_source["reviewCount"], errors="coerce").fillna(0)
popular_source = popular_source.sort_values(
    ["_review_count_numeric", "title"], ascending=[False, True], kind="mergesort"
).drop_duplicates(subset=["id"]).head(8)

popular_api_key = get_tmdb_api_key()
popular_cards = []
for _, popular_movie in popular_source.iterrows():
    popular_title = escape(str(popular_movie["title"]))
    popular_poster = get_poster(str(popular_movie["title"]), popular_api_key) if popular_api_key and requests is not None else None
    if popular_poster:
        popular_poster_html = f'<img class="carousel-poster" src="{escape(popular_poster, quote=True)}" alt="Poster for {popular_title}" loading="lazy">'
    else:
        popular_poster_html = '<div class="carousel-poster poster-missing" style="display:flex;align-items:center;justify-content:center">POSTER UNAVAILABLE</div>'
    popular_cards.append(textwrap.dedent(f"""
        <div class="carousel-card">
          {popular_poster_html}
          <div class="carousel-title">{popular_title}</div>
          <span class="score">Reviews · {int(popular_movie["_review_count_numeric"])}</span>
        </div>
    """).strip())

if popular_cards:
    st.markdown("<div class=\'carousel\'>" + "".join(popular_cards) + "</div>", unsafe_allow_html=True)
else:
    st.caption("Popular movie data is not available yet.")

def recommend(movie_id, top_n):
    movie_id = str(movie_id)
    matched = movies.index[movies["id"].astype(str) == movie_id].tolist()
    if not matched:
        return None
    idx = matched[0]
    gs = cosine_similarity(genre_matrix[idx], genre_matrix).ravel()
    ds = cosine_similarity(director_matrix[idx], director_matrix).ravel()
    ls = cosine_similarity(language_matrix[idx], language_matrix).ravel()
    scores = 0.65 * gs + 0.15 * ds + 0.10 * ls
    sentiment = pd.to_numeric(movies["smoothed_sentiment"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(sentiment)
    scores[valid] += 0.10 * sentiment[valid]
    ranking = pd.DataFrame({"row": np.arange(n), "score": scores, "id": movies["id"].astype(str).to_numpy()})
    ranking = ranking[(ranking["row"] != idx) & (ranking["id"] != movie_id)]
    ranking = ranking.sort_values(["score", "id"], ascending=[False, True], kind="mergesort").head(top_n)
    out = movies.iloc[ranking["row"].to_numpy()][["id","title","genre","director","originalLanguage","reviewCount","smoothed_sentiment"]].copy()
    out["Recommendation Score"] = [round(float(scores[i]), 4) for i in ranking["row"]]
    out = out.rename(columns={"title":"Movie","genre":"Genre","director":"Director","originalLanguage":"Language","reviewCount":"Review Count","smoothed_sentiment":"Critic Sentiment"})
    out["Critic Sentiment"] = pd.to_numeric(out["Critic Sentiment"], errors="coerce").round(4)
    return out.reset_index(drop=True)

if run:
    if selected_id is None:
        st.warning("Please search for a movie and select it from the results first.")
    else:
        chosen = movies[movies["id"].astype(str) == selected_id].iloc[0]
        title = str(chosen["title"])
        with st.spinner("Finding your next favourite films..."):
            recs = recommend(selected_id, top_n)
        if recs is None or recs.empty:
            st.warning("No recommendations are available for this selection.")
        else:
            st.markdown("---")
            st.markdown(f"## ✨ Because you liked <span style='color:#ff4059'>{title}</span>", unsafe_allow_html=True)
            st.caption("Here are some similar movies you might enjoy.")
            api_key = get_tmdb_api_key()
            if not api_key:
                st.info("Movie posters are disabled because TMDB_API_KEY is not configured. Add it in Streamlit Community Cloud → App settings → Secrets to load real posters.")
            elif requests is None:
                st.warning("The requests package is missing. Add requests to requirements.txt and redeploy to enable posters.")
            card_html = []
            for _, movie in recs.iterrows():
                poster = get_poster(movie["Movie"], api_key) if api_key and requests is not None else None
                sentiment_value = movie["Critic Sentiment"]
                sentiment_text = f"{float(sentiment_value):.3f}" if pd.notna(sentiment_value) else "N/A"
                movie_title = escape(str(movie["Movie"]))
                genre_text = escape(str(movie["Genre"])) if pd.notna(movie["Genre"]) else "Genre unavailable"
                director_text = escape(str(movie["Director"])) if pd.notna(movie["Director"]) else "Director unavailable"
                language_text = escape(str(movie["Language"])) if pd.notna(movie["Language"]) else "Language unavailable"
                review_text = escape(str(movie["Review Count"])) if pd.notna(movie["Review Count"]) else "N/A"
                if poster:
                    poster_html = f'<img class="carousel-poster" src="{escape(poster, quote=True)}" alt="Poster for {movie_title}" loading="lazy">'
                else:
                    poster_html = '<div class="carousel-poster" style="display:flex;align-items:center;justify-content:center;color:#777;font-size:11px;letter-spacing:1px">POSTER UNAVAILABLE</div>'
                card_html.append(textwrap.dedent(f"""
                    <div class="carousel-card">
                      {poster_html}
                      <div class="carousel-title">{movie_title}</div>
                      <span class="score">Match score · {float(movie["Recommendation Score"]):.3f}</span>
                      <div class="carousel-meta">Genre · {genre_text}</div>
                      <div class="carousel-meta">Director · {director_text}</div>
                      <div class="carousel-meta">Language · {language_text}</div>
                      <div class="carousel-meta">Reviews · {review_text} &nbsp; Sentiment · {sentiment_text}</div>
                    </div>
                """).strip())
            carousel_html = "\n<div class=\"carousel\">\n" + "\n".join(card_html) + "\n</div>"
            st.markdown(carousel_html, unsafe_allow_html=True)
            st.markdown("## ☷ All Recommended Movies")
            table = recs.copy()
            table.insert(0, "Rank", np.arange(1, len(table) + 1))
            table = table.drop(columns=["id"]).rename(columns={"Recommendation Score":"Match Score"})
            st.dataframe(table, use_container_width=True, hide_index=True, column_config={
                "Rank": st.column_config.NumberColumn("№", width="small"),
                "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0.0, max_value=1.0, format="%.4f"),
                "Critic Sentiment": st.column_config.NumberColumn("Critic Sentiment", format="%.4f"),
            })

st.markdown("<div class='footer' id='about'>Movie Recommendation System · Recommendations use content similarity and available critic-review sentiment. Movie images and metadata provided by TMDB where available.</div>", unsafe_allow_html=True)
