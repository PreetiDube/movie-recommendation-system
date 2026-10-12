import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

try:
    import requests
except ImportError:
    requests = None

st.set_page_config(page_title="Movie Discovery", page_icon="🎬", layout="wide")
MODEL_PATH = Path(__file__).parent / "recommendation_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, api_key):
    if not api_key or requests is None:
        return None
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": api_key, "query": str(title)}, timeout=4)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + results[0]["poster_path"]
    except Exception:
        pass
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
.stApp{background:radial-gradient(ellipse at 85% 0%,#35131f 0,transparent 32%),radial-gradient(ellipse at 5% 35%,#12243b 0,transparent 35%),#080d15;color:#f5f7fb}
.block-container{max-width:1500px;padding-top:1.3rem}
.hero{padding:30px;border-radius:20px;border:1px solid #293343;background:linear-gradient(110deg,#0c1421,#111b2a 60%,#401c2a);margin-bottom:22px}
.eyebrow{color:#ff4059;letter-spacing:4px;font-size:12px;font-weight:800;text-transform:uppercase}
.hero-title{font-size:clamp(30px,4vw,48px);font-weight:850;line-height:1.15;margin:8px 0}
.hero-title span{color:#ff4059}.hero-sub{color:#bdc7d6;font-size:16px}
.panel{padding:20px;border:1px solid #303a4a;border-radius:17px;background:linear-gradient(145deg,#151e2d,#0c121c);margin:10px 0 22px}
.movie-card{border:1px solid #2d3747;border-radius:15px;background:linear-gradient(145deg,#182231,#0c121c);padding:12px;min-height:280px;margin-bottom:14px}
.poster{height:165px;border-radius:10px;background:radial-gradient(circle at 25% 20%,#a43c52,transparent 38%),linear-gradient(135deg,#263b56,#111722 60%,#512031);display:flex;align-items:center;justify-content:center;font-size:48px;margin-bottom:12px;overflow:hidden}
.poster img{width:100%;height:165px;object-fit:cover}
.movie-name{font-size:16px;font-weight:800;margin-bottom:6px}
.score{display:inline-block;border:1px solid #b82b43;background:#351520;color:#ff7181;border-radius:9px;padding:4px 8px;font-size:12px;font-weight:800}
.meta{color:#b4bfd0;font-size:12px;line-height:1.5;margin-top:7px}
div.stButton>button[kind="primary"]{border:0;border-radius:11px;min-height:47px;font-weight:800;background:linear-gradient(100deg,#ed183b,#ff4d60);color:white}
div[data-testid="stTextInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#1b2433;border-radius:10px}
div[data-testid="stDataFrame"]{border:1px solid #2d3747;border-radius:12px;overflow:hidden}
.footer{color:#8290a5;font-size:12px;text-align:center;margin-top:30px}
</style>
<div class="hero">
 <div class="eyebrow">Your next favourite film</div>
 <div class="hero-title">🎬 Movie <span>Recommendation System</span></div>
 <div class="hero-sub">Discover films using genre, director, language, and available critic-review sentiment.</div>
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

st.markdown('<div class="panel">', unsafe_allow_html=True)
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
st.markdown("</div>", unsafe_allow_html=True)

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
            api_key = st.secrets.get("TMDB_API_KEY", "") if hasattr(st, "secrets") else ""
            for start in range(0, len(recs), 4):
                cols = st.columns(4, gap="medium")
                for col, (_, movie) in zip(cols, recs.iloc[start:start+4].iterrows()):
                    with col:
                        poster = get_poster(movie["Movie"], api_key)
                        sentiment_value = movie["Critic Sentiment"]
                        sentiment_text = f"{float(sentiment_value):.3f}" if pd.notna(sentiment_value) else "N/A"
                        if poster:
                            poster_html = f'<div class="poster"><img src="{poster}" alt="Movie poster"></div>'
                        else:
                            poster_html = '<div class="poster">🎬</div>'
                        st.markdown(f"""
                        <div class="movie-card">
                          {poster_html}
                          <div class="movie-name">{movie["Movie"]}</div>
                          <span class="score">{float(movie["Recommendation Score"]):.4f}</span>
                          <div class="meta">🎭 {movie["Genre"] if pd.notna(movie["Genre"]) else "Genre unavailable"}</div>
                          <div class="meta">🎬 {movie["Director"] if pd.notna(movie["Director"]) else "Director unavailable"}</div>
                          <div class="meta">🌐 {movie["Language"] if pd.notna(movie["Language"]) else "Language unavailable"}</div>
                          <div class="meta">⭐ Reviews: {movie["Review Count"] if pd.notna(movie["Review Count"]) else "N/A"} · Sentiment: {sentiment_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("## ☷ All Recommended Movies")
            table = recs.copy()
            table.insert(0, "Rank", np.arange(1, len(table) + 1))
            table = table.drop(columns=["id"]).rename(columns={"Recommendation Score":"Match Score"})
            st.dataframe(table, use_container_width=True, hide_index=True, column_config={
                "Rank": st.column_config.NumberColumn("№", width="small"),
                "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0.0, max_value=1.0, format="%.4f"),
                "Critic Sentiment": st.column_config.NumberColumn("Critic Sentiment", format="%.4f"),
            })

st.markdown("<div class='footer'>Movie Discovery · Recommendations use content similarity and available critic-review sentiment.</div>", unsafe_allow_html=True)
