from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Always load the dataset relative to this file, not the current working directory.
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "movies_20000.csv"

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
             background: linear-gradient(120deg, #24133d 0%, #171b32 55%, #102c38 100%);
             border: 1px solid rgba(255,255,255,.12); margin-bottom: 1.3rem;}
      .hero h1 {margin: 0; font-size: 2.6rem;}
      .hero p {margin: .5rem 0 0 0; color: #e0e7ff;}
      [data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 1rem;
                                border-radius: 12px; border: 1px solid rgba(128,128,128,.18);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎬 CineMatch</h1>
      <p>Discover your next favorite movie with content-based machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_movies(csv_path: str, modified_time: float) -> pd.DataFrame:
    """Load and validate the dataset. The timestamp refreshes cache after file changes."""
    del modified_time
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    required = {"title", "genre", "rating", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The CSV is missing these required column(s): " + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["title", "genre", "description"]).copy()
    for column in ["title", "genre", "description"]:
        df[column] = df[column].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[
        (df["title"] != "")
        & (df["genre"] != "")
        & (df["description"] != "")
        & df["rating"].between(0, 10)
    ]
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Please provide at least two valid movies in movies.csv.")
    return df

@st.cache_data(show_spinner=False)
def build_similarity(data: pd.DataFrame):
    features = (
        data["genre"].fillna("")
        + " "
        + data["genre"].fillna("")
        + " "
        + data["description"].fillna("")
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    return vectorizer.fit_transform(features)

def recommend_movies(title, data, similarity, selected_genre, min_rating, limit):
    selected_indices = data.index[data["title"] == title].tolist()
    if not selected_indices:
        return data.iloc[0:0].copy()

    selected_index = selected_indices[0]
    scores = cosine_similarity(similarity[selected_index], similarity).ravel()
    ranked_indices = sorted(
        (i for i in range(len(data)) if i != selected_index),
        key=lambda i: scores[i],
        reverse=True,
    )
    results = data.iloc[ranked_indices].copy()
    results["similarity"] = [float(scores[i]) for i in ranked_indices]

    if selected_genre != "All":
        results = results[results["genre"] == selected_genre]
    results = results[results["rating"] >= min_rating]
    return results.head(limit)

if not CSV_PATH.is_file():
    st.error("The dataset file `movies_20000.csv` is missing.")
    st.markdown(
        f"**Expected location:** `{CSV_PATH}`  \n"
        "Upload `movies_20000.csv` into the same GitHub folder as `app.py`, then redeploy."
    )
    st.stop()

try:
    movies = load_movies(str(CSV_PATH), CSV_PATH.stat().st_mtime)
    similarity = build_similarity(movies)
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not load the movie dataset: {error}")
    st.info("Check that movies_20000.csv is a valid CSV with title, genre, rating, and description columns.")
    st.stop()

total_movies = len(movies)
genres = sorted(movies["genre"].unique().tolist())

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Movies available", total_movies)
metric2.metric("Genres", len(genres))
metric3.metric("Highest rating", f"{movies['rating'].max():.1f}/10")

st.sidebar.header("🎯 Your preferences")
st.sidebar.caption("Choose a movie and tune the recommendations.")
selected_genre = st.sidebar.selectbox("Filter recommendations by genre", ["All"] + genres)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
max_count = min(10, total_movies - 1)
count = st.sidebar.slider("Number of recommendations", 1, max_count, min(5, max_count))
st.sidebar.divider()
st.sidebar.caption("Similarity is calculated from movie genres and descriptions.")

movie_title = st.selectbox("🍿 Choose a movie you already like", sorted(movies["title"].tolist()))
selected = movies.loc[movies["title"] == movie_title].iloc[0]

with st.container(border=True):
    left, right = st.columns([3, 1])
    with left:
        st.subheader(selected["title"])
        st.write(selected["description"])
        st.caption(f"Genre: {selected['genre']}")
    with right:
        st.metric("Rating", f"{selected['rating']:.1f}/10")

if st.button("✨ Find similar movies", type="primary", use_container_width=True):
    recommendations = recommend_movies(
        movie_title, movies, similarity, selected_genre, min_rating, count
    )
    st.subheader("🎞️ Recommended for you")
    if recommendations.empty:
        st.warning("No movies match these filters. Try choosing 'All' genres or lowering the minimum rating.")
    else:
        for start in range(0, len(recommendations), 2):
            columns = st.columns(2)
            for column, (_, movie) in zip(columns, recommendations.iloc[start:start + 2].iterrows()):
                with column:
                    with st.container(border=True):
                        st.subheader(movie["title"])
                        st.caption(f"{movie['genre']}  ·  ⭐ {movie['rating']:.1f}/10")
                        st.write(movie["description"])
                        st.progress(
                            min(1.0, max(0.0, float(movie["similarity"]))),
                            text=f"Content similarity: {movie['similarity']:.0%}",
                        )

st.divider()
st.caption(
    f"CineMatch · {total_movies} movies · Built with Streamlit, pandas and scikit-learn. "
    "Ratings are average MovieLens user ratings; descriptions are generated from genre tags."
)
