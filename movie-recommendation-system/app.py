import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 CineMatch")
st.subheader("Discover your next favorite movie")
st.write("Choose a movie you like and find similar films using content-based filtering.")

@st.cache_data
def load_movies():
    df = pd.read_csv("movies.csv")
    required = {"title", "genre", "rating", "description"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain title, genre, rating, description.")
    df = df.dropna(subset=["title", "genre", "description"]).copy()
    for column in ["title", "genre", "description"]:
        df[column] = df[column].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The movie dataset is empty or invalid.")
    return df

@st.cache_data
def build_similarity(data):
    combined = data["genre"] + " " + data["genre"] + " " + data["description"]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(combined)
    return cosine_similarity(matrix)

def recommend_movies(title, data, similarity, genre, min_rating, limit):
    matches = data.index[data["title"] == title].tolist()
    if not matches:
        return data.iloc[0:0].copy()
    movie_index = matches[0]
    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)
    ranked_indices = [index for index, _ in scores if index != movie_index]
    results = data.iloc[ranked_indices].copy()
    results["similarity"] = [similarity[movie_index, index] for index in ranked_indices]
    if genre != "All":
        results = results[results["genre"] == genre]
    results = results[results["rating"] >= min_rating]
    return results.head(limit)

try:
    movies = load_movies()
    similarity = build_similarity(movies)
    st.sidebar.header("🎯 Your preferences")
    genres = ["All"] + sorted(movies["genre"].unique().tolist())
    selected_genre = st.sidebar.selectbox("Filter by genre", genres)
    min_rating = st.sidebar.slider("Minimum movie rating", 0.0, 10.0, 0.0, 0.5)
    movie_title = st.selectbox("Select a movie you like", sorted(movies["title"].tolist()))
    count = st.slider("Number of recommendations", 1, min(10, max(1, len(movies) - 1)), min(5, max(1, len(movies) - 1)))

    if st.button("✨ Recommend Movies", type="primary"):
        selected = movies[movies["title"] == movie_title].iloc[0]
        st.markdown("### Your selected movie")
        st.write(f"**{selected['title']}**")
        st.write(f"Genre: {selected['genre']}  |  ⭐ Rating: {selected['rating']}/10")
        st.write(selected["description"])
        recommendations = recommend_movies(movie_title, movies, similarity, selected_genre, min_rating, count)
        st.markdown("### 🍿 Recommended for you")
        if recommendations.empty:
            st.info("No movies match these filters. Try another genre or lower the minimum rating.")
        else:
            for _, movie in recommendations.iterrows():
                with st.container(border=True):
                    st.subheader(movie["title"])
                    st.caption(f"Genre: {movie['genre']} | Rating: {movie['rating']}/10")
                    st.write(movie["description"])
                    st.progress(float(movie["similarity"]), text=f"Content similarity: {movie['similarity']:.0%}")
    st.divider()
    st.caption(f"Movie database: {len(movies)} sample movies | Powered by TF-IDF and cosine similarity")
except FileNotFoundError:
    st.error("movies.csv was not found. Place it in the same folder as app.py.")
except (ValueError, KeyError) as error:
    st.error(f"Dataset error: {error}")
