# CineMatch — Movie Recommendation System 🎬

CineMatch recommends similar movies using content-based filtering with TF-IDF and cosine similarity.

## Features
- Searchable movie selection
- Genre and minimum-rating filters
- Configurable recommendation count
- Similarity scores and movie descriptions
- Dataset validation and helpful error messages
- Streamlit interface with a responsive two-column recommendation layout

## Run locally (Windows / VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `movies.csv`, `requirements.txt`, and `README.md` to GitHub.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select your repository and branch.
4. Set **Main file path** to `app.py` if these files are at the repository root. If they are inside a folder, use `movie-recommendation-system/app.py`.
5. Deploy.

**Important:** `movies.csv` must be in the same folder as `app.py`. The app resolves the CSV path relative to the Python file, so it works even if Streamlit starts from another working directory.

## Dataset format

The CSV must include these columns:

- `title`
- `genre`
- `rating` (0–10)
- `description`

The included CSV contains 20,000 movie records from the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/). Ratings are average user ratings in that dataset. MovieLens does not include plot summaries, so the `description` values are generated from each movie's genre tags and should not be treated as factual plot descriptions.

## How it works

Pandas loads and validates the dataset. Scikit-learn's TF-IDF vectorizer converts movie genre and description text into sparse numerical features, and cosine similarity ranks the movies by textual similarity without constructing a dense all-movies similarity matrix. It is a content-based recommender, not a model trained on the current user's viewing history.
