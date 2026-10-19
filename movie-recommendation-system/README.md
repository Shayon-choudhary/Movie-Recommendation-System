# CineMatch — Movie Recommendation System

A beginner-friendly movie recommender built with Python, pandas, scikit-learn, and Streamlit. It uses TF-IDF and cosine similarity to suggest movies with similar genres and descriptions.

## Requirements
- Python 3.10 or newer recommended
- Libraries listed in `requirements.txt`

## Run on Windows (VS Code)

1. Extract the ZIP file.
2. Open the extracted `movie-recommendation-system` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

5. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

6. Install the libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

8. Open the local URL Streamlit prints in the terminal, usually `http://localhost:8501`.

## Dataset
`movies.csv` contains a small sample dataset so the project works offline after installation. Ratings are sample values, not live or verified ratings.

## How it works
- pandas loads and validates movie data.
- TF-IDF converts genre and description text into numeric features.
- Cosine similarity ranks similar movies.
- Streamlit provides the user interface.

This is content-based filtering; it does not learn preferences from actual user behavior.
