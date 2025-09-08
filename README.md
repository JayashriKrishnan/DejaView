# DejaView

Capture web page paragraphs via a Chrome extension and search them with a Flask backend using embeddings and a local Chroma vector store.

## Project Structure
- `Deja_extension/` Chrome extension (MV3)
- `dejaview_backend/` Flask backend with SQLite and Chroma

## Prerequisites
- Python 3.10+
- Google Chrome
- Git

## Backend Setup
```bash
cd dejaview_backend
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Backend runs on `http://localhost:5000` and creates `dejaview.db` and `chroma_store/` locally.

## Chrome Extension Setup
1. Open Chrome → chrome://extensions
2. Enable Developer Mode
3. Load Unpacked → select `Deja_extension/`
4. The extension communicates with `http://localhost:5000`.

## Development
- Run backend: `python app.py`
- Update pinned deps: `pip freeze > requirements.txt`

## Git: How to Push
Initial push:
```bash
# From project root
git init
git add .
git commit -m "Initial commit: DejaView"
# Create repo on GitHub, then set remote:
git branch -M main
git remote add origin https://github.com/<your-username>/DejaView.git
git push -u origin main
```
Subsequent changes:
```bash
git add -A
git commit -m "Describe your change"
git push
```

## Notes
- Large/local data (vector store, SQLite DBs, virtual envs, logs) are ignored by `.gitignore`.
- On a new machine: `pip install -r requirements.txt`, run the backend, load the extension.

## Troubleshooting
- If imports fail: activate the venv and reinstall `pip install -r requirements.txt`.
- Embedding model downloads are cached by `sentence-transformers` in your user profile.
