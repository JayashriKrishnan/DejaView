# app.py
import os
import json
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; app still works without it
    pass

_openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---- Config ----
DB_FILE = "dejaview.db"
app = Flask(__name__)
CORS(app)

# Load embedding model (small, fast)
model=SentenceTransformer("all-MiniLM-L6-v2")


# Setup Chroma client (local persistent store)
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("paragraph_embeddings")

# ---- DB init ----
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        title TEXT,
        timestamp TEXT
      )
    """)
    c.execute("""
      CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_id INTEGER,
        paragraph_index INTEGER,
        text TEXT,
        FOREIGN KEY(page_id) REFERENCES pages(id)
      )
    """)
    conn.commit()

    # Add summary columns if missing
    try:
        c.execute("PRAGMA table_info(pages)")
        cols = [row[1] for row in c.fetchall()]
        if "summary" not in cols:
            c.execute("ALTER TABLE pages ADD COLUMN summary TEXT")
        if "summary_model" not in cols:
            c.execute("ALTER TABLE pages ADD COLUMN summary_model TEXT")
        conn.commit()
    except Exception:
        # Best-effort migration; continue without breaking app
        pass
    conn.close()

init_db()

# ---- Endpoints ----
@app.route("/capture", methods=["POST"])
def capture():
    payload = request.get_json(force=True)
    url = payload.get("url")
    title = payload.get("title")
    ist = ZoneInfo("Asia/Kolkata")
    timestamp = payload.get("timestamp") or datetime.now(ist).isoformat()
    paragraphs = payload.get("paragraphs", [])

    if not url or not paragraphs:
        return jsonify({"status": "error", "message": "Missing url or paragraphs"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO pages (url, title, timestamp) VALUES (?, ?, ?)", (url, title, timestamp))
    page_id = c.lastrowid

    # Store text in SQLite, embeddings in Chroma
    texts = [p.get("text", "") for p in paragraphs]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    paragraph_ids = []
    for idx, (p, emb) in enumerate(zip(paragraphs, embeddings)):
        c.execute("""
            INSERT INTO paragraphs (page_id, paragraph_index, text)
            VALUES (?, ?, ?)
        """, (page_id, p.get("index", idx), p.get("text", "")))
        paragraph_id = c.lastrowid
        paragraph_ids.append(str(paragraph_id))  # Chroma IDs must be strings

    conn.commit()
    conn.close()

    # Insert into Chroma (id, embedding, metadata)
    collection.add(
        ids=paragraph_ids,
        embeddings=embeddings,
        metadatas=[{"url": url, "title": title, "timestamp": timestamp, "paragraph_index": p.get("index", idx)} for idx, p in enumerate(paragraphs)],
        documents=texts
    )

    return jsonify({"status": "success", "stored_paragraphs": len(paragraphs)}), 200


@app.route("/search", methods=["POST"])
def search():
    body = request.get_json(force=True)
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))

    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    q_emb = model.encode([query], convert_to_numpy=True).tolist()[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    # Format results
    formatted = []
    for ids, docs, metas in zip(results["ids"], results["documents"], results["metadatas"]):
        for pid, doc, meta in zip(ids, docs, metas):
            formatted.append({
                "paragraph_id": pid,
                "text": doc,
                "url": meta.get("url"),
                "title": meta.get("title"),
                "timestamp": meta.get("timestamp"),
                "paragraph_index": meta.get("paragraph_index"),
            })

    return jsonify(formatted), 200


@app.route("/all", methods=["GET"])
def all_recent():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, url, title, timestamp FROM pages ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id":r[0],"url":r[1],"title":r[2],"timestamp":r[3]} for r in rows])


## NOTE: Server start moved to end of file so all routes register first

# ---- Summarization Utilities and Endpoints (Phase 3) ----

def _summarize_with_openai(paragraphs: List[str], max_words: int = 150) -> str:
    if not _openai_client:
        raise RuntimeError("OpenAI client not available")
    # Truncate context size
    joined = "\n\n".join(paragraphs)
    if len(joined) > 8000:
        joined = joined[:8000]
    system_prompt = (
        "You are a helpful assistant that produces concise, faithful summaries. "
        f"Limit to about {max_words} words."
    )
    user_prompt = (
        "Summarize the following web page content for later recall.\n\n" + joined
    )
    try:
        resp = _openai_client.chat.completions.create(
            model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI summarization failed: {e}")


def _summarize_locally(paragraphs: List[str], max_words: int = 150) -> str:
    # Fallback: take first few sentences/paragraphs and trim tokens
    if not paragraphs:
        return ""
    joined = " ".join(paragraphs)
    words = joined.split()
    trimmed = " ".join(words[:max_words])
    return trimmed + ("..." if len(words) > max_words else "")


def summarize_texts(paragraphs: List[str], max_words: int = 150) -> str:
    if OPENAI_API_KEY and _openai_client is not None:
        try:
            return _summarize_with_openai(paragraphs, max_words=max_words)
        except Exception:
            return _summarize_locally(paragraphs, max_words=max_words)
    return _summarize_locally(paragraphs, max_words=max_words)


@app.route("/page/<int:page_id>", methods=["GET"])
def get_page(page_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, url, title, timestamp, summary, summary_model FROM pages WHERE id = ?", (page_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "message": "page not found"}), 404
    c.execute("SELECT id, paragraph_index, text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (page_id,))
    paras = c.fetchall()
    conn.close()
    return jsonify({
        "id": row[0],
        "url": row[1],
        "title": row[2],
        "timestamp": row[3],
        "summary": row[4],
        "summary_model": row[5],
        "paragraphs": [
            {"id": p[0], "paragraph_index": p[1], "text": p[2]} for p in paras
        ]
    })


@app.route("/summarize_page", methods=["POST"])
def summarize_page():
    body = request.get_json(force=True)
    page_id = body.get("page_id")
    if not page_id:
        return jsonify({"status": "error", "message": "page_id required"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT title FROM pages WHERE id = ?", (page_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "message": "page not found"}), 404
    title = row[0] or ""
    c.execute("SELECT text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (page_id,))
    texts = [r[0] for r in c.fetchall()]
    conn.close()

    # Build context with title + paragraphs
    context = [title] + texts if title else texts
    summary = summarize_texts(context, max_words=int(os.getenv("SUMMARY_MAX_WORDS", "150")))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE pages SET summary = ?, summary_model = ? WHERE id = ?",
        (summary, os.getenv("OPENAI_SUMMARY_MODEL", "local" if _openai_client is None else "gpt-4o-mini"), page_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "page_id": page_id, "summary": summary})


@app.route("/summarize_all", methods=["POST"])
def summarize_all_pages():
    limit = int(request.args.get("limit", "50"))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title FROM pages ORDER BY id DESC LIMIT ?", (limit,))
    pages = c.fetchall()
    conn.close()

    results = []
    for pid, title in pages:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (pid,))
        texts = [r[0] for r in c.fetchall()]
        conn.close()
        context = [title] + texts if title else texts
        
        summary = summarize_texts(context, max_words=int(os.getenv("SUMMARY_MAX_WORDS", "150")))
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "UPDATE pages SET summary = ?, summary_model = ? WHERE id = ?",
            (summary, os.getenv("OPENAI_SUMMARY_MODEL", "local" if _openai_client is None else "gpt-4o-mini"), pid)
        )
        conn.commit()
        conn.close()
        results.append({"page_id": pid, "summary": summary})
    return jsonify({"status": "success", "updated": len(results), "items": results})


@app.route("/summarize_query", methods=["POST"])
def summarize_query():
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    top_k = int(body.get("top_k", 5))
    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    # Embed query, retrieve top paragraphs (reuse existing model/collection)
    q_emb = model.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    texts: List[str] = []
    used_paragraph_ids: List[str] = []
    if results and results.get("documents"):
        for ids, docs in zip(results.get("ids", []), results.get("documents", [])):
            for pid, doc in zip(ids, docs):
                used_paragraph_ids.append(pid)
                texts.append(doc)

    if not texts:
        return jsonify({"status": "success", "summary": "", "used_paragraph_ids": []})

    summary = summarize_texts(texts, max_words=int(os.getenv("SUMMARY_MAX_WORDS", "150")))
    return jsonify({
        "status": "success",
        "summary": summary,
        "used_paragraph_ids": used_paragraph_ids
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
