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

    # Create FTS5 index for paragraphs to support BM25 keyword search
    try:
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts USING fts5(text, content='paragraphs', content_rowid='id')")
        # Backfill any missing FTS rows (one-time, idempotent-ish)
        c.execute("INSERT INTO paragraphs_fts(rowid, text) SELECT id, text FROM paragraphs WHERE id NOT IN (SELECT rowid FROM paragraphs_fts)")
        conn.commit()
    except Exception:
        # If FTS5 not available, skip; hybrid search will degrade gracefully
        pass
    conn.close()

init_db()

# Auto-backfill missing data on startup (only if needed)
def auto_backfill_on_startup():
    """Ensure all SQLite data is also in ChromaDB - only run if ChromaDB is empty"""
    try:
        # Check if ChromaDB has any data
        existing_data = collection.get(include=['metadatas'])
        if existing_data.get('metadatas') and len(existing_data['metadatas']) > 0:
            print("ChromaDB already has data, skipping auto-backfill")
            return
        
        print("ChromaDB is empty, running auto-backfill...")
        from auto_backfill import auto_backfill
        auto_backfill()
    except Exception as e:
        print(f"Auto-backfill failed: {e}")

# Run backfill in background on startup (only if needed)
import threading
backfill_thread = threading.Thread(target=auto_backfill_on_startup, daemon=True)
backfill_thread.start()

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

    # Insert into Chroma (id, embedding, metadata) with error handling
    chroma_success = False
    chroma_error = None
    for attempt in range(3):
        try:
            collection.add(
                ids=paragraph_ids,
                embeddings=embeddings,
                metadatas=[{
                    "url": url, 
                    "title": title, 
                    "timestamp": timestamp, 
                    "paragraph_index": p.get("index", idx),
                    "page_id": page_id
                } for idx, p in enumerate(paragraphs)],
                documents=texts
            )
            chroma_success = True
            print(f"Successfully stored {len(paragraphs)} paragraphs in ChromaDB for page {page_id}")
            break
        except Exception as e:
            chroma_error = str(e)
            print(f"Error storing in ChromaDB (attempt {attempt+1}): {e}")
    if not chroma_success:
        print(f"Failed to store in ChromaDB after 3 attempts: {chroma_error}")

    return jsonify({"status": "success", "stored_paragraphs": len(paragraphs)}), 200


@app.route("/search", methods=["POST"])
def search():
    body = request.get_json(force=True)
    query = body.get("query", "")
    top_k = int(body.get("top_k", 20))
    min_similarity = float(body.get("min_similarity", -0.5))  # Lower default threshold for negative similarities
    from_date: Optional[str] = body.get("from_date")  # ISO date prefix e.g. "2025-10-01"
    to_date: Optional[str] = body.get("to_date")
    prefer_recent: bool = bool(body.get("prefer_recent", True))
    recency_boost: float = float(body.get("recency_boost", 0.5))  # 0..1 reasonable

    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    # Prepare date bounds (will filter results in Python)
    lower = f"{from_date}T00:00:00" if from_date and "T" not in from_date else from_date
    upper = f"{to_date}T23:59:59" if to_date and "T" not in to_date else to_date

    # Simple query expansion for synonyms/acronyms (helps AI vs Artificial Intelligence, ML vs Machine Learning)
    synonyms_map = {
        "artificial intelligence": ["ai"],
        "ai": ["artificial intelligence"],
        "machine learning": ["ml"],
        "ml": ["machine learning"],
        "natural language processing": ["nlp"],
        "nlp": ["natural language processing"],
        "database": ["db"],
        "sql": ["structured query language"],
    }
    expanded_queries: List[str] = [query]
    q_lower = query.lower()
    for key, syns in synonyms_map.items():
        if key in q_lower and syns:
            expanded_queries.extend(syns)

    # Average the embeddings of expanded terms for a more robust query vector
    q_vecs = model.encode(expanded_queries, convert_to_numpy=True)
    q_emb = np.mean(q_vecs, axis=0).tolist()

    # If date filter present, widen retrieval so post-filtering still has enough hits
    n_fetch = max(top_k * 10, 200) if (from_date or to_date) else top_k
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances"]
    )

    formatted = []
    ids_list = results.get("ids", []) or []
    docs_list = results.get("documents", []) or []
    metas_list = results.get("metadatas", []) or []
    dists_list = results.get("distances", []) or []

    # Helper to parse ISO timestamps uniformly (supports trailing Z and offsets)
    def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    # Compute a recency score with ~7-day half-life
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    half_life_days = 7.0
    ln2 = 0.69314718056
    decay = ln2 / half_life_days

    for ids, docs, metas, dists in zip(ids_list, docs_list, metas_list, dists_list):
        for pid, doc, meta, dist in zip(ids, docs, metas, dists):
            # ChromaDB returns cosine distances, convert to similarity
            # Distance 0 = perfect match (similarity 1), Distance 2 = opposite (similarity -1)
            similarity = 1 - float(dist) if dist is not None else None
            # Date filter in Python (timestamps are ISO strings)
            if lower and meta.get("timestamp") and meta["timestamp"] < lower:
                continue
            if upper and meta.get("timestamp") and meta["timestamp"] > upper:
                continue
            # Don't filter by similarity if it's negative - just use a lower threshold
            if similarity is not None and similarity < max(min_similarity, -0.5):
                continue
            # Recency boost
            combined_score = similarity if similarity is not None else 0.0
            if prefer_recent:
                ts_dt = _parse_iso(meta.get("timestamp"))
                if ts_dt is not None:
                    age_days = max((now - ts_dt).total_seconds(), 0.0) / 86400.0
                    recency_score = np.exp(-decay * age_days)
                    combined_score = similarity * (1.0 + recency_boost * recency_score)

            formatted.append({
                "paragraph_id": pid,
                "text": doc,
                "url": meta.get("url"),
                "title": meta.get("title"),
                "timestamp": meta.get("timestamp"),
                "paragraph_index": meta.get("paragraph_index"),
                "similarity_score": round(similarity, 3) if similarity is not None else None,
                "combined_score": round(combined_score, 3),
            })

    # --- BM25 keyword search via SQLite FTS5 (optional, fast) ---
    bm25_results = []
    try:
        conn2 = sqlite3.connect(DB_FILE)
        c2 = conn2.cursor()
        # Use FTS5 to match query terms; limit breadth, then we’ll merge
        c2.execute("SELECT p.id, p.page_id, p.paragraph_index, p.text FROM paragraphs p JOIN paragraphs_fts f ON p.id = f.rowid WHERE paragraphs_fts MATCH ? LIMIT ?", (query, n_fetch))
        rows = c2.fetchall()
        conn2.close()
        for rid, page_id, pidx, text in rows:
            # Lookup page metadata for URL/title/timestamp
            conn3 = sqlite3.connect(DB_FILE)
            c3 = conn3.cursor()
            c3.execute("SELECT url, title, timestamp FROM pages WHERE id = ?", (page_id,))
            pr = c3.fetchone()
            conn3.close()
            if not pr:
                continue
            url, title, ts = pr
            # Apply date bounds here too
            if lower and ts and ts < lower:
                continue
            if upper and ts and ts > upper:
                continue
            bm25_results.append({
                "paragraph_id": str(rid),
                "text": text,
                "url": url,
                "title": title,
                "timestamp": ts,
                "paragraph_index": pidx,
                "bm25": 1.0  # presence marker; we’ll weight on merge
            })
    except Exception:
        pass

    # Merge semantic + BM25: prefer recency+similarity, but add BM25 hits not already included
    seen_ids = set(r["paragraph_id"] for r in formatted)
    for r in bm25_results:
        if r["paragraph_id"] not in seen_ids:
            # Assign a conservative combined score for BM25-only hits
            r["similarity_score"] = r.get("similarity_score") or 0.4
            r["combined_score"] = r.get("combined_score") or (0.4 + (0.1 if prefer_recent else 0.0))
            formatted.append(r)

    # Trim to requested top_k after filtering (by combined score if recency preferred)
    key_field = "combined_score" if prefer_recent else "similarity_score"
    formatted = sorted(formatted, key=lambda x: x.get(key_field) or 0, reverse=True)[:top_k]

    if not formatted:
        return jsonify({
            "status": "no_results",
            "message": "No content found for the given filters. Try widening the date range or lowering min_similarity.",
            "results": []
        }), 200

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


@app.route("/backfill_recent", methods=["POST"])
def backfill_recent():
    """Re-embed paragraphs from the most recent pages into Chroma.
    Useful if Chroma missed inserts earlier. Idempotent by checking existing ids.
    """
    limit = int(request.args.get("limit", "20"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, url, title, timestamp FROM pages ORDER BY id DESC LIMIT ?", (limit,))
    pages = c.fetchall()

    reinserted = 0
    for page_id, url, title, ts in pages:
        c.execute("SELECT id, paragraph_index, text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (page_id,))
        rows = c.fetchall()
        if not rows:
            continue

        paragraph_ids = [str(r[0]) for r in rows]
        texts = [r[2] for r in rows]

        # Filter out ids that already exist in Chroma
        try:
            existing = collection.get(ids=paragraph_ids)
            existing_ids = set(existing.get("ids", []) or [])
        except Exception:
            existing_ids = set()

        new_ids = []
        new_texts = []
        new_embeddings = []
        new_metas = []
        for pid, text, row in zip(paragraph_ids, texts, rows):
            if pid in existing_ids:
                continue
            new_ids.append(pid)
            new_texts.append(text)
            new_embeddings.append(model.encode([text], convert_to_numpy=True).tolist()[0])
            new_metas.append({
                "url": url,
                "title": title,
                "timestamp": ts,
                "paragraph_index": row[1]
            })

        if new_ids:
            collection.add(
                ids=new_ids,
                embeddings=new_embeddings,
                metadatas=new_metas,
                documents=new_texts
            )
            reinserted += len(new_ids)

    conn.close()
    return jsonify({"status": "success", "reinserted": reinserted})


@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    top_k = int(body.get("top_k", 5))
    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    # Embed query and retrieve top paragraphs
    q_emb = model.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])

    docs = []
    sources = []
    key_takeaway = ""
    filtered_docs = []
    filtered_sources = []
    # Only use paragraphs from 2024 onwards
    min_year = 2024
    if results and results.get("documents"):
        for ids, docs_list, metas_list in zip(results.get("ids", []), results.get("documents", []), results.get("metadatas", [])):
            for pid, doc, meta in zip(ids, docs_list, metas_list):
                ts = meta.get("timestamp")
                year_ok = True
                if ts:
                    try:
                        year = int(ts[:4])
                        year_ok = year >= min_year
                    except Exception:
                        year_ok = True
                if year_ok:
                    filtered_docs.append(doc)
                    src = meta.get("url") or meta.get("title") or "Unknown Source"
                    filtered_sources.append(src)

    if not filtered_docs:
        return jsonify({
            "ai_answer": "No recent relevant content found.",
            "sources": [],
            "key_takeaway": ""
        })

    # Organize answer: join paragraphs with double newlines for clarity
    try:
        answer = summarize_texts(filtered_docs, max_words=180)
        answer = '\n\n'.join([p.strip() for p in answer.split('. ') if p.strip()])
        key_takeaway = answer.split('\n')[0] if answer else ""
        return jsonify({
            "ai_answer": answer,
            "sources": filtered_sources,
            "key_takeaway": key_takeaway
        })
    except Exception as e:
        return jsonify({
            "ai_answer": f"Error generating answer: {str(e)}",
            "sources": filtered_sources,
            "key_takeaway": ""
        })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
