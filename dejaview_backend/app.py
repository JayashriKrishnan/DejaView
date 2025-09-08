# app.py
import os
import json
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

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
    conn.close()

init_db()

# ---- Endpoints ----
@app.route("/capture", methods=["POST"])
def capture():
    payload = request.get_json(force=True)
    url = payload.get("url")
    title = payload.get("title")
    timestamp = payload.get("timestamp") or datetime.utcnow().isoformat()
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


if __name__ == "__main__":
    app.run(port=5000, debug=True)
