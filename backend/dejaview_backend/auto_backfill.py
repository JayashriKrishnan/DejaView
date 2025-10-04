#!/usr/bin/env python3
"""
Auto-backfill script to ensure all SQLite data is also in ChromaDB.
This will run automatically when the backend starts.
"""
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import os

def auto_backfill():
    print("Starting auto-backfill process...")
    
    # Initialize components
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_store")
    collection = chroma_client.get_or_create_collection("paragraph_embeddings")
    
    # Get all pages from SQLite
    conn = sqlite3.connect("dejaview.db")
    c = conn.cursor()
    c.execute("SELECT id, url, title, timestamp FROM pages ORDER BY id DESC")
    pages = c.fetchall()
    
    print(f"Found {len(pages)} pages in SQLite")
    
    # Check what's already in ChromaDB
    existing_data = collection.get(include=['metadatas'])
    existing_page_ids = set()
    if existing_data.get('metadatas'):
        for meta in existing_data['metadatas']:
            if 'page_id' in meta:
                existing_page_ids.add(meta['page_id'])
    
    print(f"Found {len(existing_page_ids)} pages already in ChromaDB")
    print(f"Existing page IDs: {sorted(existing_page_ids)}")
    
    # Process missing pages
    missing_pages = []
    for page_id, url, title, timestamp in pages:
        if page_id not in existing_page_ids:
            missing_pages.append((page_id, url, title, timestamp))
    
    print(f"Found {len(missing_pages)} pages missing from ChromaDB")
    
    if not missing_pages:
        print("All pages are already in ChromaDB!")
        conn.close()
        return
    
    # Process missing pages
    total_paragraphs = 0
    for page_id, url, title, timestamp in missing_pages:
        print(f"Processing page {page_id}: {title[:50]}...")
        
        # Get paragraphs for this page
        c.execute("SELECT id, paragraph_index, text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (page_id,))
        rows = c.fetchall()
        
        if not rows:
            print(f"  No paragraphs found for page {page_id}")
            continue
        
        # Prepare data for ChromaDB
        paragraph_ids = [str(r[0]) for r in rows]
        texts = [r[2] for r in rows]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        metadatas = [{
            "url": url,
            "title": title,
            "timestamp": timestamp,
            "paragraph_index": r[1],
            "page_id": page_id
        } for r in rows]
        
        # Add to ChromaDB
        try:
            collection.add(
                ids=paragraph_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            total_paragraphs += len(texts)
            print(f"  Added {len(texts)} paragraphs to ChromaDB")
        except Exception as e:
            print(f"  Error adding page {page_id}: {e}")
    
    conn.close()
    print(f"Auto-backfill complete! Added {total_paragraphs} paragraphs to ChromaDB")

if __name__ == "__main__":
    auto_backfill()
