import chromadb
import sqlite3
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer("all-MiniLM-L6-v2")
cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')

# Get the real ML pages from SQLite
conn = sqlite3.connect('dejaview.db')
c = conn.cursor()

# Get pages 18, 19, 20
c.execute("SELECT id, url, title, timestamp FROM pages WHERE id IN (18, 19, 20)")
pages = c.fetchall()

print(f"Found {len(pages)} ML pages in SQLite")

for page_id, url, title, timestamp in pages:
    print(f"\nProcessing page {page_id}: {title[:50]}...")
    
    # Get paragraphs for this page
    c.execute("SELECT id, paragraph_index, text FROM paragraphs WHERE page_id = ? ORDER BY paragraph_index ASC", (page_id,))
    rows = c.fetchall()
    
    if not rows:
        print(f"  No paragraphs found for page {page_id}")
        continue
    
    print(f"  Found {len(rows)} paragraphs")
    
    # Prepare data for ChromaDB
    paragraph_ids = [f"real_{r[0]}" for r in rows]  # Use real_ prefix to distinguish from test data
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
        col.add(
            ids=paragraph_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"  Successfully added {len(texts)} paragraphs to ChromaDB")
    except Exception as e:
        print(f"  Error adding page {page_id}: {e}")

conn.close()

# Test search
print("\nTesting search for 'evaluation metrics':")
results = col.query(
    query_texts=["evaluation metrics"],
    n_results=5,
    include=['metadatas', 'documents']
)

if results and results.get('metadatas'):
    print(f"Search found {len(results['metadatas'][0])} results:")
    for i, (meta, doc) in enumerate(zip(results['metadatas'][0], results['documents'][0])):
        print(f"{i+1}. {meta.get('title', '')[:50]}...")
        print(f"   Page ID: {meta.get('page_id', 'N/A')}")
        print(f"   URL: {meta.get('url', 'N/A')[:50]}...")
        print(f"   Text: {doc[:100]}...")
        print()
