import chromadb
import sqlite3

# Check ChromaDB
cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')
data = col.get(include=['metadatas'])
metas = data.get('metadatas', [])

print(f"Total entries in ChromaDB: {len(metas)}")

# Look for ML content by title
ml_content = [m for m in metas if any(term in m.get('title', '').lower() for term in ['evaluation', 'machine learning', 'ml'])]
print(f"ML content in ChromaDB: {len(ml_content)}")

for m in ml_content[:5]:
    print(f"Title: {m.get('title', '')[:50]}...")
    print(f"Page ID: {m.get('page_id', 'N/A')}")
    print(f"URL: {m.get('url', 'N/A')[:50]}...")
    print()

# Check SQLite for recent pages
conn = sqlite3.connect('dejaview.db')
c = conn.cursor()
c.execute("SELECT id, title FROM pages WHERE id IN (18, 19, 20)")
sqlite_pages = c.fetchall()
conn.close()

print(f"Recent pages in SQLite: {len(sqlite_pages)}")
for page_id, title in sqlite_pages:
    print(f"Page {page_id}: {title[:50]}...")

# Test search directly
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
        print(f"   Text: {doc[:100]}...")
        print()
