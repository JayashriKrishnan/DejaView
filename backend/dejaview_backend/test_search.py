import chromadb
import sqlite3

# Check ChromaDB
cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')
data = col.get(include=['metadatas'])
metas = data.get('metadatas', [])

print(f"Total embeddings in ChromaDB: {col.count()}")

# Look for recent ML pages
recent = [m for m in metas if 'evaluation' in m.get('title', '').lower() or 'machine learning' in m.get('title', '').lower()]
print(f"Found {len(recent)} recent ML pages:")
for m in recent[:5]:
    print(f"Title: {m.get('title', '')[:50]}...")

# Test search
print("\nTesting search for 'evaluation metrics':")
results = col.query(
    query_texts=["evaluation metrics"],
    n_results=5,
    include=['metadatas', 'documents']
)

if results and results.get('metadatas'):
    for i, (meta, doc) in enumerate(zip(results['metadatas'][0], results['documents'][0])):
        print(f"{i+1}. {meta.get('title', '')[:50]}...")
        print(f"   Text: {doc[:100]}...")
        print()
