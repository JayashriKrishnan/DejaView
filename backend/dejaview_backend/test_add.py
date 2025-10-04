import chromadb
import sqlite3
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer("all-MiniLM-L6-v2")
cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')

# Get page 20 data
conn = sqlite3.connect('dejaview.db')
c = conn.cursor()
c.execute("SELECT id, paragraph_index, text FROM paragraphs WHERE page_id = 20 ORDER BY paragraph_index ASC")
rows = c.fetchall()
conn.close()

if rows:
    print(f"Found {len(rows)} paragraphs for page 20")
    
    # Prepare data
    paragraph_ids = [f"test_{r[0]}" for r in rows]  # Use test prefix to avoid conflicts
    texts = [r[2] for r in rows]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    metadatas = [{
        "url": "https://test.com",
        "title": "Test Evaluation Metrics",
        "timestamp": "2025-10-04T12:41:59.654Z",
        "paragraph_index": r[1],
        "page_id": 20
    } for r in rows]
    
    # Add to ChromaDB
    try:
        col.add(
            ids=paragraph_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print("Successfully added test data to ChromaDB")
        
        # Test search
        results = col.query(
            query_texts=["evaluation metrics"],
            n_results=3,
            include=['metadatas', 'documents']
        )
        
        print("\nSearch results:")
        if results and results.get('metadatas'):
            for i, (meta, doc) in enumerate(zip(results['metadatas'][0], results['documents'][0])):
                print(f"{i+1}. {meta.get('title', '')[:50]}...")
                print(f"   Page ID: {meta.get('page_id', 'N/A')}")
                print(f"   Text: {doc[:100]}...")
                print()
                
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No paragraphs found for page 20")
