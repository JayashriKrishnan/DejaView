import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("paragraph_embeddings")

# Test "machine learning" query
query = "machine learning"
q_emb = model.encode([query], convert_to_numpy=True).tolist()[0]

# Query ChromaDB
results = collection.query(
    query_embeddings=[q_emb],
    n_results=10,  # Get more results
    include=["documents", "metadatas", "distances"]
)

print(f"Query: '{query}'")
print(f"Found {len(results.get('ids', [[]])[0])} results")

# Show all results with similarity scores
ids_list = results.get("ids", []) or []
docs_list = results.get("documents", []) or []
metas_list = results.get("metadatas", []) or []
dists_list = results.get("distances", []) or []

for i, (pid, doc, meta, dist) in enumerate(zip(ids_list[0], docs_list[0], metas_list[0], dists_list[0])):
    similarity = 1 - float(dist) if dist is not None else None
    print(f"{i+1}. Similarity: {similarity:.3f}")
    print(f"   Title: {meta.get('title', '')[:50]}...")
    print(f"   Text: {doc[:100]}...")
    print()

# Test with lower similarity threshold
print("\n=== Results with similarity >= 0.1 ===")
for i, (pid, doc, meta, dist) in enumerate(zip(ids_list[0], docs_list[0], metas_list[0], dists_list[0])):
    similarity = 1 - float(dist) if dist is not None else None
    if similarity and similarity >= 0.1:
        print(f"{i+1}. Similarity: {similarity:.3f}")
        print(f"   Title: {meta.get('title', '')[:50]}...")
        print(f"   Text: {doc[:100]}...")
        print()
