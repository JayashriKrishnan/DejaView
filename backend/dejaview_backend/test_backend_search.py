import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the same way as backend
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("paragraph_embeddings")

# Test the same search logic as backend
query = "evaluation metrics"
top_k = 5
min_similarity = 0.0

# Query expansion (same as backend)
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

expanded_queries = [query]
q_lower = query.lower()
for key, syns in synonyms_map.items():
    if key in q_lower and syns:
        expanded_queries.extend(syns)

print(f"Expanded queries: {expanded_queries}")

# Average the embeddings of expanded terms
q_vecs = model.encode(expanded_queries, convert_to_numpy=True)
q_emb = np.mean(q_vecs, axis=0).tolist()

# Query ChromaDB
results = collection.query(
    query_embeddings=[q_emb],
    n_results=top_k,
    include=["documents", "metadatas", "distances"]
)

print(f"Raw results: {results}")

# Process results
formatted = []
ids_list = results.get("ids", []) or []
docs_list = results.get("documents", []) or []
metas_list = results.get("metadatas", []) or []
dists_list = results.get("distances", []) or []

for ids, docs, metas, dists in zip(ids_list, docs_list, metas_list, dists_list):
    for pid, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1 - float(dist) if dist is not None else None
        if similarity is not None and similarity >= min_similarity:
            formatted.append({
                "paragraph_id": pid,
                "text": doc,
                "url": meta.get("url"),
                "title": meta.get("title"),
                "timestamp": meta.get("timestamp"),
                "paragraph_index": meta.get("paragraph_index"),
                "similarity_score": round(similarity, 3) if similarity is not None else None,
            })

print(f"\nFound {len(formatted)} results:")
for i, result in enumerate(formatted[:5]):
    print(f"{i+1}. {result['title'][:50]}... (similarity: {result['similarity_score']})")
    print(f"   Text: {result['text'][:100]}...")
    print()
