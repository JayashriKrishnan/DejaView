import chromadb

cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')
data = col.get(include=['metadatas'])
metas = data.get('metadatas', [])

print(f"Total metadata entries: {len(metas)}")

# Check for page_id field
with_page_id = [m for m in metas if 'page_id' in m]
print(f"Metadata with page_id: {len(with_page_id)}")

if with_page_id:
    print("Sample with page_id:")
    for m in with_page_id[:5]:
        print(f"Page {m.get('page_id')}: {m.get('title', '')[:30]}...")

# Check for recent ML content by title
ml_content = [m for m in metas if any(term in m.get('title', '').lower() for term in ['evaluation', 'machine learning', 'ml'])]
print(f"\nML-related content: {len(ml_content)}")
for m in ml_content[:3]:
    print(f"Title: {m.get('title', '')[:50]}...")
    print(f"Page ID: {m.get('page_id', 'N/A')}")
    print()
