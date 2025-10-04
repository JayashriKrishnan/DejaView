import sqlite3
import chromadb

# Check SQLite
conn = sqlite3.connect('dejaview.db')
cursor = conn.cursor()
cursor.execute('SELECT id, url, title, timestamp FROM pages ORDER BY timestamp DESC LIMIT 5')
sqlite_rows = cursor.fetchall()
print('Recent pages in SQLite:')
for row in sqlite_rows:
    print(f'ID: {row[0]}, Title: {row[2][:50]}..., Timestamp: {row[3]}')

# Check ChromaDB
cli = chromadb.PersistentClient(path='chroma_store')
col = cli.get_or_create_collection('paragraph_embeddings')
data = col.get(include=['metadatas'])
metas = data.get('metadatas', [])

# Check for recent pages
recent_page_ids = [18, 19, 20]
recent_embeddings = [m for m in metas if m.get('page_id') in recent_page_ids]
print(f'\nRecent pages in ChromaDB: {len(recent_embeddings)}')
for meta in recent_embeddings[:5]:
    print(f'Page {meta.get("page_id")}: {meta.get("title", "")[:50]}...')

print(f'\nTotal embeddings in ChromaDB: {col.count()}')

conn.close()
