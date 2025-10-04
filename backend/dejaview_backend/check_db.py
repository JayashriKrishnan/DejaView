import sqlite3
from datetime import datetime

conn = sqlite3.connect('dejaview.db')
cursor = conn.cursor()

# Check total pages
cursor.execute('SELECT COUNT(*) FROM pages')
total_pages = cursor.fetchone()[0]
print(f'Total pages: {total_pages}')

# Check recent pages
cursor.execute('SELECT title, timestamp FROM pages ORDER BY timestamp DESC LIMIT 5')
print('\nRecent pages:')
for row in cursor.fetchall():
    print(f'  {row[1]} - {row[0]}')

# Check today's pages
today = datetime.now().strftime('%Y-%m-%d')
cursor.execute("SELECT title, timestamp FROM pages WHERE timestamp LIKE ?", (f'{today}%',))
today_pages = cursor.fetchall()
print(f'\nToday\'s pages ({today}): {len(today_pages)}')
for row in today_pages:
    print(f'  {row[1]} - {row[0]}')

conn.close()
