try:
    import sqlean as sqlite3
except ImportError:
    import sqlite3
import sqlite_vec
import os
from typing import List, Dict, Any

DB_PATH = os.environ.get("PCI_DB_PATH", "pci.db")
EMBEDDING_DIM = 384  # default for all-MiniLM-L6-v2

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            source_type TEXT,
            summary TEXT,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Vector table using vec0
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    
    conn.commit()
    conn.close()

def insert_document(url: str, title: str, source_type: str, summary: str, tags: list[str], embedding: list[float]) -> int:
    conn = get_db()
    cursor = conn.cursor()
    
    tags_str = ",".join(tags) if tags else ""
    
    # Insert or replace doc
    cursor.execute("""
        INSERT INTO documents (url, title, source_type, summary, tags)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET 
            title=excluded.title,
            source_type=excluded.source_type,
            summary=excluded.summary,
            tags=excluded.tags
    """, (url, title, source_type, summary, tags_str))
    
    # Get ID
    cursor.execute("SELECT id FROM documents WHERE url = ?", (url,))
    doc_id = cursor.fetchone()[0]
    
    # Insert vector
    import struct
    embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
    
    # We must insert into vectors
    cursor.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
    cursor.execute("""
        INSERT INTO vec_documents (id, embedding)
        VALUES (?, ?)
    """, (doc_id, embedding_bytes))
    
    conn.commit()
    conn.close()
    return doc_id

def search_similar(query_embedding: list[float], limit: int = 5) -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    import struct
    embedding_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
    
    cursor.execute("""
        SELECT 
            d.id, d.url, d.title, d.source_type, d.summary, d.tags, d.created_at,
            v.distance
        FROM vec_documents v
        JOIN documents d ON d.id = v.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, (embedding_bytes, limit))
    
    results = cursor.fetchall()
    conn.close()
    return results

def search_keyword(query: str, limit: int = 20) -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    
    # simple LIKE search for now
    like_query = f"%{query}%"
    cursor.execute("""
        SELECT * FROM documents
        WHERE title LIKE ? OR summary LIKE ? OR tags LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (like_query, like_query, like_query, limit))
    
    results = cursor.fetchall()
    conn.close()
    return results

def get_all_documents() -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, url, title, source_type, summary, tags, created_at
        FROM documents
        ORDER BY created_at DESC
    """)
    results = cursor.fetchall()
    conn.close()
    return results
