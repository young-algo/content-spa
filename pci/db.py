try:
    import sqlean as sqlite3
except ImportError:
    import sqlite3
import os
import struct
from collections import Counter
from typing import Any, Dict, List, Optional

import sqlite_vec

DB_PATH = os.environ.get("PCI_DB_PATH", "pci.db")
EMBEDDING_DIM = 384  # default for all-MiniLM-L6-v2
MAX_CONTENT_LENGTH = 200_000


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_column_names(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def migrate_db() -> None:
    conn = get_db()
    cursor = conn.cursor()
    columns = _get_column_names(cursor, "documents")

    if "content" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN content TEXT")
    if "is_read" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN is_read INTEGER DEFAULT 0")
    if "read_at" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN read_at DATETIME")

    conn.commit()
    conn.close()


def init_db() -> None:
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            source_type TEXT,
            summary TEXT,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
        """
    )

    conn.commit()
    conn.close()
    migrate_db()


def insert_document(
    url: str,
    title: str,
    source_type: str,
    summary: str,
    tags: list[str],
    embedding: list[float],
    content: Optional[str] = None,
) -> int:
    conn = get_db()
    cursor = conn.cursor()

    tags_str = ",".join(tags) if tags else ""
    truncated_content = (content or "")[:MAX_CONTENT_LENGTH] if content is not None else None

    cursor.execute(
        """
        INSERT INTO documents (url, title, source_type, summary, tags, content)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            source_type=excluded.source_type,
            summary=excluded.summary,
            tags=excluded.tags,
            content=excluded.content
        """,
        (url, title, source_type, summary, tags_str, truncated_content),
    )

    cursor.execute("SELECT id FROM documents WHERE url = ?", (url,))
    doc_id = cursor.fetchone()[0]

    embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
    cursor.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
    cursor.execute(
        """
        INSERT INTO vec_documents (id, embedding)
        VALUES (?, ?)
        """,
        (doc_id, embedding_bytes),
    )

    conn.commit()
    conn.close()
    return doc_id


def get_document(doc_id: int) -> Optional[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def list_documents(
    is_read: Optional[bool] = None,
    source_type: Optional[str] = None,
    limit: int = 20,
) -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM documents"
    conditions = []
    params: list[Any] = []

    if is_read is not None:
        conditions.append("is_read = ?")
        params.append(1 if is_read else 0)
    if source_type:
        conditions.append("LOWER(source_type) = LOWER(?)")
        params.append(source_type)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results


def search_similar(
    query_embedding: list[float],
    limit: int = 5,
    source_type: Optional[str] = None,
) -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    embedding_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    query = """
        SELECT
            d.id, d.url, d.title, d.source_type, d.summary, d.tags, d.created_at,
            d.is_read, d.read_at,
            v.distance
        FROM vec_documents v
        JOIN documents d ON d.id = v.id
        WHERE v.embedding MATCH ? AND k = ?
    """
    params: list[Any] = [embedding_bytes, limit]

    if source_type:
        query += " AND LOWER(d.source_type) = LOWER(?)"
        params.append(source_type)

    query += " ORDER BY v.distance"

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results


def search_keyword(
    query: str,
    limit: int = 20,
    source_type: Optional[str] = None,
) -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()

    like_query = f"%{query}%"
    sql = """
        SELECT * FROM documents
        WHERE (title LIKE ? OR summary LIKE ? OR tags LIKE ?)
    """
    params: list[Any] = [like_query, like_query, like_query]

    if source_type:
        sql += " AND LOWER(source_type) = LOWER(?)"
        params.append(source_type)

    sql += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(sql, params)
    results = cursor.fetchall()
    conn.close()
    return results


def get_all_documents() -> List[sqlite3.Row]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, url, title, source_type, summary, tags, content, is_read, read_at, created_at
        FROM documents
        ORDER BY datetime(created_at) DESC, id DESC
        """
    )
    results = cursor.fetchall()
    conn.close()
    return results


def mark_read(doc_id: int) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE documents SET is_read = 1, read_at = CURRENT_TIMESTAMP WHERE id = ?",
        (doc_id,),
    )
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def mark_unread(doc_id: int) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE documents SET is_read = 0, read_at = NULL WHERE id = ?",
        (doc_id,),
    )
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def mark_all_read() -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE documents SET is_read = 1, read_at = CURRENT_TIMESTAMP WHERE is_read = 0"
    )
    changed = cursor.rowcount
    conn.commit()
    conn.close()
    return changed


def delete_document(doc_id: int) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_stats() -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM documents")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) AS unread_count FROM documents WHERE is_read = 0")
    unread_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT source_type, COUNT(*) AS count
        FROM documents
        GROUP BY source_type
        ORDER BY count DESC, source_type ASC
        """
    )
    by_source_type = [dict(row) for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT * FROM documents
        WHERE is_read = 0
        ORDER BY datetime(created_at) ASC, id ASC
        LIMIT 1
        """
    )
    oldest_unread = cursor.fetchone()

    cursor.execute("SELECT tags FROM documents WHERE tags IS NOT NULL AND tags != ''")
    tag_counter: Counter[str] = Counter()
    for row in cursor.fetchall():
        for tag in row[0].split(","):
            cleaned = tag.strip()
            if cleaned:
                tag_counter[cleaned] += 1

    conn.close()
    return {
        "total": total,
        "unread_count": unread_count,
        "read_count": total - unread_count,
        "by_source_type": by_source_type,
        "top_tags": tag_counter.most_common(10),
        "oldest_unread": dict(oldest_unread) if oldest_unread else None,
    }
