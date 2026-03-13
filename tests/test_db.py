import tempfile
import unittest

from pci import db


class TestDatabaseHelpers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/test.db"
        db.DB_PATH = self.db_path
        db.init_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def insert_sample(self, url: str, title: str, source_type: str, tags=None, content: str = "full content") -> int:
        return db.insert_document(
            url=url,
            title=title,
            source_type=source_type,
            summary=f"summary for {title}",
            tags=tags or ["tag1", "tag2"],
            embedding=[0.0] * db.EMBEDDING_DIM,
            content=content,
        )

    def test_migration_adds_new_columns(self):
        conn = db.get_db()
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cur.fetchall()}
        conn.close()

        self.assertIn("content", columns)
        self.assertIn("is_read", columns)
        self.assertIn("read_at", columns)

    def test_insert_document_stores_truncated_content(self):
        doc_id = self.insert_sample(
            url="https://example.com/a",
            title="Example",
            source_type="article",
            content="x" * (db.MAX_CONTENT_LENGTH + 50),
        )

        row = db.get_document(doc_id)
        self.assertIsNotNone(row)
        self.assertEqual(len(row["content"]), db.MAX_CONTENT_LENGTH)
        self.assertEqual(row["is_read"], 0)
        self.assertIsNone(row["read_at"])

    def test_list_mark_and_delete_helpers(self):
        article_id = self.insert_sample("https://example.com/article", "Article", "article", ["news", "ai"])
        youtube_id = self.insert_sample("https://example.com/video", "Video", "youtube", ["video"])

        unread_docs = db.list_documents(is_read=False, limit=10)
        self.assertEqual({row["id"] for row in unread_docs}, {article_id, youtube_id})

        youtube_only = db.list_documents(is_read=False, source_type="youtube", limit=10)
        self.assertEqual([row["id"] for row in youtube_only], [youtube_id])

        self.assertTrue(db.mark_read(article_id))
        article = db.get_document(article_id)
        self.assertEqual(article["is_read"], 1)
        self.assertIsNotNone(article["read_at"])

        self.assertTrue(db.mark_unread(article_id))
        article = db.get_document(article_id)
        self.assertEqual(article["is_read"], 0)
        self.assertIsNone(article["read_at"])

        self.assertEqual(db.mark_all_read(), 2)
        stats = db.get_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["unread_count"], 0)
        self.assertEqual(stats["read_count"], 2)
        self.assertEqual(stats["by_source_type"][0]["count"], 1)
        self.assertTrue(any(tag == "ai" for tag, _ in stats["top_tags"]))

        self.assertTrue(db.delete_document(youtube_id))
        self.assertIsNone(db.get_document(youtube_id))
