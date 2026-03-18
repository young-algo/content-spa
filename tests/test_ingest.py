import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pci import db, ingest


class TestIngestAtomicity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/ingest.db"
        db.DB_PATH = self.db_path
        db.init_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_store_and_index_rolls_back_new_row_when_indexing_fails(self):
        url = "https://example.com/new"
        data = {
            "title": "New Doc",
            "source_type": "article",
            "content": "new content",
        }

        with (
            patch("pci.ingest.async_index_document", new=AsyncMock(side_effect=RuntimeError("index failed"))),
            patch("pci.ingest.async_delete_document", new=AsyncMock(return_value=(True, None))) as mock_delete,
        ):
            with self.assertRaisesRegex(RuntimeError, "index failed"):
                await ingest._store_and_index_document(
                    url=url,
                    data=data,
                    summary="new summary",
                    tags=["new"],
                )

        self.assertIsNone(db.get_document_by_url(url))
        mock_delete.assert_awaited_once()

    async def test_store_and_index_restores_existing_row_when_indexing_fails(self):
        url = "https://example.com/existing"
        doc_id = db.insert_document(
            url=url,
            title="Original Title",
            source_type="article",
            summary="original summary",
            tags=["old"],
            content="original content",
        )
        db.mark_read(doc_id)

        mock_index = AsyncMock(side_effect=[RuntimeError("index failed"), "restored"])
        with (
            patch("pci.ingest.async_index_document", new=mock_index),
            patch("pci.ingest.async_delete_document", new=AsyncMock(return_value=(True, None))) as mock_delete,
        ):
            with self.assertRaisesRegex(RuntimeError, "index failed"):
                await ingest._store_and_index_document(
                    url=url,
                    data={
                        "title": "Updated Title",
                        "source_type": "youtube",
                        "content": "updated content",
                    },
                    summary="updated summary",
                    tags=["updated"],
                )

        restored = db.get_document(doc_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored["title"], "Original Title")
        self.assertEqual(restored["source_type"], "article")
        self.assertEqual(restored["summary"], "original summary")
        self.assertEqual(restored["tags"], "old")
        self.assertEqual(restored["content"], "original content")
        self.assertEqual(restored["is_read"], 1)
        self.assertIsNotNone(restored["read_at"])
        self.assertEqual(mock_index.await_count, 2)
        restore_call = mock_index.await_args_list[1].kwargs
        self.assertEqual(restore_call["doc_id"], doc_id)
        self.assertEqual(restore_call["title"], "Original Title")
        self.assertEqual(restore_call["source_type"], "article")
        self.assertEqual(restore_call["summary"], "original summary")
        self.assertEqual(restore_call["tags"], ["old"])
        self.assertEqual(restore_call["content"], "original content")
        mock_delete.assert_not_awaited()
