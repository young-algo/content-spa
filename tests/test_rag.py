import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pci import rag


class TestRagHelpers(unittest.IsolatedAsyncioTestCase):
    async def test_async_delete_document_skips_rag_when_no_artifacts_exist(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=False),
            patch("pci.rag._run_with_rag", new=AsyncMock()) as mock_run,
        ):
            deleted, error = await rag.async_delete_document(7)

        self.assertTrue(deleted)
        self.assertIsNone(error)
        mock_run.assert_not_awaited()

    async def test_async_delete_document_uses_delete_mode_without_embedding_api(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=True),
            patch(
                "pci.rag._run_with_rag",
                new=AsyncMock(return_value=SimpleNamespace(status="success", message="ok")),
            ) as mock_run,
        ):
            deleted, error = await rag.async_delete_document(8)

        self.assertTrue(deleted)
        self.assertIsNone(error)
        mock_run.assert_awaited_once()
        self.assertEqual(mock_run.await_args.kwargs["require_embedding_api"], False)

    async def test_async_delete_document_honors_non_success_result(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=True),
            patch(
                "pci.rag._run_with_rag",
                new=AsyncMock(return_value=SimpleNamespace(status="fail", message="delete failed")),
            ),
        ):
            deleted, error = await rag.async_delete_document(9)

        self.assertFalse(deleted)
        self.assertEqual(error, "delete failed")

    async def test_async_delete_document_allows_not_found_result(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=True),
            patch(
                "pci.rag._run_with_rag",
                new=AsyncMock(return_value=SimpleNamespace(status="not_found", message="missing")),
            ),
        ):
            deleted, error = await rag.async_delete_document(10)

        self.assertTrue(deleted)
        self.assertIsNone(error)

    async def test_async_index_document_refreshes_existing_id_before_insert(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=True),
            patch("pci.rag.async_delete_document", new=AsyncMock(return_value=(True, None))) as mock_delete,
            patch("pci.rag._run_with_rag", new=AsyncMock(return_value="track-1")) as mock_run,
        ):
            result = await rag.async_index_document(
                doc_id=11,
                title="Updated",
                url="https://example.com/updated",
                source_type="article",
                summary="summary",
                tags=["tag"],
                content="fresh content",
            )

        self.assertEqual(result, "track-1")
        mock_delete.assert_awaited_once_with(11)
        mock_run.assert_awaited_once()

    async def test_async_index_document_raises_if_refresh_fails(self):
        with (
            patch("pci.rag._has_lightrag_artifacts", return_value=True),
            patch("pci.rag.async_delete_document", new=AsyncMock(return_value=(False, "delete failed"))),
            patch("pci.rag._run_with_rag", new=AsyncMock()) as mock_run,
        ):
            with self.assertRaisesRegex(RuntimeError, "delete failed"):
                await rag.async_index_document(
                    doc_id=12,
                    title="Updated",
                    url="https://example.com/updated",
                    source_type="article",
                    summary="summary",
                    tags=["tag"],
                    content="fresh content",
                )

        mock_run.assert_not_awaited()

    async def test_run_with_rag_delete_mode_handles_partial_store_without_vector_db_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "kv_store_doc_status.json"), "w", encoding="utf-8") as f:
                f.write("{}")

            async def callback(_rag):
                return "ok"

            with patch("pci.rag._working_dir", return_value=temp_dir):
                result = await rag._run_with_rag(callback, require_embedding_api=False)

            self.assertEqual(result, "ok")
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "vdb_chunks.json")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "vdb_entities.json")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "vdb_relationships.json")))
