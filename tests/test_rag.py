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
