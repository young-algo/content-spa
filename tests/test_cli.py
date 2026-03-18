import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from pci import cli, db


class TestCliCommands(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/cli.db"
        db.DB_PATH = self.db_path
        cli.os.environ["PCI_DB_PATH"] = self.db_path
        db.init_db()
        self.runner = CliRunner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def insert_sample(self, url: str, title: str, source_type: str, tags=None) -> int:
        return db.insert_document(
            url=url,
            title=title,
            source_type=source_type,
            summary=f"summary for {title}",
            tags=tags or ["tag1", "tag2"],
            embedding=[0.0] * db.EMBEDDING_DIM,
            content=f"full content for {title}",
        )

    def test_show_open_marks_document_as_read(self):
        doc_id = self.insert_sample("https://example.com/readme", "Read Me", "article")

        with patch("pci.cli.webbrowser.open") as mock_open:
            result = self.runner.invoke(cli.app, ["show", str(doc_id), "--open"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Read Me", result.stdout)
        mock_open.assert_called_once_with("https://example.com/readme")
        doc = db.get_document(doc_id)
        self.assertEqual(doc["is_read"], 1)
        self.assertIsNotNone(doc["read_at"])

    def test_open_marks_document_as_read(self):
        doc_id = self.insert_sample("https://example.com/open", "Open Me", "article")

        with patch("pci.cli.webbrowser.open") as mock_open:
            result = self.runner.invoke(cli.app, ["open", str(doc_id)])

        self.assertEqual(result.exit_code, 0)
        mock_open.assert_called_once_with("https://example.com/open")
        self.assertEqual(db.get_document(doc_id)["is_read"], 1)

    def test_list_defaults_to_unread_and_read_toggle_works(self):
        unread_id = self.insert_sample("https://example.com/unread", "Unread Doc", "article")
        read_id = self.insert_sample("https://example.com/read", "Read Doc", "youtube")
        db.mark_read(read_id)

        result = self.runner.invoke(cli.app, ["list", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Unread Doc", result.stdout)
        self.assertNotIn("Read Doc", result.stdout)

        result = self.runner.invoke(cli.app, ["list", "--read", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Read Doc", result.stdout)
        self.assertNotIn("Unread Doc", result.stdout)
        self.assertIsNotNone(unread_id)

    def test_search_type_filter_with_keyword_search(self):
        self.insert_sample("https://example.com/golf-video", "Golf Video", "youtube", ["golf"])
        self.insert_sample("https://example.com/golf-article", "Golf Article", "article", ["golf"])

        result = self.runner.invoke(cli.app, ["search", "Golf", "--no-semantic", "--type", "youtube"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Golf Video", result.stdout)
        self.assertNotIn("Golf Article", result.stdout)

    def test_search_semantic_uses_lightrag_retrieval_results(self):
        self.insert_sample("https://example.com/elon", "Elon Profile", "article", ["elon", "musk"])
        raw_data = {
            "status": "success",
            "message": "ok",
            "data": {
                "references": [{"reference_id": "1", "file_path": "https://example.com/elon"}],
                "chunks": [
                    {
                        "reference_id": "1",
                        "file_path": "https://example.com/elon",
                        "content": "Elon Musk profile chunk",
                    }
                ],
                "entities": [{"reference_id": "1", "file_path": "https://example.com/elon"}],
                "relationships": [],
            },
            "metadata": {"query_mode": "mix"},
        }

        with patch("pci.cli.async_query_data", new=AsyncMock(return_value=raw_data)) as mock_query:
            result = self.runner.invoke(cli.app, ["search", "Elon Musk", "--mode", "mix"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Elon Profile", result.stdout)
        mock_query.assert_awaited_once_with("Elon Musk", mode="mix", top_k=5, chunk_top_k=5)

    def test_search_semantic_overfetches_when_type_filter_is_applied(self):
        self.insert_sample("https://example.com/video", "Golf Video", "youtube", ["golf"])
        raw_data = {
            "status": "success",
            "message": "ok",
            "data": {
                "references": [{"reference_id": "1", "file_path": "https://example.com/video"}],
                "chunks": [
                    {
                        "reference_id": "1",
                        "file_path": "https://example.com/video",
                        "content": "Golf swing lesson",
                    }
                ],
                "entities": [],
                "relationships": [],
            },
            "metadata": {"query_mode": "mix"},
        }

        with patch("pci.cli.async_query_data", new=AsyncMock(return_value=raw_data)) as mock_query:
            result = self.runner.invoke(cli.app, ["search", "Golf", "--type", "youtube", "--limit", "1"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Golf Video", result.stdout)
        mock_query.assert_awaited_once_with("Golf", mode="mix", top_k=20, chunk_top_k=20)

    def test_ask_returns_paragraph_answer(self):
        answer_result = {
            "answer": "QURE is presented as a Huntington's disease-related idea with FDA-linked catalysts.",
            "raw_data": {
                "data": {
                    "references": [{"reference_id": "1", "file_path": "https://example.com/qure"}],
                }
            },
        }

        with patch("pci.cli.async_query_answer", new=AsyncMock(return_value=answer_result)) as mock_ask:
            result = self.runner.invoke(cli.app, ["ask", "QURE Huntington's disease FDA", "--references"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Huntington's disease-related idea", result.stdout)
        self.assertIn("References", result.stdout)
        mock_ask.assert_awaited_once()

    def test_retrieve_displays_structured_lightrag_output(self):
        raw_data = {
            "status": "success",
            "message": "ok",
            "data": {
                "references": [{"reference_id": "1", "file_path": "https://example.com/doc"}],
                "chunks": [
                    {
                        "reference_id": "1",
                        "file_path": "https://example.com/doc",
                        "content": "Chunk about AI systems",
                    }
                ],
                "entities": [
                    {
                        "reference_id": "1",
                        "entity_name": "AI systems",
                        "entity_type": "concept",
                        "description": "A category of intelligent software",
                    }
                ],
                "relationships": [
                    {
                        "reference_id": "1",
                        "src_id": "AI systems",
                        "tgt_id": "retrieval",
                        "keywords": "search, retrieval",
                        "description": "AI systems use retrieval pipelines",
                    }
                ],
            },
            "metadata": {
                "query_mode": "mix",
                "keywords": {"high_level": ["ai"], "low_level": ["retrieval"]},
            },
        }

        with patch("pci.cli.async_query_data", new=AsyncMock(return_value=raw_data)):
            result = self.runner.invoke(cli.app, ["retrieve", "AI retrieval"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("References", result.stdout)
        self.assertIn("Chunks", result.stdout)
        self.assertIn("Entities", result.stdout)
        self.assertIn("Relationships", result.stdout)

    def test_doctor_shows_active_configuration(self):
        result = self.runner.invoke(cli.app, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("PCI Doctor", result.stdout)
        self.assertIn("LightRAG index model", result.stdout)
        self.assertIn("Embedding model", result.stdout)
        self.assertIn("Reindex completed docs", result.stdout)

    def test_reindex_rebuilds_lightrag_from_sqlite(self):
        self.insert_sample("https://example.com/reindex", "Reindex Me", "article")

        with patch(
            "pci.cli.async_reindex_all_documents",
            new=AsyncMock(return_value={"indexed": 1, "skipped": 0, "total": 1, "resumed": False}),
        ) as mock_reindex:
            result = self.runner.invoke(cli.app, ["reindex"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indexed 1 document(s), skipped 0, total 1.", result.stdout)
        mock_reindex.assert_awaited_once_with(reset=True, resume=True)

    def test_reindex_resume_mode_passes_flags(self):
        with patch(
            "pci.cli.async_reindex_all_documents",
            new=AsyncMock(return_value={"indexed": 0, "skipped": 2, "total": 2, "resumed": True}),
        ) as mock_reindex:
            result = self.runner.invoke(cli.app, ["reindex", "--no-reset", "--resume"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("skipped 2", result.stdout)
        mock_reindex.assert_awaited_once_with(reset=False, resume=True)

    def test_delete_multiple_documents(self):
        first_id = self.insert_sample("https://example.com/1", "One", "article")
        second_id = self.insert_sample("https://example.com/2", "Two", "article")

        with patch("pci.cli.async_delete_document", new=AsyncMock(return_value=(True, None))):
            result = self.runner.invoke(cli.app, ["delete", str(first_id), str(second_id)], input="y\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleted 2 document(s).", result.stdout)
        self.assertIsNone(db.get_document(first_id))
        self.assertIsNone(db.get_document(second_id))

    def test_delete_keeps_sqlite_row_when_lightrag_cleanup_fails(self):
        doc_id = self.insert_sample("https://example.com/stuck", "Stuck", "article")

        with patch("pci.cli.async_delete_document", new=AsyncMock(return_value=(False, "storage error"))):
            result = self.runner.invoke(cli.app, ["delete", str(doc_id)], input="y\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Skipped SQLite deletion", result.stdout)
        self.assertIsNotNone(db.get_document(doc_id))
