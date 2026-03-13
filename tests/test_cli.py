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

    def test_search_semantic_awaits_embedding_generation(self):
        semantic_result = {
            "id": 1,
            "url": "https://example.com/elon",
            "title": "Elon Profile",
            "source_type": "article",
            "summary": "summary for Elon Profile",
            "tags": "elon,musk",
            "created_at": "2026-03-13 00:00:00",
            "is_read": 0,
            "read_at": None,
            "distance": 0.1234,
        }

        with patch("pci.cli.get_embedding", new=AsyncMock(return_value=[0.0] * db.EMBEDDING_DIM)) as mock_embedding, patch(
            "pci.cli.search_similar",
            return_value=[semantic_result],
        ) as mock_search:
            result = self.runner.invoke(cli.app, ["search", "Elon Musk"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Elon Profile", result.stdout)
        mock_embedding.assert_awaited_once_with("Elon Musk")
        mock_search.assert_called_once()

    def test_delete_multiple_documents(self):
        first_id = self.insert_sample("https://example.com/1", "One", "article")
        second_id = self.insert_sample("https://example.com/2", "Two", "article")

        result = self.runner.invoke(cli.app, ["delete", str(first_id), str(second_id)], input="y\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleted 2 document(s).", result.stdout)
        self.assertIsNone(db.get_document(first_id))
        self.assertIsNone(db.get_document(second_id))
