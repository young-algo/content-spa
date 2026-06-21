import tempfile
import unittest
from pathlib import Path

from pci import db
from pci.vault import document_to_markdown, export_vault, slugify


class TestSlugify(unittest.TestCase):
    def test_basic_slugify(self):
        self.assertEqual(slugify("Hello World!"), "hello-world")

    def test_special_chars(self):
        self.assertEqual(slugify("AI & ML: Trends 2024"), "ai-ml-trends-2024")

    def test_empty_string(self):
        self.assertEqual(slugify(""), "untitled")

    def test_none_like(self):
        self.assertEqual(slugify("   "), "untitled")

    def test_max_length(self):
        self.assertLessEqual(len(slugify("a" * 100)), 80)

    def test_trailing_hyphens(self):
        self.assertEqual(slugify("---hello---"), "hello")


class TestDocumentToMarkdown(unittest.TestCase):
    def _row(self, **overrides):
        row = {
            "id": 1,
            "url": "https://example.com/article",
            "title": "Test Article",
            "source_type": "article",
            "tags": "ai,ml,python",
            "summary": "A concise summary.",
            "is_read": 0,
            "created_at": "2026-01-01T12:00:00",
            "content": "Full article content here.",
        }
        row.update(overrides)
        return row

    def test_frontmatter_fields(self):
        output = document_to_markdown(self._row())
        for field in (
            "title:",
            "url:",
            "source_type:",
            "tags:",
            "summary:",
            "is_read:",
            "pci_id:",
            "created_at:",
        ):
            self.assertIn(field, output)

    def test_starts_with_yaml_fence(self):
        output = document_to_markdown(self._row())
        self.assertTrue(output.startswith("---\n"))

    def test_ends_frontmatter_with_fence(self):
        output = document_to_markdown(self._row())
        frontmatter = output.split("\n\n", 1)[0]
        self.assertTrue(frontmatter.endswith("\n---"))

    def test_tags_as_yaml_list(self):
        output = document_to_markdown(self._row(tags="ai,ml,python"))
        self.assertIn("  - ai", output)
        self.assertIn("  - ml", output)
        self.assertIn("  - python", output)

    def test_empty_tags_list(self):
        for tags in (None, ""):
            output = document_to_markdown(self._row(tags=tags))
            self.assertIn("tags: []", output)

    def test_is_read_false(self):
        output = document_to_markdown(self._row(is_read=0))
        self.assertIn("is_read: false", output)

    def test_is_read_true(self):
        output = document_to_markdown(self._row(is_read=1))
        self.assertIn("is_read: true", output)

    def test_no_content_flag(self):
        output = document_to_markdown(self._row(), include_content=False)
        body = output.split("\n\n", 1)[1]
        self.assertEqual(body, "A concise summary.")
        self.assertNotIn("Full article content here.", body)

    def test_with_content(self):
        output = document_to_markdown(self._row(), include_content=True)
        body = output.split("\n\n", 1)[1]
        self.assertEqual(body, "Full article content here.")


class TestExportVault(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self._original_db_path = db.DB_PATH
        db.DB_PATH = f"{self.temp_dir.name}/test.db"
        db.init_db()
        self.vault_dir = f"{self.temp_dir.name}/vault"

    def tearDown(self):
        db.DB_PATH = self._original_db_path
        self.temp_dir.cleanup()

    def _insert_doc(
        self,
        url: str = "https://example.com",
        title: str = "Test Article",
        source_type: str = "article",
        tags: str = "ai,ml",
        content: str = "Full content here.",
    ) -> int:
        return db.insert_document(
            url=url,
            title=title,
            source_type=source_type,
            summary="A test summary.",
            tags=tags.split(",") if tags else [],
            content=content,
        )

    def test_creates_output_dir(self):
        export_vault(self.vault_dir)
        self.assertTrue(Path(self.vault_dir, "pci-content").is_dir())

    def test_exports_document(self):
        self._insert_doc()
        export_vault(self.vault_dir)
        exported_files = list(Path(self.vault_dir, "pci-content").glob("*.md"))
        self.assertEqual(len(exported_files), 1)

    def test_file_naming(self):
        doc_id = self._insert_doc(title="My Exported Document")
        export_vault(self.vault_dir)
        exported_files = list(Path(self.vault_dir, "pci-content").glob("*.md"))
        self.assertEqual(len(exported_files), 1)
        self.assertTrue(exported_files[0].name.startswith(f"{doc_id:05d}-my-exported-document"))

    def test_skips_existing(self):
        self._insert_doc()
        first_result = export_vault(self.vault_dir)
        self.assertEqual(first_result["exported"], 1)
        self.assertEqual(first_result["skipped"], 0)

        second_result = export_vault(self.vault_dir)
        self.assertEqual(second_result["exported"], 0)
        self.assertEqual(second_result["skipped"], 1)

    def test_returns_counts(self):
        result = export_vault(self.vault_dir)
        self.assertEqual(set(result), {"exported", "skipped", "vault_dir", "output_dir"})
        self.assertEqual(result["vault_dir"], self.vault_dir)
        self.assertEqual(result["output_dir"], str(Path(self.vault_dir) / "pci-content"))

    def test_empty_db(self):
        result = export_vault(self.vault_dir)
        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["skipped"], 0)
