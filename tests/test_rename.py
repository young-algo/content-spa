"""Contract tests pinning the public surface of ``pci.rename``.

Covers heuristics, sanitization, atomic rename, LLM proposal, single-file
orchestration, multi-target orchestration, error handling, and concurrent
rename safety.
"""

from __future__ import annotations

import asyncio
import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from pci.rename import (
    DEFAULT_RENAME_MODEL,
    is_generic_filename,
    propose_filename,
    rename_file_smart,
    rename_paths,
    safe_rename,
    sanitize_proposed_name,
)


class TestDefaultModelContract(unittest.TestCase):
    """Pin the default model. User-supplied ``gpt-5.4-mini`` is treated as a
    typo for ``gpt-5-mini``; that decision is surfaced in CLI help, README,
    and the module docstring. Tests lock the contract so any future change
    is intentional."""

    def test_default_model_is_gpt_5_mini_when_env_unset(self) -> None:
        import importlib

        import pci.rename as rename_mod

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PCI_RENAME_MODEL", None)
            reloaded = importlib.reload(rename_mod)
            try:
                self.assertEqual(reloaded.DEFAULT_RENAME_MODEL, "gpt-5-mini")
            finally:
                importlib.reload(rename_mod)

    def test_default_model_is_not_the_typo_literal(self) -> None:
        self.assertNotEqual(DEFAULT_RENAME_MODEL, "gpt-5.4-mini")
        self.assertTrue(DEFAULT_RENAME_MODEL)
        self.assertNotIn(" ", DEFAULT_RENAME_MODEL)


class TestIsGenericFilename(unittest.TestCase):
    """Classifier for filename stems that look auto-generated / placeholder."""

    GENERIC_STEMS: tuple[str, ...] = (
        "tmp4045",
        "tmp_4045",
        "tmp-4045",
        "Untitled",
        "Untitled 1",
        "Untitled-3",
        "document",
        "document1",
        "IMG_1234",
        "IMG-20240301-WA0005",
        "Screenshot 2024-08-01 at 11.59.05",
        "image_005",
        "Pasted image 20240101110512",
        "a1b2c3d4",
        "0123456789abcdef",
        "12345",
        "New Document",
        "scan_001",
        "file001",
        "download",
        "download (3)",
    )

    MEANINGFUL_STEMS: tuple[str, ...] = (
        "my-detailed-notes",
        "design-spec-v2",
        "Q3-budget-review",
        "README",
        "meeting-2024-09-15-priorities",
        "graphql-api-overview",
        "img-important",
        "screenshot-analysis",
        "pasted-image-notes",
        "documentation",
        "filed-under-x",
        "scanner-driver",
    )

    def test_recognizes_generic_stems(self) -> None:
        for stem in self.GENERIC_STEMS:
            with self.subTest(stem=stem):
                self.assertTrue(
                    is_generic_filename(stem),
                    f"expected {stem!r} to be classified as generic",
                )

    def test_passes_through_meaningful_stems(self) -> None:
        for stem in self.MEANINGFUL_STEMS:
            with self.subTest(stem=stem):
                self.assertFalse(
                    is_generic_filename(stem),
                    f"expected {stem!r} to be classified as meaningful",
                )

    def test_empty_string_is_not_generic(self) -> None:
        """An empty stem is degenerate, not generic — caller decides."""
        self.assertFalse(is_generic_filename(""))


class TestSanitizeProposedName(unittest.TestCase):
    """Pure sanitizer that turns raw LLM output into a safe filesystem stem."""

    def test_strips_surrounding_quotes(self) -> None:
        self.assertEqual(sanitize_proposed_name('"GraphQL API Overview"'), "graphql-api-overview")

    def test_strips_trailing_extension(self) -> None:
        self.assertEqual(sanitize_proposed_name("GraphQL API Overview.md"), "graphql-api-overview")

    def test_already_clean_is_noop(self) -> None:
        self.assertEqual(sanitize_proposed_name("my-notes"), "my-notes")

    def test_truncates_to_max_length(self) -> None:
        result = sanitize_proposed_name("a" * 200)
        self.assertLessEqual(len(result), 80)
        self.assertGreater(len(result), 0)

    def test_strips_whitespace_and_collapses(self) -> None:
        self.assertEqual(
            sanitize_proposed_name("   leading and trailing   "),
            "leading-and-trailing",
        )

    def test_empty_string_falls_back_to_untitled(self) -> None:
        self.assertEqual(sanitize_proposed_name(""), "untitled")

    def test_whitespace_only_falls_back_to_untitled(self) -> None:
        self.assertEqual(sanitize_proposed_name("   "), "untitled")

    def test_strips_unsafe_punctuation(self) -> None:
        self.assertEqual(sanitize_proposed_name("Foo / Bar : Baz?"), "foo-bar-baz")

    def test_pure_punctuation_falls_back_to_untitled(self) -> None:
        self.assertEqual(sanitize_proposed_name("....."), "untitled")

    def test_path_traversal_components_are_neutralized(self) -> None:
        """``../etc/passwd`` must not survive — no separators, no leading dot."""
        result = sanitize_proposed_name("../etc/passwd")
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)
        self.assertFalse(result.startswith("."))
        self.assertFalse(result.startswith("-"))

    def test_absolute_path_is_neutralized(self) -> None:
        result = sanitize_proposed_name("/absolute/path/note")
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)
        self.assertFalse(result.startswith("-"))
        self.assertGreater(len(result), 0)


class TestSafeRename(unittest.TestCase):
    """Filesystem-level safe rename: collision suffixing, extension preservation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_plain_rename(self) -> None:
        src = self.tmpdir / "tmp4045.md"
        src.write_text("hello")

        result = safe_rename(src, "graphql-api-overview")

        expected = self.tmpdir / "graphql-api-overview.md"
        self.assertFalse(src.exists())
        self.assertTrue(expected.exists())
        self.assertEqual(expected.read_text(), "hello")
        self.assertEqual(result, expected)

    def test_conflict_appends_suffix(self) -> None:
        existing = self.tmpdir / "target.md"
        existing.write_text("DO NOT TOUCH")
        src = self.tmpdir / "tmp.md"
        src.write_text("new content")

        result = safe_rename(src, "target")

        self.assertEqual(result.name, "target-1.md")
        self.assertTrue(result.exists())
        self.assertEqual(result.read_text(), "new content")
        # The pre-existing target must be unchanged.
        self.assertTrue(existing.exists())
        self.assertEqual(existing.read_text(), "DO NOT TOUCH")

    def test_double_conflict(self) -> None:
        (self.tmpdir / "target.md").write_text("a")
        (self.tmpdir / "target-1.md").write_text("b")
        src = self.tmpdir / "tmp.md"
        src.write_text("c")

        result = safe_rename(src, "target")

        self.assertEqual(result.name, "target-2.md")
        self.assertTrue(result.exists())
        self.assertEqual(result.read_text(), "c")

    def test_extension_preserved(self) -> None:
        src = self.tmpdir / "report.pdf"
        src.write_bytes(b"%PDF-1.4\n%fake pdf content")

        result = safe_rename(src, "quarterly-review")

        self.assertEqual(result.suffix, ".pdf")
        self.assertEqual(result.stem, "quarterly-review")
        self.assertTrue(result.exists())
        self.assertFalse(src.exists())

    def test_no_op_when_stem_matches(self) -> None:
        src = self.tmpdir / "notes.md"
        src.write_text("unchanged")
        mtime_before = os.path.getmtime(src)

        result = safe_rename(src, "notes")

        self.assertEqual(str(result), str(src))
        self.assertTrue(src.exists())
        self.assertEqual(src.read_text(), "unchanged")
        # Filesystem should not have been touched.
        self.assertEqual(os.path.getmtime(src), mtime_before)

    def test_returns_path_object(self) -> None:
        src = self.tmpdir / "tmp.md"
        src.write_text("x")

        result = safe_rename(src, "renamed")

        self.assertIsInstance(result, Path)


class TestSafeRenameSymlink(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_safe_rename_refuses_symlink_source(self) -> None:
        target_path = self.tmpdir / "target.md"
        link_path = self.tmpdir / "link.md"
        target_path.write_text("real")
        os.symlink(target_path, link_path)

        with self.assertRaisesRegex(RuntimeError, "symlink"):
            safe_rename(link_path, "renamed")

        self.assertTrue(link_path.is_symlink())
        self.assertTrue(target_path.exists())
        self.assertEqual(target_path.read_text(), "real")


class TestSafeRenameFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fallback_path_taken_when_link_unsupported(self) -> None:
        src = self.tmpdir / "tmp.md"
        src.write_text("fallback")
        unsupported = OSError(getattr(errno, "EOPNOTSUPP", errno.EPERM), "not supported")

        with patch("pci.rename.os.link", side_effect=unsupported):
            result = safe_rename(src, "newname")

        self.assertEqual(result, self.tmpdir / "newname.md")
        self.assertFalse(src.exists())
        self.assertEqual(result.read_text(), "fallback")


class TestSafeRenameFallbackCleanup(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_placeholder_cleaned_when_os_replace_fails(self) -> None:
        src = self.tmpdir / "tmp.md"
        src.write_text("original")
        unsupported = OSError(getattr(errno, "EOPNOTSUPP", errno.EPERM), "not supported")

        with (
            patch("pci.rename.os.link", side_effect=unsupported),
            patch("pci.rename.os.replace", side_effect=FileNotFoundError("simulated")),
        ):
            with self.assertRaises(FileNotFoundError):
                safe_rename(src, "newname")

        self.assertEqual(sorted(p.name for p in self.tmpdir.iterdir()), ["tmp.md"])
        self.assertEqual(src.read_text(), "original")


class TestProposeFilename(unittest.IsolatedAsyncioTestCase):
    """LLM-backed filename proposal — must be fully mocked, no network."""

    async def test_returns_sanitized_llm_output(self) -> None:
        fake_resp = Mock(choices=[Mock(message=Mock(content='"graphql-api-overview"'))])
        fake_client = Mock(
            chat=Mock(completions=Mock(create=AsyncMock(return_value=fake_resp))),
        )

        with patch("pci.rename._get_openai_client", return_value=fake_client):
            result = await propose_filename(
                content="a long markdown about graphql",
                original_stem="tmp4045",
                model="gpt-5-mini",
            )

        self.assertEqual(result, "graphql-api-overview")

        fake_client.chat.completions.create.assert_awaited_once()
        call_kwargs = fake_client.chat.completions.create.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-5-mini")

        msgs = call_kwargs["messages"]
        self.assertIsInstance(msgs, list)
        self.assertGreater(len(msgs), 0)

        user_content = " ".join(m["content"] for m in msgs if m["role"] == "user")
        self.assertIn("tmp4045", user_content)
        self.assertIn("graphql", user_content)

    async def test_truncates_long_content(self) -> None:
        """Content > 4000 chars must be bounded before being sent to the LLM."""
        long_content = "x" * 6000
        fake_resp = Mock(choices=[Mock(message=Mock(content="bounded-name"))])
        fake_client = Mock(
            chat=Mock(completions=Mock(create=AsyncMock(return_value=fake_resp))),
        )

        with patch("pci.rename._get_openai_client", return_value=fake_client):
            result = await propose_filename(
                content=long_content,
                original_stem="tmp",
                model="gpt-5-mini",
            )

        self.assertEqual(result, "bounded-name")

        call_kwargs = fake_client.chat.completions.create.await_args.kwargs
        user_content = " ".join(
            m["content"] for m in call_kwargs["messages"] if m["role"] == "user"
        )
        # 5000 char ceiling leaves room for prompt scaffolding around the snippet.
        self.assertLessEqual(len(user_content), 5000)


class TestSafeRenameConcurrency(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_two_concurrent_renames_to_same_stem_preserves_both(self) -> None:
        a_path = self.tmpdir / "a.md"
        b_path = self.tmpdir / "b.md"
        a_path.write_text("AAA")
        b_path.write_text("BBB")

        result_a, result_b = await asyncio.gather(
            asyncio.to_thread(safe_rename, a_path, "shared"),
            asyncio.to_thread(safe_rename, b_path, "shared"),
        )

        self.assertNotEqual(result_a, result_b)
        self.assertFalse(a_path.exists())
        self.assertFalse(b_path.exists())
        self.assertTrue(result_a.exists())
        self.assertTrue(result_b.exists())
        self.assertEqual(
            sorted([result_a.read_text(), result_b.read_text()]),
            ["AAA", "BBB"],
        )


class TestRenameFileSmart(unittest.IsolatedAsyncioTestCase):
    """End-to-end single-file orchestrator: dispatch + extract + propose + rename."""

    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_dry_run_returns_proposal_without_rename(self) -> None:
        src = self.tmpdir / "tmp4045.md"
        src.write_text("# GraphQL\n\ncontent")

        extractor = AsyncMock(
            return_value={
                "title": "tmp",
                "content": "GraphQL overview ...",
                "source_type": "markdown",
            }
        )
        proposer = AsyncMock(return_value="graphql-api-overview")

        with (
            patch("pci.rename.extract_text_file", new=extractor),
            patch("pci.rename.propose_filename", new=proposer),
        ):
            result = await rename_file_smart(src, model="gpt-5-mini", dry_run=True)

        self.assertEqual(result["renamed"], False)
        self.assertEqual(result["reason"], "dry-run")
        self.assertEqual(result["proposed"], "graphql-api-overview.md")
        self.assertEqual(result["original"], "tmp4045.md")
        # File must still exist under its original name.
        self.assertTrue(src.exists())

    async def test_execute_renames_file(self) -> None:
        src = self.tmpdir / "tmp4045.md"
        src.write_text("# GraphQL\n\ncontent")

        extractor = AsyncMock(
            return_value={
                "title": "tmp",
                "content": "GraphQL overview ...",
                "source_type": "markdown",
            }
        )
        proposer = AsyncMock(return_value="graphql-api-overview")

        with (
            patch("pci.rename.extract_text_file", new=extractor),
            patch("pci.rename.propose_filename", new=proposer),
        ):
            result = await rename_file_smart(src, model="gpt-5-mini", dry_run=False)

        self.assertEqual(result["renamed"], True)
        new_path_str = str(result["new_path"])
        self.assertTrue(new_path_str.endswith("graphql-api-overview.md"))

        self.assertFalse(src.exists())
        new_path = Path(new_path_str)
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.read_text(), "# GraphQL\n\ncontent")

    async def test_pdf_dispatch_uses_extract_pdf(self) -> None:
        """PDF files must dispatch to ``extract_pdf``, never to ``extract_text_file``."""
        pdf_path = self.tmpdir / "tmp123.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        pdf_extractor = AsyncMock(
            return_value={
                "title": "pdf",
                "content": "research paper ...",
                "source_type": "pdf",
            }
        )
        text_extractor = AsyncMock(
            return_value={
                "title": "should-not-be-called",
                "content": "",
                "source_type": "markdown",
            }
        )
        proposer = AsyncMock(return_value="quantum-research")

        with (
            patch("pci.rename.extract_pdf", new=pdf_extractor),
            patch("pci.rename.extract_text_file", new=text_extractor),
            patch("pci.rename.propose_filename", new=proposer),
        ):
            await rename_file_smart(pdf_path, model="x", dry_run=True)

        pdf_extractor.assert_awaited()
        text_extractor.assert_not_awaited()

    async def test_unsupported_extension_returns_skip(self) -> None:
        png_path = self.tmpdir / "image.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        proposer = AsyncMock(return_value="should-not-be-called")

        with patch("pci.rename.propose_filename", new=proposer):
            result = await rename_file_smart(png_path, model="x", dry_run=False)

        self.assertEqual(result["renamed"], False)
        self.assertEqual(result["reason"], "unsupported-extension")
        self.assertTrue(png_path.exists())
        proposer.assert_not_awaited()

    async def test_no_change_when_llm_proposes_same_stem(self) -> None:
        src = self.tmpdir / "existing-notes.md"
        src.write_text("body")

        extractor = AsyncMock(
            return_value={
                "title": "existing",
                "content": "stuff",
                "source_type": "markdown",
            }
        )
        proposer = AsyncMock(return_value="existing-notes")

        with (
            patch("pci.rename.extract_text_file", new=extractor),
            patch("pci.rename.propose_filename", new=proposer),
        ):
            result = await rename_file_smart(src, model="x", dry_run=False)

        self.assertEqual(result["renamed"], False)
        self.assertEqual(result["reason"], "no-change")
        self.assertTrue(src.exists())
        self.assertEqual(src.name, "existing-notes.md")


class TestRenameFileSmartErrorHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_llm_error_returns_structured_reason(self) -> None:
        path = self.tmpdir / "tmp4045.md"
        path.write_text("original")
        extractor = AsyncMock(return_value={"content": "x"})
        proposer = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("pci.rename.extract_text_file", new=extractor):
            result = await rename_file_smart(
                path, model="x", dry_run=False, propose_fn=proposer,
            )

        self.assertEqual(result["renamed"], False)
        self.assertTrue(result["reason"].startswith("llm-error"))
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "original")

    async def test_rename_error_returns_structured_reason(self) -> None:
        path = self.tmpdir / "tmp4045.md"
        path.write_text("original")
        extractor = AsyncMock(return_value={"content": "x"})
        proposer = AsyncMock(return_value="better-name")

        with (
            patch("pci.rename.extract_text_file", new=extractor),
            patch("pci.rename.safe_rename", side_effect=RuntimeError("disk full")),
        ):
            result = await rename_file_smart(
                path, model="x", dry_run=False, propose_fn=proposer,
            )

        self.assertEqual(result["renamed"], False)
        self.assertTrue(result["reason"].startswith("rename-error"))
        self.assertEqual(result["proposed"], "better-name.md")
        self.assertTrue(path.exists())


class TestRenameFileSmartSymlinkEarlyReject(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_explicit_symlink_arg_rejected_before_extraction(self) -> None:
        real_path = self.tmpdir / "real.md"
        link_path = self.tmpdir / "link.md"
        real_path.write_text("SECRET")
        os.symlink(real_path, link_path)
        text_extractor = AsyncMock()
        pdf_extractor = AsyncMock()
        proposer = AsyncMock(return_value="renamed")

        with (
            patch("pci.rename.extract_text_file", new=text_extractor),
            patch("pci.rename.extract_pdf", new=pdf_extractor),
        ):
            result = await rename_file_smart(
                link_path, model="x", dry_run=False, propose_fn=proposer,
            )

        self.assertEqual(result["renamed"], False)
        self.assertTrue(result["reason"].startswith("rename-error"))
        self.assertIn("symlink", result["reason"])
        text_extractor.assert_not_awaited()
        pdf_extractor.assert_not_awaited()
        proposer.assert_not_awaited()
        self.assertTrue(link_path.is_symlink())
        self.assertTrue(real_path.exists())
        self.assertEqual(real_path.read_text(), "SECRET")


class TestRenameFileSmartTrustBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_dirty_injected_propose_fn_output_is_sanitized(self) -> None:
        path = self.tmpdir / "tmp4045.md"
        path.write_text("original")
        extractor = AsyncMock(return_value={"content": "x"})
        proposer = AsyncMock(return_value="../escaped")
        escaped_path = self.tmpdir.parent / "escaped.md"

        with patch("pci.rename.extract_text_file", new=extractor):
            result = await rename_file_smart(
                path, model="x", dry_run=False, propose_fn=proposer,
            )

        new_path = Path(result["new_path"])
        self.assertEqual(result["renamed"], True)
        self.assertEqual(new_path.parent, self.tmpdir)
        self.assertEqual(new_path.name, "escaped.md")
        self.assertFalse(escaped_path.exists())

    async def test_injected_propose_fn_returning_slashes_is_sanitized(self) -> None:
        path = self.tmpdir / "tmp4045.md"
        path.write_text("original")
        extractor = AsyncMock(return_value={"content": "x"})
        proposer = AsyncMock(return_value="foo/bar/baz")

        with patch("pci.rename.extract_text_file", new=extractor):
            result = await rename_file_smart(
                path, model="x", dry_run=False, propose_fn=proposer,
            )

        new_path = Path(result["new_path"])
        self.assertEqual(result["renamed"], True)
        self.assertEqual(new_path.parent, self.tmpdir)
        self.assertEqual(new_path.name, "foo-bar-baz.md")


class TestRenamePaths(unittest.IsolatedAsyncioTestCase):
    """Multi-file orchestrator: filtering, recursion, and summary shape."""

    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _make_counting_mock() -> AsyncMock:
        return AsyncMock(
            return_value={
                "renamed": True,
                "reason": "renamed",
                "original": "x",
                "proposed": "y",
            }
        )

    async def test_only_generic_filter_works(self) -> None:
        """With ``only_generic=True``, non-generic stems are skipped, not renamed."""
        (self.tmpdir / "tmp4045.md").write_text("a")
        (self.tmpdir / "my-detailed-notes.md").write_text("b")

        smart_mock = self._make_counting_mock()
        with patch("pci.rename.rename_file_smart", new=smart_mock):
            summary = await rename_paths(
                [self.tmpdir],
                model="x",
                dry_run=True,
                only_generic=True,
                recursive=False,
            )

        self.assertEqual(smart_mock.await_count, 1)
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["renamed"], 1)
        self.assertEqual(summary["skipped"], 1)

    async def test_recursive_walks_subdirs(self) -> None:
        (self.tmpdir / "tmp4045.md").write_text("a")
        sub = self.tmpdir / "sub"
        sub.mkdir()
        (sub / "tmp9999.txt").write_text("b")

        smart_mock = self._make_counting_mock()
        with patch("pci.rename.rename_file_smart", new=smart_mock):
            await rename_paths(
                [self.tmpdir],
                model="x",
                dry_run=True,
                only_generic=True,
                recursive=True,
            )

        self.assertEqual(smart_mock.await_count, 2)

    async def test_all_files_processes_non_generic(self) -> None:
        """``only_generic=False`` must process every supported file regardless of stem."""
        (self.tmpdir / "tmp4045.md").write_text("a")
        (self.tmpdir / "my-detailed-notes.md").write_text("b")
        (self.tmpdir / "design-spec-v2.md").write_text("c")

        smart_mock = self._make_counting_mock()
        with patch("pci.rename.rename_file_smart", new=smart_mock):
            await rename_paths(
                [self.tmpdir],
                model="x",
                dry_run=True,
                only_generic=False,
                recursive=True,
            )

        self.assertEqual(smart_mock.await_count, 3)

    async def test_summary_dict_shape(self) -> None:
        (self.tmpdir / "tmp4045.md").write_text("a")

        smart_mock = self._make_counting_mock()
        with patch("pci.rename.rename_file_smart", new=smart_mock):
            summary = await rename_paths(
                [self.tmpdir],
                model="x",
                dry_run=True,
                only_generic=False,
                recursive=False,
            )

        self.assertEqual(
            set(summary.keys()),
            {"processed", "renamed", "skipped", "errors", "results"},
        )
        self.assertIsInstance(summary["errors"], list)
        self.assertIsInstance(summary["results"], list)


class TestRenamePathsSymlinkFilter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_directory_walk_skips_symlinks(self) -> None:
        real_path = self.tmpdir / "tmp1.md"
        link_path = self.tmpdir / "tmp2.md"
        real_path.write_text("real")
        os.symlink(real_path, link_path)
        proposer = AsyncMock(return_value="renamed")

        summary = await rename_paths(
            [self.tmpdir],
            model="x",
            dry_run=False,
            only_generic=True,
            recursive=False,
            propose_fn=proposer,
        )

        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["renamed"], 1)
        self.assertNotIn("tmp2.md", [r["original"] for r in summary["results"]])
        self.assertTrue(link_path.is_symlink())


if __name__ == "__main__":
    unittest.main()
