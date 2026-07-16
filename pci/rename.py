from __future__ import annotations

import asyncio
import errno
import os
import re
import threading
from pathlib import Path
from typing import Awaitable, Callable, Optional

from openai import AsyncOpenAI

from pci.extractors import ExtractionError, extract_pdf, extract_text_file
from pci.vault import slugify


DEFAULT_RENAME_MODEL: str = os.environ.get("PCI_RENAME_MODEL", "gpt-5-mini")
SUPPORTED_EXTS: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".pdf"})
LLM_INPUT_MAX_CHARS: int = 4000
MAX_STEM_LEN: int = 80
MAX_SUFFIX_ATTEMPTS: int = 1000

GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"tmp[\s_-]*\d+", re.IGNORECASE),
    re.compile(r"untitled([\s_-]*\d*)?", re.IGNORECASE),
    re.compile(r"document[\s_-]*\d*", re.IGNORECASE),
    re.compile(r"new\s+document([\s_-]*\d*)?", re.IGNORECASE),
    re.compile(r"img[\s_-]+\d[\w_-]*", re.IGNORECASE),
    re.compile(r"image[\s_-]*\d*", re.IGNORECASE),
    re.compile(r"scre+n?shot([\s_-]*\d.*|\s+at\s.*)?", re.IGNORECASE),
    re.compile(r"pasted\s+image\s+\d+", re.IGNORECASE),
    re.compile(r"scan[\s_-]*\d*", re.IGNORECASE),
    re.compile(r"file[\s_-]*\d*", re.IGNORECASE),
    re.compile(r"download(\s*\(\d+\))?", re.IGNORECASE),
    re.compile(r"[a-f0-9]{8,}", re.IGNORECASE),
    re.compile(r"\d+"),
)

_QUOTE_CHARS = "\"'`\u201c\u201d\u2018\u2019"

_SYSTEM_PROMPT = (
    "You are a filename generator. Given the contents of a document with a "
    "generic filename, propose a single short, descriptive, kebab-case file "
    "stem (no extension, no quotes, no path components, lowercase ASCII, "
    "3-7 words). Respond with ONLY the stem - no commentary."
)

_client: Optional[AsyncOpenAI] = None
_rename_lock: threading.Lock = threading.Lock()


def is_generic_filename(name: str) -> bool:
    if not name:
        return False
    stem = Path(name).stem.strip()
    if not stem:
        return False
    return any(p.fullmatch(stem) for p in GENERIC_PATTERNS)


def sanitize_proposed_name(proposed: str) -> str:
    if not proposed:
        return "untitled"
    s = proposed.strip().strip(_QUOTE_CHARS).strip()
    if not s:
        return "untitled"
    if "." in s:
        head, _, tail = s.rpartition(".")
        if 1 <= len(tail) <= 9 and tail.isalnum() and head:
            s = head
    s = s.replace("..", " ").replace("/", " ").replace("\\", " ")
    s = s.lstrip("./\\- ")
    return slugify(s)[:MAX_STEM_LEN]


def safe_rename(old_path: Path, new_stem: str) -> Path:
    if not isinstance(old_path, Path):
        old_path = Path(old_path)
    if old_path.is_symlink():
        raise RuntimeError(f"refusing to rename symlink {old_path}")
    if not old_path.exists():
        raise FileNotFoundError(str(old_path))
    if old_path.stem == new_stem:
        return old_path
    parent, ext = old_path.parent, old_path.suffix
    with _rename_lock:
        n = 0
        while True:
            name = f"{new_stem}{ext}" if n == 0 else f"{new_stem}-{n}{ext}"
            candidate = parent / name
            try:
                os.link(str(old_path), str(candidate))
            except FileExistsError:
                n += 1
                if n > MAX_SUFFIX_ATTEMPTS:
                    raise RuntimeError(f"Too many conflicts for stem {new_stem!r}")
                continue
            except OSError as exc:
                unsupported = (errno.EXDEV, errno.EPERM, getattr(errno, "EOPNOTSUPP", -1))
                if exc.errno not in unsupported:
                    raise
                fd = -1
                try:
                    fd = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    n += 1
                    if n > MAX_SUFFIX_ATTEMPTS:
                        raise RuntimeError(f"Too many conflicts for stem {new_stem!r}")
                    continue
                finally:
                    if fd >= 0:
                        os.close(fd)
                try:
                    os.replace(str(old_path), str(candidate))
                except OSError:
                    try:
                        os.unlink(str(candidate))
                    except OSError:
                        pass
                    raise
                return candidate
            try:
                os.unlink(str(old_path))
            except OSError:
                try:
                    os.unlink(str(candidate))
                except OSError:
                    pass
                raise
            return candidate


def _get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for `pci rename`. "
                "Export it in your shell environment."
            )
        base_url = os.environ.get("OPENAI_BASE_URL")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _client = AsyncOpenAI(**kwargs)
    return _client


async def propose_filename(
    content: str,
    original_stem: str,
    *,
    model: str,
) -> str:
    snippet = (content or "").strip()[:LLM_INPUT_MAX_CHARS]
    user_msg = (
        f"Original filename stem: {original_stem!r}\n\n"
        f"Document content (truncated to {LLM_INPUT_MAX_CHARS} chars):\n"
        f"---\n{snippet}\n---\n\n"
        "Reply with ONLY the new filename stem. No extension. No quotes."
    )
    client = _get_openai_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
    )
    raw = (response.choices[0].message.content or "").strip()
    return sanitize_proposed_name(raw)


def _result(file_path: Path, *, renamed: bool, reason: str, **extra: str) -> dict:
    base: dict = {
        "path": str(file_path),
        "original": file_path.name,
        "renamed": renamed,
        "reason": reason,
    }
    base.update(extra)
    return base


async def rename_file_smart(
    file_path: Path,
    *,
    model: str,
    dry_run: bool,
    propose_fn: Optional[Callable[..., Awaitable[str]]] = None,
) -> dict:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.is_symlink():
        return _result(
            file_path,
            renamed=False,
            reason=f"rename-error: refusing to rename symlink {file_path}",
        )
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return _result(file_path, renamed=False, reason="unsupported-extension")
    if not file_path.exists():
        return _result(file_path, renamed=False, reason="not-found")
    try:
        if ext == ".pdf":
            data = await extract_pdf(str(file_path))
        else:
            data = await extract_text_file(str(file_path))
    except ExtractionError as exc:
        return _result(file_path, renamed=False, reason=f"extraction-error: {exc}")
    effective_propose = propose_fn if propose_fn is not None else propose_filename
    try:
        proposed_stem = await effective_propose(
            data.get("content", ""), file_path.stem, model=model
        )
    except Exception as exc:
        return _result(file_path, renamed=False, reason=f"llm-error: {exc}")
    proposed_stem = sanitize_proposed_name(proposed_stem)
    proposed_name = f"{proposed_stem}{file_path.suffix}"
    if proposed_stem == file_path.stem:
        return _result(file_path, renamed=False, reason="no-change", proposed=proposed_name)
    if dry_run:
        return _result(file_path, renamed=False, reason="dry-run", proposed=proposed_name)
    try:
        new_path = await asyncio.to_thread(safe_rename, file_path, proposed_stem)
    except Exception as exc:
        return _result(
            file_path, renamed=False, reason=f"rename-error: {exc}", proposed=proposed_name,
        )
    return _result(
        file_path, renamed=True, reason="renamed",
        proposed=proposed_name, new_path=str(new_path),
    )


async def rename_paths(
    paths: list[Path],
    *,
    model: str,
    dry_run: bool,
    only_generic: bool,
    recursive: bool,
    propose_fn: Optional[Callable[..., Awaitable[str]]] = None,
) -> dict:
    targets: list[Path] = []
    for raw in paths:
        p = raw if isinstance(raw, Path) else Path(raw)
        if p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            targets.extend(
                c for c in iterator
                if c.is_file() and not c.is_symlink() and c.suffix.lower() in SUPPORTED_EXTS
            )
        else:
            targets.append(p)
    semaphore = asyncio.Semaphore(5)

    async def _bounded(t: Path) -> dict:
        async with semaphore:
            if only_generic and not is_generic_filename(t.name):
                return _result(t, renamed=False, reason="not-generic")
            return await rename_file_smart(
                t, model=model, dry_run=dry_run, propose_fn=propose_fn,
            )

    results: list[dict] = list(await asyncio.gather(*(_bounded(t) for t in targets)))
    renamed = sum(1 for r in results if r.get("renamed"))
    errors = [
        r for r in results
        if isinstance(r.get("reason"), str) and (
            r["reason"].startswith(("extraction-error", "llm-error", "rename-error"))
            or r["reason"] == "not-found"
        )
    ]
    return {
        "processed": len(results),
        "renamed": renamed,
        "skipped": len(results) - renamed,
        "errors": errors,
        "results": results,
    }
