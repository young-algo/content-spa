import json
import os
import re
from pathlib import Path
from typing import Optional

from pci import db


_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_DUPLICATE_HYPHENS = re.compile(r"-+")


def slugify(text: str) -> str:
    value = (text or "").lower()
    value = _NON_SLUG_CHARS.sub("-", value)
    value = _DUPLICATE_HYPHENS.sub("-", value).strip("-")
    value = value[:80].strip("-")
    return value or "untitled"


def _quote(value: object) -> str:
    if value is None:
        return json.dumps("")
    return json.dumps(str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " "))


def _normalize_single_line(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()


def _format_tags(raw_tags: object) -> str:
    if not raw_tags:
        return "tags: []"

    tags = [tag.strip() for tag in str(raw_tags).split(",") if tag.strip()]
    if not tags:
        return "tags: []"

    return "tags:\n" + "\n".join(f"  - {tag}" for tag in tags)


def document_to_markdown(row: dict, include_content: bool = True) -> str:
    title = _normalize_single_line(row.get("title"), "Untitled") or "Untitled"
    url = _normalize_single_line(row.get("url"), "")
    source_type = _normalize_single_line(row.get("source_type"), "unknown") or "unknown"
    summary = _normalize_single_line(row.get("summary"), "")
    created_at = _normalize_single_line(row.get("created_at"), "")
    content = row.get("content") or ""

    frontmatter_lines = [
        "---",
        f"title: {_quote(title)}",
        f"url: {_quote(url)}",
        f"source_type: {source_type}",
        _format_tags(row.get("tags")),
        f"summary: {_quote(summary)}",
        f"is_read: {'true' if row.get('is_read') == 1 else 'false'}",
        f"pci_id: {int(row['id'])}",
        f"created_at: {_quote(created_at)}",
        "---",
    ]

    body = content if include_content and content else summary
    return "\n".join(frontmatter_lines) + "\n\n" + body


def export_vault(
    vault_dir: str,
    include_content: bool = True,
    source_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    base_dir = Path(os.fspath(vault_dir))
    output_dir = base_dir / "pci-content"
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_type is not None:
        rows = db.list_documents(source_type=source_type, limit=limit or 100_000)
    else:
        rows = db.get_all_documents()
        if limit is not None:
            rows = rows[:limit]

    exported = 0
    skipped = 0

    for row in rows:
        row_dict = dict(row)
        filename = f"{row_dict['id']:05d}-{slugify(row_dict.get('title') or '')}.md"
        target_path = output_dir / filename

        if target_path.exists():
            skipped += 1
            continue

        target_path.write_text(document_to_markdown(row_dict, include_content), encoding="utf-8")
        exported += 1

    return {
        "exported": exported,
        "skipped": skipped,
        "vault_dir": str(base_dir),
        "output_dir": str(output_dir),
    }
