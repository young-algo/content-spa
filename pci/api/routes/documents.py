import asyncio
import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from pci import db
from pci.api.schemas import (
    DocumentListResponse,
    DocumentOut,
    DocumentUpdate,
)
from pci.api.deps import pagination_params

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _row_to_dict(row) -> dict:
    return dict(row) if row else {}


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=30, ge=1, le=200),
    is_read: Optional[bool] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    sort: str = Query(default="newest", pattern="^(newest|oldest|title)$"),
):
    if tag:
        rows = await asyncio.to_thread(db.get_documents_by_tag, tag, limit=500)
    else:
        rows = await asyncio.to_thread(
            db.list_documents,
            is_read=is_read,
            source_type=source_type,
            limit=500,
        )

    if source_type:
        rows = [r for r in rows if (r["source_type"] or "").lower() == source_type.lower()]

    if is_read is not None:
        expected = 1 if is_read else 0
        rows = [r for r in rows if r["is_read"] == expected]

    if sort == "oldest":
        rows = sorted(rows, key=lambda r: r["created_at"] or "")
    elif sort == "title":
        rows = sorted(rows, key=lambda r: (r["title"] or "").lower())
    else:
        pass

    total = len(rows)
    pages = max(1, math.ceil(total / per_page))
    page = min(page, pages)
    start = (page - 1) * per_page
    items = [_row_to_dict(r) for r in rows[start:start + per_page]]

    return DocumentListResponse(
        items=[DocumentOut.model_validate(i) for i in items],
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
    )


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: int):
    row = await asyncio.to_thread(db.get_document, doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut.model_validate(_row_to_dict(row))


@router.patch("/{doc_id}", response_model=DocumentOut)
async def update_document(doc_id: int, update: DocumentUpdate):
    if update.is_read is True:
        ok = await asyncio.to_thread(db.mark_read, doc_id)
    elif update.is_read is False:
        ok = await asyncio.to_thread(db.mark_unread, doc_id)
    else:
        raise HTTPException(status_code=400, detail="No fields to update")

    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")

    row = await asyncio.to_thread(db.get_document, doc_id)
    return DocumentOut.model_validate(_row_to_dict(row))


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: int):
    from pci.rag import async_delete_document

    await async_delete_document(doc_id)
    ok = await asyncio.to_thread(db.delete_document, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
