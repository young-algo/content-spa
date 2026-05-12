from typing import Optional

from fastapi import Query


def pagination_params(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=30, ge=1, le=200),
    is_read: Optional[bool] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    sort: str = Query(default="newest", pattern="^(newest|oldest|title)$"),
) -> dict:
    return {
        "page": page,
        "per_page": per_page,
        "is_read": is_read,
        "source_type": source_type,
        "tag": tag,
        "sort": sort,
    }


def search_params(
    q: str = Query(..., min_length=1),
    semantic: bool = Query(default=True),
    source_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    return {
        "q": q,
        "semantic": semantic,
        "source_type": source_type,
        "limit": limit,
    }
