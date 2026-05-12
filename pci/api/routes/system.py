import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from pci import db
from pci.api.schemas import (
    ReindexRequest,
    ReindexResponse,
    StatsResponse,
    TagCount,
    TopicsResponse,
    HealthResponse,
    TaskStatus,
)
from pci.api.tasks import create_task, get_task, run_task, update_task
from pci.llm import cluster_tags

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/stats", response_model=StatsResponse)
async def stats():
    return await asyncio.to_thread(db.get_stats)


@router.get("/topics", response_model=TopicsResponse)
async def topics(
    cluster: bool = Query(default=False),
    source_type: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
):
    all_tags = await asyncio.to_thread(db.get_all_tags)
    counter: dict[str, int] = {}
    for row in all_tags:
        tags_string = row[1] or ""
        for tag in tags_string.split(","):
            tag = tag.strip().lower()
            if tag:
                counter[tag] = counter.get(tag, 0) + 1

    if source_type:
        from pci.db import get_db
        conn = db.get_db()
        cursor = conn.cursor()
        filtered: dict[str, int] = {}
        for tag, total_count in counter.items():
            cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE tags LIKE ? AND LOWER(source_type) = LOWER(?)",
                (f"%{tag}%", source_type),
            )
            filtered_count = cursor.fetchone()[0]
            if filtered_count:
                filtered[tag] = filtered_count
        conn.close()
        counter = filtered

    sorted_tags = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    tags = [TagCount(tag=t, count=c) for t, c in sorted_tags[:100]]

    clusters = None
    cluster_created_at = None
    if cluster:
        cache_path = os.path.join(
            os.path.dirname(os.environ.get("PCI_DB_PATH", "pci.db")),
            ".pci_topic_clusters.json",
        )

        if not refresh and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                clusters = cache.get("clusters") or None
                cluster_created_at = cache.get("created_at")
            except Exception:
                clusters = None

        if clusters is None:
            top_50 = [(t, c) for t, c in sorted_tags[:50]]
            if top_50:
                try:
                    cluster_results = await cluster_tags(top_50)
                    clusters = cluster_results
                    cluster_created_at = datetime.now().isoformat()
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"created_at": cluster_created_at, "clusters": clusters},
                            f,
                            indent=2,
                        )
                except Exception:
                    clusters = None

    return TopicsResponse(tags=tags, clusters=clusters, cluster_created_at=cluster_created_at)


@router.post("/reindex", response_model=TaskStatus)
async def reindex(req: ReindexRequest, background_tasks: BackgroundTasks):
    task_id = create_task()
    background_tasks.add_task(run_task, task_id, _reindex_task(req.reset, req.resume, task_id))
    return TaskStatus.model_validate(get_task(task_id))


async def _reindex_task(reset: bool, resume: bool, task_id: str):
    from pci.rag import async_reindex_all_documents

    update_task(task_id, message="Reindexing documents...", progress=10)
    result = await async_reindex_all_documents(reset=reset, resume=resume)
    update_task(task_id, message="Reindex complete", result=result)
    return result


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus.model_validate(task)
