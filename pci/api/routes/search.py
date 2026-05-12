import asyncio
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from pci import db
from pci.rag import (
    async_query_answer,
    async_query_data,
    build_search_results,
    filter_query_data_by_source_type,
)
from pci.api.schemas import (
    AskRequest,
    LLMResponse,
    SearchResponse,
    SearchResult,
    SynthesizeRequest,
)
from pci.api.deps import search_params
from pci.ingest import async_ingest_url

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    semantic: bool = Query(default=True),
    source_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    if semantic:
        fetch_limit = max(limit * 4, 40)
        raw_data = await async_query_data(
            q,
            mode="mix",
            top_k=fetch_limit,
        )
        filtered = filter_query_data_by_source_type(raw_data, source_type)
        results = build_search_results(filtered, source_type)[:limit]
        return SearchResponse(
            results=[SearchResult.model_validate(r) for r in results],
            total=len(results),
            query=q,
            semantic=True,
        )
    else:
        rows = await asyncio.to_thread(
            db.search_keyword,
            q,
            limit=limit,
            source_type=source_type,
        )
        results = []
        for row in rows:
            d = dict(row)
            results.append(SearchResult(
                id=d["id"],
                title=d.get("title"),
                url=d.get("url"),
                source_type=d.get("source_type"),
                summary=d.get("summary"),
                is_read=d.get("is_read", 0),
            ))
        return SearchResponse(
            results=results,
            total=len(results),
            query=q,
            semantic=False,
        )


@router.post("/ask", response_model=LLMResponse)
async def ask(req: AskRequest):
    try:
        result = await async_query_answer(
            req.question,
            response_type="Multiple Paragraphs",
            include_references=req.include_references,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = result.get("answer", "") or ""
    if isinstance(answer, str) and not answer.strip():
        answer = "No relevant information found in the index."

    return LLMResponse(answer=answer, references=result.get("raw_data"))


@router.post("/synthesize", response_model=LLMResponse)
async def synthesize(req: SynthesizeRequest):
    try:
        result = await async_query_answer(
            req.topic,
            response_type=req.response_type,
            include_references=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = result.get("answer", "") or ""

    if req.save:
        background_tasks = BackgroundTasks()
        doc_url = f"pci://synthesis/{req.topic.lower().replace(' ', '-')}"
        background_tasks.add_task(_save_synthesis, answer, req.topic, doc_url)

    return LLMResponse(answer=answer, references=result.get("raw_data"))


async def _save_synthesis(content: str, topic: str, url: str):
    from pci.llm import summarize_and_tag

    summary_result = await summarize_and_tag(content[:5000])
    summary = summary_result.get("summary", "")
    tags = summary_result.get("tags", [])
    await async_ingest_url(url)
