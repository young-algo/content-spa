import asyncio
import json
import logging
import os
import shutil
from typing import Any, Optional

import anthropic
import numpy as np

from pci.db import get_all_documents, get_documents_by_urls
from pci.embeddings import MODEL_NAME, get_embedding_dimension, get_embeddings


def _index_model() -> str:
    return os.environ.get(
        "PCI_LIGHTRAG_INDEX_MODEL",
        os.environ.get("PCI_LIGHTRAG_MODEL", "claude-haiku-4-5-20251001"),
    )


def _query_model() -> str:
    return os.environ.get("PCI_LIGHTRAG_QUERY_MODEL", "claude-sonnet-4-6")


def _working_dir() -> str:
    return os.environ.get("PCI_LIGHTRAG_DIR", ".pci_lightrag")


def _max_tokens() -> int:
    return int(os.environ.get("PCI_LIGHTRAG_MAX_TOKENS", "4000"))


def _temperature() -> float:
    return float(os.environ.get("PCI_LIGHTRAG_TEMPERATURE", "0.1"))

logging.getLogger("lightrag").setLevel(logging.WARNING)


REINDEX_STATE_FILENAME = "reindex_state.json"


def _reindex_state_path() -> str:
    return os.path.join(_working_dir(), REINDEX_STATE_FILENAME)


def _doc_status_path() -> str:
    return os.path.join(_working_dir(), "kv_store_doc_status.json")


def load_processed_doc_ids_from_lightrag() -> set[str]:
    path = _doc_status_path()
    if not os.path.exists(path):
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    processed_ids: set[str] = set()
    for doc_id, payload in data.items():
        if not str(doc_id).isdigit():
            continue
        if isinstance(payload, dict) and payload.get("status") == "processed":
            processed_ids.add(str(doc_id))
    return processed_ids


def load_reindex_state() -> dict[str, Any]:
    path = _reindex_state_path()
    completed_ids = set(load_processed_doc_ids_from_lightrag())
    last_completed_id: Optional[str] = None

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed_ids.update(str(item) for item in data.get("completed_ids", []) if str(item).isdigit())
            raw_last_completed_id = data.get("last_completed_id")
            last_completed_id = str(raw_last_completed_id) if raw_last_completed_id is not None else None
        except Exception:
            pass

    if last_completed_id is None and completed_ids:
        last_completed_id = max(completed_ids, key=int)

    return {
        "completed_ids": sorted(completed_ids, key=int),
        "last_completed_id": last_completed_id,
    }


def save_reindex_state(completed_ids: set[str], last_completed_id: Optional[str]) -> None:
    os.makedirs(_working_dir(), exist_ok=True)
    with open(_reindex_state_path(), "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_ids": sorted(completed_ids, key=lambda value: int(value) if value.isdigit() else value),
                "last_completed_id": last_completed_id,
            },
            f,
            indent=2,
        )


def clear_reindex_state() -> None:
    path = _reindex_state_path()
    if os.path.exists(path):
        os.remove(path)


def rag_settings() -> dict[str, str]:
    state = load_reindex_state()
    return {
        "working_dir": _working_dir(),
        "index_model": _index_model(),
        "query_model": _query_model(),
        "max_tokens": str(_max_tokens()),
        "temperature": str(_temperature()),
        "anthropic_api_key_present": "yes" if os.environ.get("ANTHROPIC_API_KEY") else "no",
        "reindex_state_path": _reindex_state_path(),
        "resume_completed_count": str(len(state.get("completed_ids", []))),
        "resume_last_completed_id": str(state.get("last_completed_id") or "-"),
    }


async def _anthropic_complete_for_model(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    stream: Optional[bool] = False,
    **_: Any,
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for LightRAG indexing and semantic retrieval.")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages: list[dict[str, Any]] = []

    for message in history_messages or []:
        role = message.get("role", "user")
        if role not in {"user", "assistant"}:
            role = "user"
        content = str(message.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    system = system_prompt if system_prompt else anthropic.NOT_GIVEN

    if stream:
        async def generator():
            async with client.messages.stream(
                model=model,
                max_tokens=_max_tokens(),
                temperature=_temperature(),
                system=system,
                messages=messages,
            ) as response_stream:
                async for text in response_stream.text_stream:
                    yield text

        return generator()

    response = await client.messages.create(
        model=model,
        max_tokens=_max_tokens(),
        temperature=_temperature(),
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


async def _anthropic_index_complete(*args: Any, **kwargs: Any):
    return await _anthropic_complete_for_model(_index_model(), *args, **kwargs)


async def _anthropic_query_complete(*args: Any, **kwargs: Any):
    return await _anthropic_complete_for_model(_query_model(), *args, **kwargs)


async def _batch_embed(texts: list[str]) -> np.ndarray:
    embeddings = await get_embeddings(texts)
    return np.array(embeddings, dtype=np.float32)


async def _noop_embed(texts: list[str], embedding_dim: int) -> np.ndarray:
    return np.zeros((len(texts), embedding_dim), dtype=np.float32)


def _vector_db_paths() -> list[str]:
    working_dir = _working_dir()
    return [
        os.path.join(working_dir, "vdb_chunks.json"),
        os.path.join(working_dir, "vdb_entities.json"),
        os.path.join(working_dir, "vdb_relationships.json"),
    ]


def _has_lightrag_artifacts() -> bool:
    working_dir = _working_dir()
    if not os.path.exists(working_dir):
        return False

    interesting_files = {
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_text_chunks.json",
        "graph_chunk_entity_relation.graphml",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    }
    return any(os.path.exists(os.path.join(working_dir, name)) for name in interesting_files)


def _stored_embedding_dimension() -> Optional[int]:
    for path in _vector_db_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        embedding_dim = payload.get("embedding_dim") if isinstance(payload, dict) else None
        if isinstance(embedding_dim, int) and embedding_dim > 0:
            return embedding_dim
    return None


async def _create_rag(*, require_embedding_api: bool = True):
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG is not installed. Run `uv sync` to install the lightrag-hku dependency."
        ) from exc

    cleanup_paths: list[str] = []
    if require_embedding_api:
        embedding_dim = await get_embedding_dimension()
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            model_name=MODEL_NAME,
            func=_batch_embed,
        )
    else:
        embedding_dim = _stored_embedding_dimension()
        if embedding_dim is None:
            cleanup_paths = [path for path in _vector_db_paths() if not os.path.exists(path)]
            embedding_dim = 1
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            model_name=MODEL_NAME,
            func=lambda texts: _noop_embed(texts, embedding_dim),
        )

    rag = LightRAG(
        working_dir=_working_dir(),
        embedding_func=embedding_func,
        llm_model_func=_anthropic_index_complete,
        llm_model_name=_index_model(),
    )
    await rag.initialize_storages()
    return rag, cleanup_paths


async def _run_with_rag(callback, *, require_embedding_api: bool = True):
    rag, cleanup_paths = await _create_rag(require_embedding_api=require_embedding_api)
    try:
        return await callback(rag)
    finally:
        try:
            await rag.finalize_storages()
        finally:
            for path in cleanup_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass


def build_rag_document(
    *,
    title: str,
    url: str,
    source_type: str,
    summary: str,
    tags: list[str],
    content: str,
) -> str:
    header = [
        f"Title: {title}",
        f"Source: {url}",
        f"Type: {source_type}",
    ]
    if tags:
        header.append(f"Tags: {', '.join(tags)}")
    if summary:
        header.append(f"Summary: {summary}")

    return "\n".join([
        "# Document Metadata",
        *header,
        "",
        "# Document Content",
        content,
    ])


async def _refresh_document_id(doc_id: int) -> None:
    if not _has_lightrag_artifacts():
        return

    deleted, error = await async_delete_document(doc_id)
    if not deleted:
        raise RuntimeError(error or f"Failed to refresh existing LightRAG document {doc_id} before reindexing.")


async def async_index_document(
    *,
    doc_id: int,
    title: str,
    url: str,
    source_type: str,
    summary: str,
    tags: list[str],
    content: str,
) -> str:
    document_text = build_rag_document(
        title=title,
        url=url,
        source_type=source_type,
        summary=summary,
        tags=tags,
        content=content,
    )

    await _refresh_document_id(doc_id)

    async def _index(rag):
        return await rag.ainsert(document_text, ids=str(doc_id), file_paths=url)

    return await _run_with_rag(_index)


def _coerce_delete_result(result: Any) -> tuple[bool, Optional[str]]:
    status = getattr(result, "status", None)
    message = getattr(result, "message", None)
    if isinstance(result, dict):
        status = result.get("status", status)
        message = result.get("message", message)

    if status in {"success", "not_found"}:
        return True, None
    if status:
        return False, message or f"LightRAG deletion returned status '{status}'."
    return False, message or "LightRAG deletion did not report success."


async def async_delete_document(doc_id: int) -> tuple[bool, Optional[str]]:
    async def _delete(rag):
        return await rag.adelete_by_doc_id(str(doc_id))

    if not _has_lightrag_artifacts():
        return True, None

    try:
        result = await _run_with_rag(_delete, require_embedding_api=False)
    except Exception as exc:
        return False, str(exc)

    return _coerce_delete_result(result)


async def async_query_data(
    query: str,
    *,
    mode: str = "mix",
    top_k: int = 5,
    chunk_top_k: Optional[int] = None,
) -> dict[str, Any]:
    async def _query(rag):
        from lightrag import QueryParam

        return await rag.aquery_data(
            query,
            QueryParam(
                mode=mode,
                top_k=max(top_k, 1),
                chunk_top_k=max(chunk_top_k or top_k, 1),
                model_func=_anthropic_query_complete,
            ),
        )

    return await _run_with_rag(_query)


async def async_query_answer(
    query: str,
    *,
    mode: str = "mix",
    top_k: int = 5,
    chunk_top_k: Optional[int] = None,
    response_type: str = "Multiple Paragraphs",
    include_references: bool = False,
) -> dict[str, Any]:
    async def _query(rag):
        from lightrag import QueryParam

        result = await rag.aquery_llm(
            query,
            QueryParam(
                mode=mode,
                top_k=max(top_k, 1),
                chunk_top_k=max(chunk_top_k or top_k, 1),
                response_type=response_type,
                model_func=_anthropic_query_complete,
                include_references=include_references,
            ),
        )
        llm_response = result.get("llm_response", {})
        return {
            "answer": llm_response.get("content") or "",
            "raw_data": {
                "status": result.get("status"),
                "message": result.get("message"),
                "data": result.get("data", {}),
                "metadata": result.get("metadata", {}),
            },
        }

    return await _run_with_rag(_query)


async def async_reset_rag_index() -> None:
    working_dir = _working_dir()
    if os.path.exists(working_dir):
        await asyncio.to_thread(shutil.rmtree, working_dir)
    await asyncio.to_thread(clear_reindex_state)


async def async_reindex_all_documents(reset: bool = True, resume: bool = True) -> dict[str, int | bool]:
    documents = await asyncio.to_thread(get_all_documents)

    if reset:
        await async_reset_rag_index()

    state = await asyncio.to_thread(load_reindex_state)
    completed_ids = set(state.get("completed_ids", [])) if resume and not reset else set()
    last_completed_id = state.get("last_completed_id") if resume and not reset else None
    documents_to_index = [doc for doc in documents if str(doc["id"]) not in completed_ids]

    async def _reindex(rag):
        nonlocal last_completed_id
        indexed = 0
        skipped = len(documents) - len(documents_to_index)
        for doc in documents_to_index:
            doc_id = str(doc["id"])
            delete_result = await rag.adelete_by_doc_id(doc_id)
            deleted, delete_error = _coerce_delete_result(delete_result)
            if not deleted:
                raise RuntimeError(delete_error or f"Failed to refresh LightRAG document {doc_id} before reindexing.")

            await rag.ainsert(
                build_rag_document(
                    title=doc["title"] or doc["url"],
                    url=doc["url"],
                    source_type=doc["source_type"] or "unknown",
                    summary=doc["summary"] or "",
                    tags=[tag.strip() for tag in (doc["tags"] or "").split(",") if tag.strip()],
                    content=doc["content"] or "",
                ),
                ids=doc_id,
                file_paths=doc["url"],
            )
            completed_ids.add(doc_id)
            last_completed_id = doc_id
            indexed += 1
            await asyncio.to_thread(save_reindex_state, completed_ids, last_completed_id)
        return {
            "indexed": indexed,
            "skipped": skipped,
            "total": len(documents),
            "resumed": bool(resume and not reset),
        }

    result = await _run_with_rag(_reindex)
    await asyncio.to_thread(save_reindex_state, completed_ids, last_completed_id)
    return result


def _reference_path(item: dict[str, Any], ref_map: dict[str, str]) -> Optional[str]:
    file_path = item.get("file_path")
    if file_path:
        return file_path
    reference_id = item.get("reference_id")
    if reference_id:
        return ref_map.get(reference_id)
    return None


def filter_query_data_by_source_type(raw_data: dict[str, Any], source_type: Optional[str]) -> dict[str, Any]:
    if not source_type or raw_data.get("status") != "success":
        return raw_data

    data = raw_data.get("data", {})
    refs = data.get("references", [])
    ref_map = {ref.get("reference_id"): ref.get("file_path") for ref in refs if ref.get("reference_id")}
    url_map = get_documents_by_urls([ref.get("file_path") for ref in refs if ref.get("file_path")])
    allowed_urls = {
        url
        for url, doc in url_map.items()
        if (doc["source_type"] or "").lower() == source_type.lower()
    }

    filtered_refs = [ref for ref in refs if ref.get("file_path") in allowed_urls]
    allowed_ref_ids = {ref.get("reference_id") for ref in filtered_refs if ref.get("reference_id")}

    def _keep(item: dict[str, Any]) -> bool:
        path = _reference_path(item, ref_map)
        if path in allowed_urls:
            return True
        reference_id = item.get("reference_id")
        return reference_id in allowed_ref_ids

    filtered = {
        **raw_data,
        "data": {
            "entities": [item for item in data.get("entities", []) if _keep(item)],
            "relationships": [item for item in data.get("relationships", []) if _keep(item)],
            "chunks": [item for item in data.get("chunks", []) if _keep(item)],
            "references": filtered_refs,
        },
    }
    return filtered


def build_search_results(raw_data: dict[str, Any], source_type: Optional[str] = None) -> list[dict[str, Any]]:
    filtered = filter_query_data_by_source_type(raw_data, source_type)
    if filtered.get("status") != "success":
        return []

    data = filtered.get("data", {})
    references = data.get("references", [])
    ref_map = {ref.get("reference_id"): ref.get("file_path") for ref in references if ref.get("reference_id")}
    docs_by_url = get_documents_by_urls([ref.get("file_path") for ref in references if ref.get("file_path")])

    grouped: dict[str, dict[str, Any]] = {}

    for ref in references:
        url = ref.get("file_path")
        if not url:
            continue
        doc = docs_by_url.get(url)
        grouped[url] = {
            "id": doc["id"] if doc else None,
            "url": url,
            "title": doc["title"] if doc else url,
            "source_type": doc["source_type"] if doc else None,
            "summary": doc["summary"] if doc else None,
            "is_read": doc["is_read"] if doc else 0,
            "read_at": doc["read_at"] if doc else None,
            "chunk_count": 0,
            "entity_count": 0,
            "relationship_count": 0,
            "snippet": None,
        }

    for chunk in data.get("chunks", []):
        url = _reference_path(chunk, ref_map)
        if not url or url not in grouped:
            continue
        grouped[url]["chunk_count"] += 1
        if not grouped[url]["snippet"]:
            grouped[url]["snippet"] = str(chunk.get("content", "")).strip().replace("\n", " ")

    for entity in data.get("entities", []):
        url = _reference_path(entity, ref_map)
        if url and url in grouped:
            grouped[url]["entity_count"] += 1

    for relationship in data.get("relationships", []):
        url = _reference_path(relationship, ref_map)
        if url and url in grouped:
            grouped[url]["relationship_count"] += 1

    results = list(grouped.values())
    results.sort(
        key=lambda item: (
            item["chunk_count"],
            item["entity_count"],
            item["relationship_count"],
            1 if item["id"] is not None else 0,
        ),
        reverse=True,
    )
    return results
