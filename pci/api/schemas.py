from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    url: str
    title: Optional[str] = None
    source_type: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[str] = None
    content: Optional[str] = None


class DocumentOut(DocumentBase):
    id: int
    is_read: int = 0
    read_at: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentListParams(BaseModel):
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=30, ge=1, le=200)
    is_read: Optional[bool] = None
    source_type: Optional[str] = None
    tag: Optional[str] = None
    sort: str = Field(default="newest", pattern="^(newest|oldest|title)$")


class DocumentUpdate(BaseModel):
    is_read: Optional[bool] = None


class DocumentListResponse(BaseModel):
    items: List[DocumentOut]
    total: int
    page: int
    per_page: int
    pages: int


class SearchParams(BaseModel):
    q: str = Field(..., min_length=1)
    semantic: bool = True
    source_type: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class SearchResult(BaseModel):
    id: int
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    summary: Optional[str] = None
    is_read: int = 0
    score: Optional[float] = None
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str
    semantic: bool


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    source_type: Optional[str] = None
    include_references: bool = True
    save: bool = False


class SynthesizeRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    response_type: str = "Comprehensive Markdown Article"
    save: bool = False


class LLMResponse(BaseModel):
    answer: str
    references: Optional[List[Dict[str, Any]]] = None


class IngestURLRequest(BaseModel):
    url: str = Field(..., min_length=1)


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, complete, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    total: int
    unread_count: int
    read_count: int
    by_source_type: List[Dict[str, Any]]
    top_tags: List[List[Any]]
    oldest_unread: Optional[Dict[str, Any]] = None


class TagCount(BaseModel):
    tag: str
    count: int


class TopicCluster(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str]


class TopicsResponse(BaseModel):
    tags: List[TagCount]
    clusters: Optional[List[TopicCluster]] = None
    cluster_created_at: Optional[str] = None


class ReindexRequest(BaseModel):
    reset: bool = True
    resume: bool = True


class ReindexResponse(BaseModel):
    indexed: int
    skipped: int
    total: int
    resumed: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class VaultExportRequest(BaseModel):
    vault_dir: str = Field(..., min_length=1, description="Absolute path to the Obsidian vault directory")
    no_content: bool = Field(default=False, description="If true, exclude full content; include summary only")
    source_type: Optional[str] = Field(default=None, description="Filter by source type (article, youtube, pdf, etc.)")
    limit: Optional[int] = Field(default=None, ge=1, description="Maximum number of documents to export")


class VaultExportResponse(BaseModel):
    exported: int
    skipped: int
    vault_dir: str
    output_dir: str
