import asyncio
import os

from rich.console import Console

from pci.db import insert_document
from pci.extractors import (
    ExtractionError,
    RateLimitError,
    extract_article,
    extract_pdf,
    extract_text_file,
    extract_youtube,
    is_youtube_url,
)
from pci.llm import summarize_and_tag
from pci.rag import async_index_document

console = Console()


async def _store_and_index_document(*, url: str, data: dict, summary: str, tags: list[str]) -> int:
    console.print("[cyan]Storing metadata locally...[/cyan]")
    doc_id = await asyncio.to_thread(
        insert_document,
        url=url,
        title=data["title"],
        source_type=data["source_type"],
        summary=summary,
        tags=tags,
        content=data["content"],
    )

    console.print("[cyan]Indexing with LightRAG...[/cyan]")
    await async_index_document(
        doc_id=doc_id,
        title=data["title"],
        url=url,
        source_type=data["source_type"],
        summary=summary,
        tags=tags,
        content=data["content"],
    )
    return doc_id


async def async_ingest_url(url: str):
    console.print(f"[bold blue]Processing:[/bold blue] {url}")

    try:
        if is_youtube_url(url):
            data = await extract_youtube(url)
        else:
            data = await extract_article(url)
    except RateLimitError as e:
        console.print(f"[red]Rate Limit Error:[/red] {e}")
        console.print("[yellow]Skipping ingestion to prevent empty database entries.[/yellow]")
        return
    except ExtractionError as e:
        console.print(f"[red]Extraction Error:[/red] {e}")
        console.print("[yellow]Skipping ingestion to prevent garbage database entries.[/yellow]")
        return

    if not data.get("content") or not data["content"].strip() or "Could not retrieve transcript" in data["content"]:
        console.print(f"[yellow]Warning: Could not extract useful content from {url}. Skipping ingestion to prevent hallucinations.[/yellow]")
        return

    console.print(f"[green]Extracted title:[/green] {data['title']} ({data['source_type']})")

    console.print("[cyan]Generating summary and tags via Claude Haiku...[/cyan]")
    llm_result = await summarize_and_tag(data["content"][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")

    doc_id = await _store_and_index_document(url=url, data=data, summary=summary, tags=tags)
    console.print(f"[bold green]Successfully ingested '{data['title']}' (ID: {doc_id})[/bold green]")


async def async_ingest_local_file(file_path: str):
    console.print(f"[bold blue]Processing Local File:[/bold blue] {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            data = await extract_pdf(file_path)
        elif ext in [".md", ".markdown", ".txt"]:
            data = await extract_text_file(file_path)
        else:
            console.print(f"[yellow]Unsupported file extension: {ext}. Skipping.[/yellow]")
            return
    except ExtractionError as e:
        console.print(f"[red]Extraction Error:[/red] {e}")
        console.print("[yellow]Skipping ingestion to prevent garbage database entries.[/yellow]")
        return

    if not data.get("content") or not data["content"].strip():
        console.print(f"[yellow]Warning: Could not extract useful content from {file_path}. Skipping ingestion.[/yellow]")
        return

    console.print(f"[green]Extracted title:[/green] {data['title']} ({data['source_type']})")

    console.print("[cyan]Generating summary and tags via Claude Haiku...[/cyan]")
    llm_result = await summarize_and_tag(data["content"][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")

    file_url = f"file://{file_path}"
    doc_id = await _store_and_index_document(url=file_url, data=data, summary=summary, tags=tags)
    console.print(f"[bold green]Successfully ingested '{data['title']}' (ID: {doc_id})[/bold green]")
