import os
import asyncio
from pci.extractors import is_youtube_url, extract_youtube, extract_article, extract_pdf, extract_text_file, ExtractionError, RateLimitError
from pci.llm import summarize_and_tag
from pci.embeddings import get_embedding
from pci.db import insert_document
from rich.console import Console

console = Console()

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
        
    if not data.get('content') or not data['content'].strip() or "Could not retrieve transcript" in data['content']:
        console.print(f"[yellow]Warning: Could not extract useful content from {url}. Skipping ingestion to prevent hallucinations.[/yellow]")
        return
        
    console.print(f"[green]Extracted title:[/green] {data['title']} ({data['source_type']})")
    
    console.print("[cyan]Generating summary and tags via Claude Haiku...[/cyan]")
    llm_result = await summarize_and_tag(data['content'][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")
    
    console.print("[cyan]Generating embeddings...[/cyan]")
    embedding = await get_embedding(summary)
    
    console.print("[cyan]Storing locally...[/cyan]")
    doc_id = await asyncio.to_thread(
        insert_document,
        url=url,
        title=data['title'],
        source_type=data['source_type'],
        summary=summary,
        tags=tags,
        embedding=embedding
    )
    
    console.print(f"[bold green]Successfully ingested '{data['title']}' (ID: {doc_id})[/bold green]")

async def async_ingest_local_file(file_path: str):
    console.print(f"[bold blue]Processing Local File:[/bold blue] {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            data = await extract_pdf(file_path)
        elif ext in ['.md', '.markdown', '.txt']:
            data = await extract_text_file(file_path)
        else:
            console.print(f"[yellow]Unsupported file extension: {ext}. Skipping.[/yellow]")
            return
    except ExtractionError as e:
        console.print(f"[red]Extraction Error:[/red] {e}")
        console.print("[yellow]Skipping ingestion to prevent garbage database entries.[/yellow]")
        return
        
    if not data.get('content') or not data['content'].strip():
        console.print(f"[yellow]Warning: Could not extract useful content from {file_path}. Skipping ingestion.[/yellow]")
        return
        
    console.print(f"[green]Extracted title:[/green] {data['title']} ({data['source_type']})")
    
    console.print("[cyan]Generating summary and tags via Claude Haiku...[/cyan]")
    # PDF's could be very long, so truncate just like URLs
    llm_result = await summarize_and_tag(data['content'][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")
    
    console.print("[cyan]Generating embeddings...[/cyan]")
    embedding = await get_embedding(summary)
    
    console.print("[cyan]Storing locally...[/cyan]")
    # Use file_path as the URL for local files
    doc_id = await asyncio.to_thread(
        insert_document,
        url=f"file://{file_path}",
        title=data['title'],
        source_type=data['source_type'],
        summary=summary,
        tags=tags,
        embedding=embedding
    )
    
    console.print(f"[bold green]Successfully ingested '{data['title']}' (ID: {doc_id})[/bold green]")

