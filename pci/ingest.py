from pci.extractors import is_youtube_url, extract_youtube, extract_article, extract_pdf, ExtractionError, RateLimitError
from pci.llm import summarize_and_tag
from pci.embeddings import get_embedding
from pci.db import insert_document
from rich.console import Console

console = Console()

def ingest_url(url: str):
    console.print(f"[bold blue]Processing:[/bold blue] {url}")
    
    try:
        if is_youtube_url(url):
            data = extract_youtube(url)
        else:
            data = extract_article(url)
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
    llm_result = summarize_and_tag(data['content'][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")
    
    console.print("[cyan]Generating embeddings...[/cyan]")
    embedding = get_embedding(summary)
    
    console.print("[cyan]Storing locally...[/cyan]")
    doc_id = insert_document(
        url=url,
        title=data['title'],
        source_type=data['source_type'],
        summary=summary,
        tags=tags,
        embedding=embedding
    )
    
    console.print(f"[bold green]Successfully ingested '{data['title']}' (ID: {doc_id})[/bold green]")

def ingest_pdf(file_path: str):
    console.print(f"[bold blue]Processing PDF:[/bold blue] {file_path}")
    
    try:
        data = extract_pdf(file_path)
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
    llm_result = summarize_and_tag(data['content'][:50000])
    summary = llm_result.get("summary", "No summary generated.")
    tags = llm_result.get("tags", [])
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")
    
    console.print("[cyan]Generating embeddings...[/cyan]")
    embedding = get_embedding(summary)
    
    console.print("[cyan]Storing locally...[/cyan]")
    # Use file_path as the URL for local files
    doc_id = insert_document(
        url=f"file://{file_path}",
        title=data['title'],
        source_type=data['source_type'],
        summary=summary,
        tags=tags,
        embedding=embedding
    )
    
    console.print(f"[bold green]Successfully ingested PDF '{data['title']}' (ID: {doc_id})[/bold green]")

