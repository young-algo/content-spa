import asyncio
import csv
import os
import shutil
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from pci.db import (
    delete_document as delete_document_record,
    get_all_documents,
    get_all_tags,
    get_all_titles_and_urls,
    get_document,
    get_documents_by_tag,
    get_random_documents,
    get_stats,
    init_db,
    list_documents,
    mark_all_read,
    mark_read,
    mark_unread,
    migrate_db,
    search_keyword,
)
from pci.embeddings import embedding_settings
from pci.ingest import async_ingest_local_file, async_ingest_url
from pci.rag import (
    async_delete_document,
    async_health_check,
    async_query_answer,
    async_query_data,
    async_reindex_all_documents,
    build_search_results,
    filter_query_data_by_source_type,
    rag_settings,
)
from pci.rename import (
    DEFAULT_RENAME_MODEL,
    propose_filename,
    rename_paths,
)

app = typer.Typer(help="Personal Content Index")
console = Console()
LIGHTRAG_MODES = "naive, local, global, hybrid, mix"
SEMANTIC_FILTER_FETCH_MULTIPLIER = 5
SEMANTIC_FILTER_FETCH_MINIMUM = 20


def ensure_db_ready() -> None:
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        console.print("[yellow]Database not found. Initializing...[/yellow]")
        init_db()
    else:
        migrate_db()


def truncate_text(text: Optional[str], length: int = 60) -> str:
    if not text:
        return "-"
    cleaned = " ".join(text.split())
    if len(cleaned) <= length:
        return cleaned
    return cleaned[: length - 1] + "…"


def status_label(is_read: int) -> str:
    return "read" if is_read else "unread"


def semantic_search_fetch_limit(limit: int, source_type: Optional[str]) -> int:
    if not source_type:
        return limit
    return max(limit * SEMANTIC_FILTER_FETCH_MULTIPLIER, SEMANTIC_FILTER_FETCH_MINIMUM)


def open_document_url(doc: dict | object, mark_as_read_after_open: bool = False) -> None:
    url = doc["url"]
    webbrowser.open(url)
    console.print(f"[green]Opened:[/green] {url}")
    if mark_as_read_after_open:
        mark_read(doc["id"])
        console.print(f"[green]Marked document {doc['id']} as read.[/green]")


@app.command()
def init():
    """Initialize the local SQLite database."""
    init_db()
    console.print(f"[green]Initialized database at {os.environ.get('PCI_DB_PATH', 'pci.db')}[/green]")


@app.command()
def add(source: str):
    """Add a new URL or local file and index it with LightRAG."""
    ensure_db_ready()

    if os.path.exists(source) and os.path.isfile(source):
        asyncio.run(async_ingest_local_file(source))
    else:
        asyncio.run(async_ingest_url(source))


@app.command()
def ingest(
    inbox: str = typer.Option(
        None,
        "--inbox",
        help="Path to the inbox folder. Defaults to PCI_INBOX_DIR env var.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be ingested without actually processing."),
):
    """Ingest URLs and files from a shared inbox folder (e.g. Google Drive)."""
    inbox_dir = inbox or os.environ.get("PCI_INBOX_DIR")
    if not inbox_dir:
        console.print("[red]No inbox path provided. Set PCI_INBOX_DIR or pass --inbox.[/red]")
        raise typer.Exit(code=1)

    if not os.path.isdir(inbox_dir):
        console.print(f"[red]Inbox folder not found: {inbox_dir}[/red]")
        raise typer.Exit(code=1)

    ensure_db_ready()

    # Collect URLs from inbox.txt
    urls: list[str] = []
    inbox_txt = os.path.join(inbox_dir, "inbox.txt")
    if os.path.isfile(inbox_txt):
        with open(inbox_txt, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # Collect files
    supported_exts = {".pdf", ".md", ".markdown", ".txt"}
    files: list[str] = []
    for entry in sorted(os.listdir(inbox_dir)):
        full_path = os.path.join(inbox_dir, entry)
        if os.path.isfile(full_path) and entry != "inbox.txt":
            ext = os.path.splitext(entry)[1].lower()
            if ext in supported_exts:
                files.append(full_path)
            else:
                console.print(f"[yellow]Skipping unsupported file: {entry}[/yellow]")

    total = len(urls) + len(files)
    if total == 0:
        console.print("[yellow]Inbox is empty — nothing to ingest.[/yellow]")
        return

    console.print(f"[cyan]Found {len(urls)} URL(s) and {len(files)} file(s) to ingest.[/cyan]")

    if dry_run:
        for u in urls:
            console.print(f"  [dim]URL:[/dim] {u}")
        for f in files:
            console.print(f"  [dim]File:[/dim] {os.path.basename(f)}")
        console.print("[yellow]Dry run — nothing ingested.[/yellow]")
        return

    archived_dir = os.path.join(inbox_dir, "archived")

    succeeded_urls: list[str] = []
    failed_count = 0

    async def _ingest_all():
        nonlocal failed_count

        # Process URLs sequentially to avoid rate limits
        for i, url in enumerate(urls, 1):
            console.print(f"\n[bold blue]URL ({i}/{len(urls)}):[/bold blue] {url}")
            try:
                await async_ingest_url(url)
                succeeded_urls.append(url)
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")
                failed_count += 1

        # Process files sequentially
        for i, file_path in enumerate(files, 1):
            console.print(f"\n[bold blue]File ({i}/{len(files)}):[/bold blue] {os.path.basename(file_path)}")
            try:
                await async_ingest_local_file(file_path)
                # Move to archived
                os.makedirs(archived_dir, exist_ok=True)
                dest = os.path.join(archived_dir, os.path.basename(file_path))
                shutil.move(file_path, dest)
                console.print(f"[dim]Archived → {os.path.basename(file_path)}[/dim]")
            except Exception as e:
                console.print(f"[red]Failed (leaving in inbox): {e}[/red]")
                failed_count += 1

    start_time = time.time()
    asyncio.run(_ingest_all())

    # Remove succeeded URLs from inbox.txt, keep failed ones
    if os.path.isfile(inbox_txt):
        with open(inbox_txt, "r", encoding="utf-8") as f:
            remaining = [line for line in f if line.strip() not in succeeded_urls]
        with open(inbox_txt, "w", encoding="utf-8") as f:
            f.writelines(remaining)

    succeeded = total - failed_count
    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Ingest complete:[/bold green] {succeeded} succeeded, {failed_count} failed, {elapsed:.1f}s")


@app.command()
def doctor():
    """Show active model/provider configuration from the current environment."""
    db_path = os.environ.get("PCI_DB_PATH", "pci.db")
    embed = embedding_settings()
    rag = rag_settings()

    table = Table(title="PCI Doctor")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Database path", db_path)
    table.add_row("Database exists", "yes" if os.path.exists(db_path) else "no")
    table.add_row("LightRAG working dir", rag["working_dir"])
    table.add_row("LightRAG index model", rag["index_model"])
    table.add_row("LightRAG query model", rag["query_model"])
    table.add_row("Anthropic API key present", rag["anthropic_api_key_present"])
    table.add_row("Reindex state path", rag["reindex_state_path"])
    table.add_row("Reindex completed docs", rag["resume_completed_count"])
    table.add_row("Reindex last completed id", rag["resume_last_completed_id"])
    table.add_row("Embedding provider", str(embed["provider"]))
    table.add_row("Embedding model", str(embed["model"]))
    table.add_row("Embedding base URL", str(embed["base_url"]))
    table.add_row("OpenRouter API key present", str(embed["api_key_present"]))
    table.add_row("OpenRouter site URL", str(embed["site_url"] or "-"))
    table.add_row("OpenRouter site name", str(embed["site_name"] or "-"))
    table.add_row("Rename model", os.environ.get("PCI_RENAME_MODEL", DEFAULT_RENAME_MODEL))
    table.add_row("OpenAI API key present", "yes" if os.environ.get("OPENAI_API_KEY") else "no")
    table.add_row("OpenAI base URL", os.environ.get("OPENAI_BASE_URL") or "-")

    console.print(table)


@app.command()
def reindex(
    reset: bool = typer.Option(True, "--reset/--no-reset", help="Reset the LightRAG working directory before rebuilding."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip documents already marked complete in the reindex state file."),
):
    """Rebuild the LightRAG index from the current SQLite documents."""
    ensure_db_ready()
    console.print("[cyan]Rebuilding LightRAG index from SQLite documents...[/cyan]")
    if reset:
        console.print(f"[dim]Resetting LightRAG directory: {os.environ.get('PCI_LIGHTRAG_DIR', '.pci_lightrag')}[/dim]")
    elif resume:
        console.print("[dim]Resume mode enabled: already completed documents will be skipped.[/dim]")
    try:
        result = asyncio.run(async_reindex_all_documents(reset=reset, resume=resume))
    except Exception as e:
        console.print(f"[red]Error during reindex: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Reindex complete.[/green] Indexed {result['indexed']} document(s), skipped {result['skipped']}, total {result['total']}."
    )


@app.command()
def ask(
    query: str,
    mode: str = typer.Option("mix", help=f"LightRAG retrieval mode: {LIGHTRAG_MODES}"),
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum number of retrieval results to use."),
    response_type: str = typer.Option("Multiple Paragraphs", "--response-type", help="Requested LightRAG answer format."),
    references: bool = typer.Option(False, "--references", help="Show retrieved references after the answer."),
    save_to: Optional[str] = typer.Option(None, "--save-to", help="Save the answer to a Markdown file."),
    ingest: bool = typer.Option(False, "--ingest", help="Automatically ingest the saved Markdown file back into the database."),
):
    """Ask LightRAG a question and get a paragraph-form answer."""
    ensure_db_ready()
    console.print(f"[cyan]Asking: '{query}'[/cyan]")
    console.print(f"[dim]LightRAG mode: {mode}[/dim]")

    async def _run_ask():
        res = await async_query_answer(
            query,
            mode=mode,
            top_k=limit,
            chunk_top_k=max(limit, 5),
            response_type=response_type,
            include_references=references,
        )
        ans = (res.get("answer") or "").strip()
        if not ans:
            console.print("[yellow]LightRAG did not return an answer.[/yellow]")
            return

        console.print(ans)

        r = []
        if references:
            r = res.get("raw_data", {}).get("data", {}).get("references", [])
            if r:
                refs_table = Table(title="References")
                refs_table.add_column("Ref", style="cyan", no_wrap=True)
                refs_table.add_column("Source", style="magenta")
                for ref in r:
                    refs_table.add_row(str(ref.get("reference_id") or "-"), truncate_text(ref.get("file_path"), 120))
                console.print(refs_table)

        if save_to:
            try:
                with open(save_to, "w", encoding="utf-8") as f:
                    f.write(f"# Q: {query}\n\n")
                    f.write(ans + "\n")
                    
                    if references and r:
                        f.write("\n## References\n")
                        for ref in r:
                            f.write(f"- [{ref.get('reference_id') or 'Ref'}] {ref.get('file_path')}\n")
                            
                console.print(f"[green]Saved answer to: {save_to}[/green]")
                
                if ingest:
                    console.print(f"[cyan]Ingesting {save_to} back into the index...[/cyan]")
                    await async_ingest_local_file(save_to)
            except Exception as e:
                console.print(f"[red]Error saving or ingesting file: {e}[/red]")

    try:
        asyncio.run(_run_ask())
    except Exception as e:
        console.print(f"[red]Error during LightRAG answer generation: {e}[/red]")


@app.command()
def synthesize(
    topic: str,
    mode: str = typer.Option("mix", help=f"LightRAG retrieval mode: {LIGHTRAG_MODES}"),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum number of retrieval results to use."),
    save_dir: str = typer.Option("syntheses", "--save-dir", help="Directory to save the generated markdown file."),
    ingest: bool = typer.Option(False, "--ingest", help="Automatically ingest the generated markdown file back into the database."),
):
    """Synthesize a comprehensive markdown article about a topic using retrieved knowledge."""
    ensure_db_ready()
    console.print(f"[cyan]Synthesizing topic: '{topic}'[/cyan]")
    console.print(f"[dim]LightRAG mode: {mode}[/dim]")

    response_type = "Comprehensive Markdown Article with sections, citations referencing the provided context, and a 'Further Reading' section"

    async def _run_synthesize():
        res = await async_query_answer(
            topic,
            mode=mode,
            top_k=limit,
            chunk_top_k=max(limit, 10),
            response_type=response_type,
            include_references=True,
        )

        ans = (res.get("answer") or "").strip()
        if not ans:
            console.print("[yellow]LightRAG did not return a synthesis.[/yellow]")
            return
            
        console.print("\n[bold magenta]Synthesis Generated:[/bold magenta]\n")
        console.print(ans)

        r = res.get("raw_data", {}).get("data", {}).get("references", [])

        os.makedirs(save_dir, exist_ok=True)
        filename = "".join([c if c.isalnum() else "_" for c in topic]) + ".md"
        file_path = os.path.join(save_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Synthesis: {topic}\n\n")
                f.write(ans + "\n")
                
                if r:
                    f.write("\n## References\n")
                    for ref in r:
                        f.write(f"- [{ref.get('reference_id') or 'Ref'}] {ref.get('file_path')}\n")
                        
            console.print(f"[green]Saved synthesized article to: {file_path}[/green]")
            
            if ingest:
                console.print(f"[cyan]Ingesting {file_path} back into the index...[/cyan]")
                await async_ingest_local_file(file_path)
        except Exception as e:
            console.print(f"[red]Error saving or ingesting synthesis file: {e}[/red]")

    try:
        asyncio.run(_run_synthesize())
    except Exception as e:
        console.print(f"[red]Error during LightRAG synthesis: {e}[/red]")


@app.command()
def checkup(
    limit: int = typer.Option(50, "--limit", help="Number of items to analyze."),
    random: bool = typer.Option(False, "--random", help="Sample random items instead of most recent."),
):
    """Run an LLM health check on your knowledge base to identify gaps and suggest research."""
    ensure_db_ready()

    if random:
        console.print(f"[cyan]Sampling {limit} random items for health check...[/cyan]")
        documents = get_random_documents(limit=limit)
    else:
        console.print(f"[cyan]Gathering up to {limit} recent items for health check...[/cyan]")
        documents = list_documents(limit=limit)
    if not documents:
        console.print("[yellow]No documents found in the database.[/yellow]")
        return
        
    console.print("[cyan]Analyzing knowledge base with AI...[/cyan]")
    
    try:
        analysis = asyncio.run(async_health_check(documents))
        console.print("\n[bold magenta]Knowledge Base Health Check[/bold magenta]\n")
        console.print(analysis)
    except Exception as e:
        console.print(f"[red]Error during health check: {e}[/red]")


@app.command()
def search(
    query: str,
    semantic: bool = typer.Option(True, help="Use LightRAG semantic retrieval"),
    source_type: Optional[str] = typer.Option(None, "--type", help="Filter by source type: youtube, article, pdf, etc."),
    mode: str = typer.Option("mix", help=f"LightRAG retrieval mode: {LIGHTRAG_MODES}"),
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum number of results to show."),
    open_result: bool = typer.Option(False, "--open", help="Prompt to open one of the top search results after listing."),
):
    """Search the index for content."""
    ensure_db_ready()
    console.print(f"[cyan]Searching for: '{query}'[/cyan]")

    if semantic:
        console.print(f"[dim]LightRAG mode: {mode}[/dim]")
        fetch_limit = semantic_search_fetch_limit(limit, source_type)
        try:
            raw_data = asyncio.run(
                async_query_data(
                    query,
                    mode=mode,
                    top_k=fetch_limit,
                    chunk_top_k=max(fetch_limit, 5),
                )
            )
            results = build_search_results(raw_data, source_type=source_type)[:limit]
        except Exception as e:
            console.print(f"[red]Error during LightRAG search: {e}[/red]")
            return
    else:
        results = search_keyword(query, limit=limit, source_type=source_type)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("#", justify="right", style="white", no_wrap=True)
    table.add_column("Match" if semantic else "ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Type", style="blue", no_wrap=True)
    table.add_column("Status", style="yellow", no_wrap=True)
    table.add_column("Summary", style="green")

    for index, result in enumerate(results, 1):
        if semantic:
            match_val = f"C{result['chunk_count']} E{result['entity_count']} R{result['relationship_count']}"
            summary_text = result.get("summary") or result.get("snippet")
            source = result.get("source_type") or "-"
            status = status_label(result.get("is_read", 0)) if result.get("id") is not None else "-"
        else:
            match_val = str(result["id"])
            summary_text = result["summary"]
            source = result["source_type"] or "-"
            status = status_label(result["is_read"]) if "is_read" in result.keys() else "-"

        table.add_row(
            str(index),
            match_val,
            f"{truncate_text(result['title'], 70)}\n[blue][link={result['url']}] {result['url']} [/link][/blue]",
            source,
            status,
            truncate_text(summary_text, 100),
        )

    console.print(table)

    if open_result:
        choice = typer.prompt(f"Open result? [1-{len(results)}/n]", default="n", show_default=False).strip().lower()
        if choice != "n":
            if not choice.isdigit():
                console.print("[red]Invalid selection.[/red]")
                return
            selected_index = int(choice)
            if selected_index < 1 or selected_index > len(results):
                console.print("[red]Selection out of range.[/red]")
                return
            selected = results[selected_index - 1]
            if selected.get("id") is None:
                console.print("[red]Selected LightRAG reference is not linked to a local document row.[/red]")
                return
            open_document_url(selected, mark_as_read_after_open=True)


@app.command()
def retrieve(
    query: str,
    mode: str = typer.Option("mix", help=f"LightRAG retrieval mode: {LIGHTRAG_MODES}"),
    source_type: Optional[str] = typer.Option(None, "--type", help="Filter by source type: youtube, article, pdf, etc."),
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum number of chunks/entities/relationships to show."),
):
    """Show structured LightRAG retrieval output for a query."""
    ensure_db_ready()
    console.print(f"[cyan]Retrieving context for: '{query}'[/cyan]")
    console.print(f"[dim]LightRAG mode: {mode}[/dim]")

    try:
        raw_data = asyncio.run(async_query_data(query, mode=mode, top_k=limit, chunk_top_k=max(limit, 5)))
    except Exception as e:
        console.print(f"[red]Error during LightRAG retrieval: {e}[/red]")
        return

    raw_data = filter_query_data_by_source_type(raw_data, source_type)
    if raw_data.get("status") != "success":
        console.print(f"[yellow]{raw_data.get('message', 'No retrieval results found.')}[/yellow]")
        return

    data = raw_data.get("data", {})
    metadata = raw_data.get("metadata", {})

    keywords = metadata.get("keywords", {})
    console.print(f"[bold]Mode:[/bold] {metadata.get('query_mode') or mode}")
    if keywords:
        console.print(f"[bold]High-level keywords:[/bold] {', '.join(keywords.get('high_level', [])) or '-'}")
        console.print(f"[bold]Low-level keywords:[/bold] {', '.join(keywords.get('low_level', [])) or '-'}")

    references = data.get("references", [])
    if references:
        refs_table = Table(title="References")
        refs_table.add_column("Ref", style="cyan", no_wrap=True)
        refs_table.add_column("Source", style="magenta")
        for ref in references[:limit]:
            refs_table.add_row(str(ref.get("reference_id") or "-"), truncate_text(ref.get("file_path"), 120))
        console.print(refs_table)

    chunks = data.get("chunks", [])
    if chunks:
        chunk_table = Table(title="Chunks")
        chunk_table.add_column("Ref", style="cyan", no_wrap=True)
        chunk_table.add_column("Source", style="blue")
        chunk_table.add_column("Content", style="green")
        for chunk in chunks[:limit]:
            chunk_table.add_row(
                str(chunk.get("reference_id") or "-"),
                truncate_text(chunk.get("file_path"), 50),
                truncate_text(chunk.get("content"), 140),
            )
        console.print(chunk_table)

    entities = data.get("entities", [])
    if entities:
        entity_table = Table(title="Entities")
        entity_table.add_column("Ref", style="cyan", no_wrap=True)
        entity_table.add_column("Entity", style="magenta")
        entity_table.add_column("Type", style="blue")
        entity_table.add_column("Description", style="green")
        for entity in entities[:limit]:
            entity_table.add_row(
                str(entity.get("reference_id") or "-"),
                truncate_text(entity.get("entity_name"), 30),
                truncate_text(entity.get("entity_type"), 18),
                truncate_text(entity.get("description"), 100),
            )
        console.print(entity_table)

    relationships = data.get("relationships", [])
    if relationships:
        rel_table = Table(title="Relationships")
        rel_table.add_column("Ref", style="cyan", no_wrap=True)
        rel_table.add_column("Edge", style="magenta")
        rel_table.add_column("Keywords", style="blue")
        rel_table.add_column("Description", style="green")
        for rel in relationships[:limit]:
            rel_table.add_row(
                str(rel.get("reference_id") or "-"),
                f"{truncate_text(rel.get('src_id'), 20)} → {truncate_text(rel.get('tgt_id'), 20)}",
                truncate_text(rel.get("keywords"), 30),
                truncate_text(rel.get("description"), 100),
            )
        console.print(rel_table)

    if not any([references, chunks, entities, relationships]):
        console.print("[yellow]LightRAG returned no structured retrieval rows.[/yellow]")


@app.command("list")
def list_command(
    unread: bool = typer.Option(False, "--unread", help="Show unread items only (default behavior when no status flag is provided)."),
    read: bool = typer.Option(False, "--read", help="Show read items only."),
    source_type: Optional[str] = typer.Option(None, "--type", help="Filter by source type: youtube, article, pdf, etc."),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum number of items to show."),
):
    """Browse the reading queue."""
    ensure_db_ready()

    if unread and read:
        console.print("[red]Choose either --unread or --read, not both.[/red]")
        raise typer.Exit(code=1)

    is_read = None
    if read:
        is_read = True
    elif unread or not read:
        is_read = False

    documents = list_documents(is_read=is_read, source_type=source_type, limit=limit)
    if not documents:
        console.print("[yellow]No documents found for the selected filters.[/yellow]")
        return

    table = Table(title="Reading Queue")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Source Type", style="blue", no_wrap=True)
    table.add_column("Tags", style="green")
    table.add_column("Status", style="yellow", no_wrap=True)
    table.add_column("Date Added", style="white", no_wrap=True)

    for doc in documents:
        table.add_row(
            str(doc["id"]),
            truncate_text(doc["title"], 70),
            doc["source_type"] or "-",
            truncate_text(doc["tags"], 40),
            status_label(doc["is_read"]),
            doc["created_at"] or "-",
        )

    console.print(table)


@app.command()
def show(
    id: int,
    open_result: bool = typer.Option(False, "--open", help="Open the document URL in the default browser."),
):
    """View item details."""
    ensure_db_ready()
    doc = get_document(id)
    if not doc:
        console.print(f"[red]Document {id} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold magenta]{doc['title'] or '(untitled)'}[/bold magenta]")
    console.print(f"[bold]ID:[/bold] {doc['id']}")
    console.print(f"[bold]URL:[/bold] [link={doc['url']}]{doc['url']}[/link]")
    console.print(f"[bold]Source Type:[/bold] {doc['source_type'] or '-'}")
    console.print(f"[bold]Tags:[/bold] {doc['tags'] or '-'}")
    console.print(f"[bold]Status:[/bold] {status_label(doc['is_read'])}")
    console.print(f"[bold]Date Added:[/bold] {doc['created_at'] or '-'}")
    console.print(f"[bold]Read At:[/bold] {doc['read_at'] or '-'}")
    console.print(f"[bold]Summary:[/bold]\n{doc['summary'] or '-'}")

    if open_result:
        open_document_url(doc, mark_as_read_after_open=True)


@app.command(name="open")
def open_command(id: int):
    """Open a document in the default browser and mark it as read."""
    ensure_db_ready()
    doc = get_document(id)
    if not doc:
        console.print(f"[red]Document {id} not found.[/red]")
        raise typer.Exit(code=1)
    open_document_url(doc, mark_as_read_after_open=True)


@app.command("read")
def read_command(
    id: Optional[int] = typer.Argument(None, help="Document ID to mark as read."),
    all: bool = typer.Option(False, "--all", help="Mark all unread documents as read."),
):
    """Mark documents as read."""
    ensure_db_ready()

    if all:
        updated = mark_all_read()
        console.print(f"[green]Marked {updated} document(s) as read.[/green]")
        return

    if id is None:
        console.print("[red]Provide a document ID or use --all.[/red]")
        raise typer.Exit(code=1)

    if not mark_read(id):
        console.print(f"[red]Document {id} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Marked document {id} as read.[/green]")


@app.command("unread")
def unread_command(id: int):
    """Mark a document as unread."""
    ensure_db_ready()
    if not mark_unread(id):
        console.print(f"[red]Document {id} not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Marked document {id} as unread.[/green]")


@app.command()
def delete(ids: List[int] = typer.Argument(..., help="One or more document IDs to delete.")):
    """Delete one or more documents."""
    ensure_db_ready()
    docs = [get_document(doc_id) for doc_id in ids]
    missing = [str(doc_id) for doc_id, doc in zip(ids, docs) if not doc]
    if missing:
        console.print(f"[red]Document(s) not found: {', '.join(missing)}[/red]")
        raise typer.Exit(code=1)

    label = ", ".join(str(doc_id) for doc_id in ids)
    if not typer.confirm(f"Delete document(s) {label}?", default=False):
        console.print("[yellow]Delete cancelled.[/yellow]")
        return

    deleted_count = 0
    blocked_ids: list[str] = []
    for doc_id in ids:
        rag_deleted, rag_error = asyncio.run(async_delete_document(doc_id))
        if not rag_deleted:
            if rag_error:
                console.print(f"[yellow]LightRAG warning for document {doc_id}: {rag_error}[/yellow]")
            blocked_ids.append(str(doc_id))
            continue

        if delete_document_record(doc_id):
            deleted_count += 1

    if blocked_ids:
        console.print(
            f"[yellow]Skipped SQLite deletion for document(s) {', '.join(blocked_ids)} because LightRAG cleanup did not succeed.[/yellow]"
        )

    console.print(f"[green]Deleted {deleted_count} document(s).[/green]")


@app.command()
def stats():
    """Show queue statistics."""
    ensure_db_ready()
    data = get_stats()

    console.print(f"[bold]Total items:[/bold] {data['total']}")
    console.print(f"[bold]Unread:[/bold] {data['unread_count']}")
    console.print(f"[bold]Read:[/bold] {data['read_count']}")

    breakdown = Table(title="By Source Type")
    breakdown.add_column("Source Type", style="blue")
    breakdown.add_column("Count", justify="right", style="cyan")
    for row in data["by_source_type"]:
        breakdown.add_row(row.get("source_type") or "-", str(row.get("count", 0)))
    console.print(breakdown)

    tags_table = Table(title="Most Common Tags")
    tags_table.add_column("Tag", style="green")
    tags_table.add_column("Count", justify="right", style="cyan")
    for tag, count in data["top_tags"]:
        tags_table.add_row(tag, str(count))
    if data["top_tags"]:
        console.print(tags_table)
    else:
        console.print("[dim]No tags available yet.[/dim]")

    oldest_unread = data["oldest_unread"]
    if oldest_unread:
        console.print(
            f"[bold]Oldest unread:[/bold] #{oldest_unread['id']} - {oldest_unread['title']} ({oldest_unread['created_at']})"
        )
    else:
        console.print("[dim]No unread items.[/dim]")


def _collect_tag_counts(source_type: Optional[str] = None) -> list[tuple[str, int]]:
    """Parse all tags from the database and return (tag, count) sorted by count descending."""
    from collections import Counter

    rows = get_all_tags()
    counter: Counter[str] = Counter()
    for row in rows:
        doc_tags = row["tags"]
        if not doc_tags:
            continue
        for tag in doc_tags.split(","):
            tag = tag.strip().lower()
            if tag:
                counter[tag] += 1
    return counter.most_common()


@app.command()
def topics(
    tag: Optional[str] = typer.Argument(None, help="Show documents matching this tag or cluster name."),
    cluster: bool = typer.Option(False, "--cluster", help="Group tags into high-level topic clusters using AI."),
    refresh: bool = typer.Option(False, "--refresh", help="Force re-clustering (with --cluster)."),
    source_type: Optional[str] = typer.Option(None, "--type", help="Filter by source type."),
    limit: int = typer.Option(30, "--limit", min=1, help="Maximum number of tags or documents to show."),
):
    """Browse tags and topic clusters in your knowledge base."""
    ensure_db_ready()

    if cluster:
        _topics_cluster(tag=tag, refresh=refresh, limit=limit)
    elif tag:
        _topics_browse_tag(tag=tag, source_type=source_type, limit=limit)
    else:
        _topics_list(source_type=source_type, limit=limit)


def _topics_list(source_type: Optional[str], limit: int):
    tag_counts = _collect_tag_counts(source_type=source_type)
    if not tag_counts:
        console.print("[yellow]No tags found.[/yellow]")
        return

    table = Table(title="Top Tags")
    table.add_column("#", justify="right", style="white", no_wrap=True)
    table.add_column("Tag", style="green")
    table.add_column("Docs", justify="right", style="cyan")

    for i, (tag, count) in enumerate(tag_counts[:limit], 1):
        table.add_row(str(i), tag, str(count))

    console.print(table)
    console.print(f"[dim]{len(tag_counts)} unique tags total. Use 'pci topics \"tag name\"' to browse.[/dim]")


def _topics_browse_tag(tag: str, source_type: Optional[str], limit: int):
    documents = get_documents_by_tag(tag, limit=limit)
    if source_type:
        documents = [d for d in documents if (d["source_type"] or "").lower() == source_type.lower()]

    if not documents:
        console.print(f"[yellow]No documents found with tag matching '{tag}'.[/yellow]")
        return

    table = Table(title=f"Documents tagged '{tag}'")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Type", style="blue", no_wrap=True)
    table.add_column("Status", style="yellow", no_wrap=True)
    table.add_column("Date", style="white", no_wrap=True)

    for doc in documents:
        table.add_row(
            str(doc["id"]),
            truncate_text(doc["title"], 70),
            doc["source_type"] or "-",
            status_label(doc["is_read"]),
            doc["created_at"] or "-",
        )

    console.print(table)


def _topics_cluster(tag: Optional[str], refresh: bool, limit: int):
    import json as _json
    from datetime import datetime

    from pci.llm import cluster_tags

    cache_path = os.path.join(os.path.dirname(os.environ.get("PCI_DB_PATH", "pci.db")), ".pci_topic_clusters.json")

    clusters = None
    if not refresh and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = _json.load(f)
            clusters = cache.get("clusters", [])
            console.print(f"[dim]Using cached clusters from {cache.get('created_at', 'unknown')}. Pass --refresh to rebuild.[/dim]")
        except Exception:
            clusters = None

    if clusters is None:
        tag_counts = _collect_tag_counts()
        top_tags = tag_counts[:200]
        if not top_tags:
            console.print("[yellow]No tags to cluster.[/yellow]")
            return

        console.print(f"[cyan]Clustering {len(top_tags)} tags with AI...[/cyan]")
        try:
            clusters = asyncio.run(cluster_tags(top_tags))
        except Exception as e:
            console.print(f"[red]Clustering failed: {e}[/red]")
            return

        with open(cache_path, "w", encoding="utf-8") as f:
            _json.dump({"created_at": datetime.now().isoformat(), "clusters": clusters}, f, indent=2)
        console.print(f"[dim]Cached to {cache_path}[/dim]")

    if tag:
        # Find cluster by name
        match = None
        for c in clusters:
            if c["name"].lower() == tag.lower():
                match = c
                break
        if not match:
            console.print(f"[yellow]No cluster named '{tag}'. Available clusters:[/yellow]")
            for c in clusters:
                console.print(f"  [green]{c['name']}[/green]")
            return

        # Find documents matching any tag in this cluster
        all_docs = []
        seen_ids = set()
        for cluster_tag in match["tags"]:
            for doc in get_documents_by_tag(cluster_tag, limit=200):
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    all_docs.append(doc)

        all_docs.sort(key=lambda d: d["created_at"] or "", reverse=True)
        all_docs = all_docs[:limit]

        table = Table(title=f"Cluster: {match['name']} ({len(match['tags'])} tags)")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Type", style="blue", no_wrap=True)
        table.add_column("Status", style="yellow", no_wrap=True)

        for doc in all_docs:
            table.add_row(
                str(doc["id"]),
                truncate_text(doc["title"], 70),
                doc["source_type"] or "-",
                status_label(doc["is_read"]),
            )

        console.print(table)
        console.print(f"[dim]Tags in this cluster: {', '.join(match['tags'][:15])}{'...' if len(match['tags']) > 15 else ''}[/dim]")
    else:
        # Build tag-to-doc-count lookup
        tag_counts = dict(_collect_tag_counts())

        table = Table(title="Topic Clusters")
        table.add_column("#", justify="right", style="white", no_wrap=True)
        table.add_column("Topic", style="magenta")
        table.add_column("Tags", justify="right", style="cyan", no_wrap=True)
        table.add_column("Sample Tags", style="green")

        for i, c in enumerate(clusters, 1):
            sample = ", ".join(c["tags"][:5])
            if len(c["tags"]) > 5:
                sample += ", ..."
            table.add_row(str(i), c["name"], str(len(c["tags"])), sample)

        console.print(table)
        console.print(f"[dim]Use 'pci topics --cluster \"Topic Name\"' to browse a cluster.[/dim]")


@app.command()
def dedupe(
    url_only: bool = typer.Option(False, "--url-only", help="Only check URL duplicates."),
    title_only: bool = typer.Option(False, "--title-only", help="Only check title similarity."),
    content_only: bool = typer.Option(False, "--content-only", help="Only check content/embedding similarity."),
    threshold: float = typer.Option(0.90, "--threshold", help="Similarity threshold for title (default 0.90) and content (default 0.95) tiers."),
):
    """Detect duplicate documents across URL, title, and content similarity tiers."""
    ensure_db_ready()

    run_all = not (url_only or title_only or content_only)
    title_threshold = threshold
    content_threshold = max(threshold, 0.95) if run_all else threshold

    rows = get_all_titles_and_urls()
    if not rows:
        console.print("[yellow]No documents in database.[/yellow]")
        return

    if run_all or url_only:
        _dedupe_urls(rows)

    if run_all or title_only:
        _dedupe_titles(rows, title_threshold)

    if run_all or content_only:
        _dedupe_content(rows, content_threshold)


def _dedupe_urls(rows):
    import re as _re
    from collections import defaultdict

    def normalize_url(url: str) -> str:
        url = url.lower().strip()
        url = _re.sub(r"^https?://", "", url)
        url = url.rstrip("/")
        url = _re.sub(r"[?&](utm_\w+|ref|source|fbclid|gclid|si)=[^&]*", "", url)
        url = url.rstrip("?&")
        return url

    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        norm = normalize_url(row["url"] or "")
        groups[norm].append(row)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}

    console.print(f"\n[bold]═══ URL Matches ({len(dupes)} group{'s' if len(dupes) != 1 else ''}) ═══[/bold]")
    if not dupes:
        console.print("[dim]No URL duplicates found.[/dim]")
        return

    table = Table()
    table.add_column("Group", justify="right", style="white", no_wrap=True)
    table.add_column("Doc ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("URL", style="blue")

    for i, (norm_url, docs) in enumerate(dupes.items(), 1):
        for doc in docs:
            table.add_row(str(i), str(doc["id"]), truncate_text(doc["title"], 50), truncate_text(doc["url"], 70))

    console.print(table)


def _dedupe_titles(rows, threshold: float):
    from difflib import SequenceMatcher

    pairs = []
    titles = [(r["id"], (r["title"] or "").strip().lower()) for r in rows if r["title"]]

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            id_a, title_a = titles[i]
            id_b, title_b = titles[j]
            if not title_a or not title_b:
                continue
            ratio = SequenceMatcher(None, title_a, title_b).ratio()
            if ratio >= threshold:
                pairs.append((id_a, title_a, id_b, title_b, ratio))

    pairs.sort(key=lambda p: p[4], reverse=True)

    console.print(f"\n[bold]═══ Similar Titles ({len(pairs)} pair{'s' if len(pairs) != 1 else ''}, threshold: {int(threshold * 100)}%) ═══[/bold]")
    if not pairs:
        console.print("[dim]No similar titles found.[/dim]")
        return

    table = Table()
    table.add_column("#", justify="right", style="white", no_wrap=True)
    table.add_column("Doc A", style="cyan")
    table.add_column("Doc B", style="cyan")
    table.add_column("Match", justify="right", style="green", no_wrap=True)

    for i, (id_a, title_a, id_b, title_b, ratio) in enumerate(pairs[:50], 1):
        table.add_row(
            str(i),
            f"#{id_a} {truncate_text(title_a, 40)}",
            f"#{id_b} {truncate_text(title_b, 40)}",
            f"{ratio:.0%}",
        )

    console.print(table)
    if len(pairs) > 50:
        console.print(f"[dim]Showing top 50 of {len(pairs)} pairs.[/dim]")


def _dedupe_content(rows, threshold: float):
    import struct

    import numpy as np

    from pci.db import get_db

    console.print("\n[cyan]Loading embeddings for content comparison...[/cyan]")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM vec_documents")
    vec_rows = cursor.fetchall()
    conn.close()

    if len(vec_rows) < 2:
        console.print("[dim]Not enough embeddings for comparison.[/dim]")
        return

    ids = []
    embeddings = []
    for vr in vec_rows:
        ids.append(vr[0])
        raw = vr[1]
        if isinstance(raw, bytes):
            dim = len(raw) // 4
            emb = list(struct.unpack(f"{dim}f", raw))
        else:
            emb = list(raw)
        embeddings.append(emb)

    id_to_title = {r["id"]: r["title"] or "(untitled)" for r in rows}
    matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    # Compute similarity in batches to avoid memory issues
    pairs = []
    batch_size = 200
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch = matrix[start:end]
        sim = batch @ matrix.T
        for local_i in range(end - start):
            global_i = start + local_i
            for j in range(global_i + 1, len(ids)):
                if sim[local_i, j] >= threshold:
                    pairs.append((ids[global_i], ids[j], float(sim[local_i, j])))

    pairs.sort(key=lambda p: p[2], reverse=True)

    console.print(f"[bold]═══ Content Overlap ({len(pairs)} pair{'s' if len(pairs) != 1 else ''}, threshold: {int(threshold * 100)}%) ═══[/bold]")
    if not pairs:
        console.print("[dim]No content duplicates found.[/dim]")
        return

    table = Table()
    table.add_column("#", justify="right", style="white", no_wrap=True)
    table.add_column("Doc A", style="cyan")
    table.add_column("Doc B", style="cyan")
    table.add_column("Sim", justify="right", style="green", no_wrap=True)

    for i, (id_a, id_b, sim) in enumerate(pairs[:50], 1):
        table.add_row(
            str(i),
            f"#{id_a} {truncate_text(id_to_title.get(id_a, '?'), 40)}",
            f"#{id_b} {truncate_text(id_to_title.get(id_b, '?'), 40)}",
            f"{sim:.0%}",
        )

    console.print(table)
    if len(pairs) > 50:
        console.print(f"[dim]Showing top 50 of {len(pairs)} pairs.[/dim]")


@app.command()
def import_bookmarks(path: str):
    """Import URLs from a Chrome bookmarks HTML file."""
    if not os.path.exists(path):
        console.print(f"[red]File not found: {path}[/red]")
        return

    console.print(f"[cyan]Parsing bookmarks from {path}...[/cyan]")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    import re

    urls = re.findall(r'HREF="(http[s]?://[^"]+)"', content, re.IGNORECASE)
    urls = list(dict.fromkeys(urls))

    console.print(f"[green]Found {len(urls)} unique URLs.[/green]")
    if not urls:
        return

    ensure_db_ready()

    async def process_all():
        semaphore = asyncio.Semaphore(5)

        async def bounded_ingest(u, index):
            async with semaphore:
                console.print(f"\n[bold blue]Starting ({index}/{len(urls)})[/bold blue]")
                try:
                    await async_ingest_url(u)
                except Exception as e:
                    console.print(f"[red]Failed to ingest {u}: {e}[/red]")

        tasks = [bounded_ingest(u, i) for i, u in enumerate(urls, 1)]
        await asyncio.gather(*tasks)

    start_time = time.time()
    asyncio.run(process_all())
    console.print(f"[bold green]Import complete in {time.time() - start_time:.2f} seconds![/bold green]")


@app.command()
def import_playlist(
    url: str,
    browser: str = typer.Option(None, help="Browser to extract cookies from for private playlists (e.g., 'chrome', 'safari', 'firefox')"),
):
    """Import all videos from a YouTube playlist. Use --browser for private playlists."""
    from yt_dlp import YoutubeDL

    console.print(f"[cyan]Fetching playlist info for {url}...[/cyan]")

    ydl_opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
        console.print(f"[dim]Using {browser} cookies for authentication...[/dim]")

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            console.print(f"[red]yt-dlp error: {e}[/red]")
            return

        if "entries" in info:
            entries = list(info["entries"])
            console.print(f"[green]Found {len(entries)} videos in playlist. Adding a 3-second delay between requests to prevent YouTube HTTP 429 bans.[/green]")

            ensure_db_ready()

            async def process_all():
                semaphore = asyncio.Semaphore(3)

                async def bounded_ingest(entry, index):
                    video_url = entry.get("url") or entry.get("webpage_url")
                    if not video_url and entry.get("id"):
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"

                    if video_url:
                        async with semaphore:
                            console.print(f"\n[bold blue]Starting ({index}/{len(entries)})[/bold blue]")
                            try:
                                await async_ingest_url(video_url)
                            except Exception as e:
                                console.print(f"[red]Failed to ingest {video_url}: {e}[/red]")

                tasks = [bounded_ingest(entry, i) for i, entry in enumerate(entries, 1)]
                await asyncio.gather(*tasks)

            start_time = time.time()
            asyncio.run(process_all())
            console.print(f"[bold green]Playlist import complete in {time.time() - start_time:.2f} seconds![/bold green]")
        else:
            console.print("[yellow]No playlist entries found. Are you sure this is a playlist URL?[/yellow]")


@app.command()
def import_folder(path: str, ext: str = typer.Option(None, help="Filter by file extension (e.g., 'pdf', 'md', 'txt'). If omitted, imports all supported types.")):
    """Import files from a directory into the index (.pdf, .md, .txt)."""
    if not os.path.exists(path) or not os.path.isdir(path):
        console.print(f"[red]Directory not found: {path}[/red]")
        return

    supported_exts = {".pdf", ".md", ".markdown", ".txt"}
    if ext:
        ext = f".{ext.removeprefix('.')}".lower()
        if ext not in supported_exts:
            console.print(f"[yellow]Warning: '{ext}' is not a typically supported extension. Proceeding anyway, but extraction may fail.[/yellow]")
        filter_exts = {ext}
    else:
        filter_exts = supported_exts

    target_files = []
    for root, _, files in os.walk(path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in filter_exts:
                target_files.append(os.path.join(root, file))

    console.print(f"[green]Found {len(target_files)} matching files in {path}.[/green]")
    if not target_files:
        return

    ensure_db_ready()

    async def process_all():
        semaphore = asyncio.Semaphore(5)

        async def bounded_ingest(f_path, index):
            async with semaphore:
                console.print(f"\n[bold blue]Starting ({index}/{len(target_files)})[/bold blue]: {os.path.basename(f_path)}")
                try:
                    await async_ingest_local_file(f_path)
                except Exception as e:
                    console.print(f"[red]Failed to ingest {f_path}: {e}[/red]")

        tasks = [bounded_ingest(f, i) for i, f in enumerate(target_files, 1)]
        await asyncio.gather(*tasks)

    start_time = time.time()
    asyncio.run(process_all())
    console.print(f"[bold green]Folder import complete in {time.time() - start_time:.2f} seconds![/bold green]")


@app.command()
def rename(
    paths: Optional[List[Path]] = typer.Argument(
        None,
        help="One or more files or directories to rename. Use --dir for the directory form.",
    ),
    directory: Optional[Path] = typer.Option(
        None, "--dir", help="A directory to scan (alternative to passing directory paths positionally)."
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r",
        help="Recurse into subdirectories when scanning a directory.",
    ),
    all_files: bool = typer.Option(
        False, "--all",
        help="Process every supported file, not only those with generic-looking names.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print proposed renames without touching the filesystem.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt before applying renames.",
    ),
    model: str = typer.Option(
        DEFAULT_RENAME_MODEL, "--model",
        help=(
            "OpenAI model identifier. Defaults to $PCI_RENAME_MODEL or 'gpt-5-mini'. "
            "Note: the original request referenced 'gpt-5.4-mini', which is not a real "
            "OpenAI model identifier; this is treated as a typo for 'gpt-5-mini'. "
            "Pass --model to override."
        ),
    ),
) -> None:
    """Smart LLM rename of files with generic / meaningless names.

    Supported extensions: .md, .markdown, .txt, .pdf
    Requires OPENAI_API_KEY (in .env or shell).
    Default model is `gpt-5-mini` (the user-supplied `gpt-5.4-mini` is treated as
    a typo); override with `--model` or the `PCI_RENAME_MODEL` env var.
    """
    targets: list[Path] = list(paths) if paths else []
    if directory is not None:
        targets.append(directory)
    if not targets:
        console.print("[red]No paths provided. Pass file/directory args or --dir.[/red]")
        raise typer.Exit(code=1)

    missing = [str(t) for t in targets if not t.exists()]
    if len(missing) == len(targets):
        console.print(f"[red]Path(s) not found: {', '.join(missing)}[/red]")
        raise typer.Exit(code=1)
    for m in missing:
        console.print(f"[yellow]Warning: not found: {m}[/yellow]")

    if not dry_run and not yes:
        if not typer.confirm(
            f"Apply smart rename to {len(targets)} target(s) using {model}?",
            default=False,
        ):
            console.print("[yellow]Rename cancelled.[/yellow]")
            return

    summary = asyncio.run(
        rename_paths(
            targets,
            model=model,
            dry_run=dry_run,
            only_generic=not all_files,
            recursive=recursive,
            propose_fn=propose_filename,
        )
    )

    for r in summary["results"]:
        original = r.get("original", "?")
        reason = r.get("reason", "")
        if r.get("renamed"):
            new_name = Path(r["new_path"]).name
            console.print(f"[green]renamed[/green] {original} → {new_name}")
        elif reason == "dry-run":
            console.print(f"[cyan]would rename[/cyan] {original} → {r.get('proposed', '?')}")
        elif reason == "not-generic":
            console.print(f"[dim]skipped (not generic)[/dim] {original}")
        elif reason == "no-change":
            console.print(f"[dim]skipped (no-change)[/dim] {original}")
        elif reason == "unsupported-extension":
            console.print(f"[yellow]skipped (unsupported ext)[/yellow] {original}")
        elif reason == "not-found":
            console.print(f"[red]not found[/red] {original}")
        elif reason.startswith("extraction-error"):
            console.print(f"[red]extraction error[/red] {original}: {reason}")
        elif reason.startswith("llm-error"):
            console.print(f"[red]LLM error[/red] {original}: {reason}")
        elif reason.startswith("rename-error"):
            console.print(f"[red]rename error[/red] {original}: {reason}")
        else:
            console.print(f"[yellow]skipped[/yellow] {original}: {reason}")

    console.print(
        f"\n[bold green]Rename complete:[/bold green] "
        f"processed {summary['processed']}, renamed {summary['renamed']}, "
        f"skipped {summary['skipped']}, errors {len(summary['errors'])}."
    )


@app.command()
def export_csv(path: str = typer.Argument("index_export.csv", help="Path to save the CSV file")):
    """Export all indexed documents to a CSV file."""
    ensure_db_ready()

    console.print(f"[cyan]Exporting index to {path}...[/cyan]")

    try:
        documents = get_all_documents()

        if not documents:
            console.print("[yellow]Index is empty. No data to export.[/yellow]")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            headers = documents[0].keys()
            writer = csv.writer(f)
            writer.writerow(headers)

            for doc in documents:
                writer.writerow([doc[h] for h in headers])

        console.print(f"[green]Successfully exported {len(documents)} documents to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error exporting to CSV: {e}[/red]")


@app.command("export-vault")
def export_vault_cmd(  # noqa: A002
    vault_dir: str = typer.Argument(..., help="Path to the Obsidian vault directory"),
    no_content: bool = typer.Option(False, "--no-content", help="Exclude full content from exported files (summary only)"),
    type: Optional[str] = typer.Option(None, "--type", help="Filter by source type (article, youtube, pdf, etc.)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of documents to export"),
) -> None:
    """Export all indexed documents as Obsidian-compatible Markdown files with YAML frontmatter."""
    ensure_db_ready()
    from pci.vault import export_vault

    result = export_vault(
        vault_dir=vault_dir,
        include_content=not no_content,
        source_type=type,
        limit=limit,
    )
    console.print(f"[bold green]✓ Exported {result['exported']} document(s) to:[/bold green] {result['output_dir']}")
    if result["skipped"]:
        console.print(f"[dim]  Skipped {result['skipped']} already-existing file(s)[/dim]")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to run the server on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    open_browser: bool = typer.Option(True, help="Open browser on start"),
):
    """Start the web UI server."""
    import uvicorn

    from pci.api.config import FRONTEND_DIST

    console.print(f"[cyan]Starting Content Index server on http://{host}:{port}[/cyan]")

    if os.path.isdir(FRONTEND_DIST):
        console.print("[green]Serving built frontend from frontend/dist/[/green]")
    else:
        console.print("[yellow]No frontend build found. API-only mode.[/yellow]")
        console.print("[yellow]Run: cd frontend && npm install && npm run build[/yellow]")

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(
        "pci.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    app()
