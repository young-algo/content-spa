import asyncio
import csv
import os
import time
import webbrowser
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from pci.db import (
    delete_document,
    get_all_documents,
    get_document,
    get_stats,
    init_db,
    list_documents,
    mark_all_read,
    mark_read,
    mark_unread,
    migrate_db,
    search_keyword,
    search_similar,
)
from pci.embeddings import get_embedding
from pci.ingest import async_ingest_local_file, async_ingest_url

load_dotenv()

app = typer.Typer(help="Personal Content Index")
console = Console()


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
    """Add a new URL (Article or YouTube video) or a local file (.pdf, .md, .txt) to the index."""
    ensure_db_ready()

    if os.path.exists(source) and os.path.isfile(source):
        asyncio.run(async_ingest_local_file(source))
    else:
        asyncio.run(async_ingest_url(source))


@app.command()
def search(
    query: str,
    semantic: bool = typer.Option(True, help="Use semantic vector search"),
    source_type: Optional[str] = typer.Option(None, "--type", help="Filter by source type: youtube, article, pdf, etc."),
    open_result: bool = typer.Option(False, "--open", help="Prompt to open one of the top search results after listing."),
):
    """Search the index for content."""
    ensure_db_ready()
    console.print(f"[cyan]Searching for: '{query}'[/cyan]")

    if semantic:
        console.print("[dim]Generating embedding for query...[/dim]")
        q_emb = get_embedding(query)
        try:
            results = search_similar(q_emb, limit=5, source_type=source_type)
        except Exception as e:
            console.print(f"[red]Error during semantic search: {e}[/red]")
            return
    else:
        results = search_keyword(query, limit=5, source_type=source_type)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("#", justify="right", style="white", no_wrap=True)
    table.add_column("Score" if semantic else "ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Type", style="blue", no_wrap=True)
    table.add_column("Status", style="yellow", no_wrap=True)
    table.add_column("Summary", style="green")

    for index, r in enumerate(results, 1):
        score_val = f"{r['distance']:.4f}" if semantic else str(r['id'])
        table.add_row(
            str(index),
            score_val,
            f"{truncate_text(r['title'], 70)}\n[blue][link={r['url']}] {r['url']} [/link][/blue]",
            r["source_type"] or "-",
            status_label(r["is_read"]) if "is_read" in r.keys() else "-",
            truncate_text(r["summary"], 100),
        )

    console.print(table)

    if open_result:
        choice = typer.prompt("Open result? [1-5/n]", default="n", show_default=False).strip().lower()
        if choice != "n":
            if not choice.isdigit():
                console.print("[red]Invalid selection.[/red]")
                return
            selected_index = int(choice)
            if selected_index < 1 or selected_index > len(results):
                console.print("[red]Selection out of range.[/red]")
                return
            open_document_url(results[selected_index - 1], mark_as_read_after_open=True)


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


@app.command()
def open(id: int):
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
    for doc_id in ids:
        if delete_document(doc_id):
            deleted_count += 1

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


if __name__ == "__main__":
    app()
