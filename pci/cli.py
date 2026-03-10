import typer
from rich.console import Console
from rich.table import Table
import os
from dotenv import load_dotenv

load_dotenv()

from pci.db import init_db, search_similar, search_keyword, get_all_documents
from pci.ingest import ingest_url, ingest_pdf
from pci.embeddings import get_embedding

app = typer.Typer(help="Personal Content Index")
console = Console()

@app.command()
def init():
    """Initialize the local SQLite database."""
    init_db()
    console.print(f"[green]Initialized database at {os.environ.get('PCI_DB_PATH', 'pci.db')}[/green]")

@app.command()
def add(url: str):
    """Add a new URL (Article or YouTube video) to the index."""
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        console.print("[yellow]Database not found. Initializing...[/yellow]")
        init_db()
    
    ingest_url(url)

@app.command()
def search(query: str, semantic: bool = typer.Option(True, help="Use semantic vector search")):
    """Search the index for content."""
    console.print(f"[cyan]Searching for: '{query}'[/cyan]")
    
    if semantic:
        console.print("[dim]Generating embedding for query...[/dim]")
        q_emb = get_embedding(query)
        try:
            results = search_similar(q_emb, limit=5)
        except Exception as e:
            console.print(f"[red]Error during semantic search: {e}[/red]")
            return
    else:
        results = search_keyword(query, limit=5)
        
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
        
    table = Table(title="Search Results")
    table.add_column("Score" if semantic else "ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Summary", style="green")
    
    for r in results:
        # semantic has distance, keyword doesn't
        score_val = f"{r['distance']:.4f}" if semantic else str(r['id'])
        
        table.add_row(
            score_val,
            f"{r['title']}\n[blue][link={r['url']}]URL[/link][/blue]",
            r['summary']
        )
        
    console.print(table)

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
    # Match href attributes containing http/https URLs
    urls = re.findall(r'HREF="(http[s]?://[^"]+)"', content, re.IGNORECASE)
    # Deduplicate while preserving order
    urls = list(dict.fromkeys(urls))
    
    console.print(f"[green]Found {len(urls)} unique URLs.[/green]")
    if not urls:
        return
        
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        init_db()
        
    import time
    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold blue]({i}/{len(urls)})[/bold blue]")
        try:
            ingest_url(url)
        except Exception as e:
            console.print(f"[red]Failed to ingest {url}: {e}[/red]")
        if i < len(urls):
            time.sleep(3)

@app.command()
def import_playlist(
    url: str, 
    browser: str = typer.Option(None, help="Browser to extract cookies from for private playlists (e.g., 'chrome', 'safari', 'firefox')")
):
    """Import all videos from a YouTube playlist. Use --browser for private playlists."""
    from yt_dlp import YoutubeDL
    console.print(f"[cyan]Fetching playlist info for {url}...[/cyan]")
    
    ydl_opts = {'quiet': True, 'extract_flat': True, 'no_warnings': True}
    if browser:
        ydl_opts['cookiesfrombrowser'] = (browser,)
        console.print(f"[dim]Using {browser} cookies for authentication...[/dim]")
        
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            console.print(f"[red]yt-dlp error: {e}[/red]")
            return
            
        if 'entries' in info:
            import time
            entries = list(info['entries'])
            console.print(f"[green]Found {len(entries)} videos in playlist. Adding a 3-second delay between requests to prevent YouTube HTTP 429 bans.[/green]")
            
            if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
                init_db()
                
            for i, entry in enumerate(entries, 1):
                video_url = entry.get('url') or entry.get('webpage_url')
                if not video_url and entry.get('id'):
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    
                if video_url:
                    console.print(f"\n[bold blue]({i}/{len(entries)})[/bold blue]")
                    try:
                        ingest_url(video_url)
                    except Exception as e:
                        console.print(f"[red]Failed to ingest {video_url}: {e}[/red]")
                    if i < len(entries):
                        time.sleep(3)
        else:
            console.print("[yellow]No playlist entries found. Are you sure this is a playlist URL?[/yellow]")

@app.command()
def import_pdf(path: str):
    """Import a single PDF file into the index."""
    if not os.path.exists(path):
        console.print(f"[red]File not found: {path}[/red]")
        return
        
    if not path.lower().endswith(".pdf"):
        console.print(f"[yellow]Warning: {path} does not have a .pdf extension. Attempting to parse anyway.[/yellow]")
        
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        console.print("[yellow]Database not found. Initializing...[/yellow]")
        init_db()
        
    ingest_pdf(path)

@app.command()
def import_pdf_folder(path: str):
    """Import all PDF files from a directory into the index."""
    if not os.path.exists(path) or not os.path.isdir(path):
        console.print(f"[red]Directory not found: {path}[/red]")
        return
        
    pdf_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
                
    console.print(f"[green]Found {len(pdf_files)} PDF files in {path}.[/green]")
    if not pdf_files:
        return
        
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        init_db()
        
    import time
    for i, file_path in enumerate(pdf_files, 1):
        console.print(f"\n[bold blue]({i}/{len(pdf_files)})[/bold blue]")
        try:
            ingest_pdf(file_path)
        except Exception as e:
            console.print(f"[red]Failed to ingest {file_path}: {e}[/red]")
            
        # Optional sleep to avoid overly thrashing the system/API, similar to import_bookmarks
        if i < len(pdf_files):
            time.sleep(1)

@app.command()
def export_csv(path: str = typer.Argument("index_export.csv", help="Path to save the CSV file")):
    """Export all indexed documents to a CSV file."""
    import csv
    
    if not os.path.exists(os.environ.get("PCI_DB_PATH", "pci.db")):
        console.print("[red]Database not found. Nothing to export.[/red]")
        return
        
    console.print(f"[cyan]Exporting index to {path}...[/cyan]")
    
    try:
        documents = get_all_documents()
        
        if not documents:
            console.print("[yellow]Index is empty. No data to export.[/yellow]")
            return
            
        with open(path, "w", newline="", encoding="utf-8") as f:
            # Get headers from the first row's keys
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
