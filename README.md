# Content SPA

**Content SPA** (Searchable Personal Archive) is a CLI tool and local database for indexing, embedding, searching, and managing your personal content library. It acts as an intelligent "read it later" / "watch it later" engine that ingests web articles, YouTube videos/playlists, PDFs, and text files, storing them in a local vector database for AI-powered semantic search.

## Features

- **Multi-Format Ingestion**: Feed it single URLs, Chrome bookmark HTML exports, YouTube playlists, local PDFs, Markdown files, and text files.
- **Smart Extraction**: Uses `trafilatura` for clean article extraction and `yt-dlp` for YouTube transcripts.
- **AI Processing**: Summarizes content using Anthropic LLMs and generates local embeddings using `sentence-transformers`.
- **Semantic Search**: Fast, local vector searches using `sqlite-vec` to find what you're looking for based on meaning, not just exact keywords.
- **Read-Later Queue**: Track unread vs read items, open items in your browser, inspect details, and delete stale entries.
- **Local First**: All data is stored in a local SQLite database (`pci.db`), keeping your personal archive private.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/young-algo/content-spa.git
   cd content-spa
   ```

2. **Install dependencies:**
   The project uses `uv` for dependency management.
   ```bash
   uv sync
   ```

3. **Environment setup:**
   Create a `.env` file in the project root.
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   PCI_DB_PATH=pci.db
   ```

## CLI Usage

The tool is accessible via the `pci` command once installed, or via `uv run python -m pci.cli`.

### Core Commands

- **Initialize the database**
  ```bash
  pci init
  ```

- **Add a single item**
  Ingest a URL or local file.
  ```bash
  pci add <url_or_path>
  ```

- **Search**
  Semantic search is enabled by default. Use `--no-semantic` for keyword search.
  ```bash
  pci search "your query here"
  pci search "your query here" --no-semantic
  pci search "your query here" --type youtube
  pci search "your query here" --open
  ```

### Read-Later / Queue Commands

- **List queue items**
  Defaults to unread items, newest first.
  ```bash
  pci list
  pci list --limit 50
  pci list --read
  pci list --unread
  pci list --type article
  ```

- **Show item details**
  ```bash
  pci show <id>
  pci show <id> --open
  ```
  `--open` opens the URL in your default browser and marks the item as read.

- **Open an item directly**
  ```bash
  pci open <id>
  ```
  This opens the URL and marks the item as read.

- **Mark items read / unread**
  ```bash
  pci read <id>
  pci read --all
  pci unread <id>
  ```

- **Delete items**
  ```bash
  pci delete <id>
  pci delete 1 2 3
  ```

- **View stats**
  ```bash
  pci stats
  ```

### Bulk Import Commands

- **Import Chrome bookmarks**
  ```bash
  pci import-bookmarks path/to/bookmarks.html
  ```

- **Import a YouTube playlist**
  ```bash
  pci import-playlist <playlist_url>
  pci import-playlist <playlist_url> --browser chrome
  ```

- **Import a folder of local files**
  ```bash
  pci import-folder path/to/my-documents/
  pci import-folder path/to/my-obsidian-vault/ --ext md
  ```

- **Export to CSV**
  ```bash
  pci export-csv
  pci export-csv backup.csv
  ```

## Testing

Run the test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

The current tests cover:
- DB migration and new schema columns
- Content storage and truncation
- Read/unread helpers
- List/search/delete CLI flows
- `show --open` and `open` marking items as read

## Architecture

- **CLI Framework**: `typer`
- **Database**: `sqlite3` + `sqlite-vec` + `sqlean.py`
- **Embeddings**: `sentence-transformers` (local)
- **Summarization**: `anthropic` (Claude)
- **Extraction**: `trafilatura` (web), `yt-dlp` (YouTube)
