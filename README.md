# Content SPA

**Content SPA** (Searchable Personal Archive) is a CLI tool and local database for indexing, embedding, searching, and managing your personal content library. It acts as an intelligent "read it later" / "watch it later" engine that ingests web articles, YouTube videos/playlists, PDFs, and text files, storing them in a local vector database for AI-powered semantic search.

## Features

- **Multi-Format Ingestion**: Feed it single URLs, Chrome bookmark HTML exports, YouTube playlists, local PDFs, Markdown files, and text files.
- **Smart Extraction**: Uses `trafilatura` for clean article extraction and `yt-dlp` for YouTube transcripts.
- **AI Processing**: Summarizes content using Anthropic LLMs and indexes the full document with LightRAG using OpenRouter-hosted `qwen/qwen3-embedding-8b` embeddings.
- **Semantic Search & Retrieval**: Uses [LightRAG](https://github.com/HKUDS/LightRAG) for document indexing, graph-aware retrieval, and structured search results.
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
   OPENROUTER_API_KEY=your_openrouter_key_here
   PCI_DB_PATH=pci.db
   PCI_LIGHTRAG_DIR=.pci_lightrag
   PCI_EMBEDDING_MODEL=qwen/qwen3-embedding-8b
   PCI_LIGHTRAG_INDEX_MODEL=claude-haiku-4-5-20251001
   PCI_LIGHTRAG_QUERY_MODEL=claude-sonnet-4-6
   ```

## CLI Usage

The tool is accessible via the `pci` command once installed, or via `uv run python -m pci.cli`.

### Core Commands

- **Initialize the database**
  ```bash
  pci init
  ```

- **Inspect active model configuration**
  ```bash
  pci doctor
  ```

- **Rebuild the LightRAG index from SQLite**
  ```bash
  pci reindex
  pci reindex --no-reset --resume
  pci reindex --no-reset --no-resume
  ```
  Reindex progress is tracked in `.pci_lightrag/reindex_state.json` so interrupted runs can resume.

- **Add a single item**
  Ingest a URL or local file.
  ```bash
  pci add <url_or_path>
  ```

- **Ask for an answer**
  Generate a paragraph-form answer with LightRAG.
  ```bash
  pci ask "your question here"
  pci ask "your question here" --mode mix
  pci ask "your question here" --references
  ```

- **Search**
  LightRAG semantic search is enabled by default. Use `--no-semantic` for SQL keyword search.
  ```bash
  pci search "your query here"
  pci search "your query here" --mode naive
  pci search "your query here" --mode mix --type youtube
  pci search "your query here" --no-semantic
  pci search "your query here" --open
  ```

- **Retrieve structured context**
  Inspect the chunks, entities, relationships, and references LightRAG retrieved.
  ```bash
  pci retrieve "your query here"
  pci retrieve "your query here" --mode hybrid
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
- **Metadata Database**: `sqlite3` + `sqlite-vec` + `sqlean.py`
- **Indexing / Retrieval**: `lightrag-hku`
- **Embeddings**: OpenRouter `qwen/qwen3-embedding-8b` via the `openai` SDK
- **LLM**: `anthropic` (Claude) — Haiku for indexing/extraction, Sonnet for query-time retrieval reasoning
- **Extraction**: `trafilatura` (web), `yt-dlp` (YouTube)
