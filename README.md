# Content SPA

**Content SPA** (Searchable Personal Archive) is a CLI tool and local database for indexing, embedding, searching, and managing your personal content library. It acts as an intelligent "read it later" / "watch it later" engine that ingests web articles, YouTube videos/playlists, PDFs, and text files, storing them in a local vector database for AI-powered semantic search.

## Features

- **Multi-Format Ingestion**: Feed it single URLs, Chrome bookmark HTML exports, YouTube playlists, local PDFs, Markdown files, and text files.
- **Inbox Folder Sync**: Drop URLs into `inbox.txt` or files into an inbox directory (e.g. a Google Drive folder) and have `pci ingest` pull them in and archive processed files automatically.
- **Smart Extraction**: Uses `trafilatura` for clean article extraction and `yt-dlp` for YouTube transcripts.
- **AI Processing**: Summarizes content using Anthropic LLMs and indexes the full document with LightRAG using OpenRouter-hosted `qwen/qwen3-embedding-8b` embeddings.
- **Semantic Search & Retrieval**: Uses [LightRAG](https://github.com/HKUDS/LightRAG) for document indexing, graph-aware retrieval, and structured search results. Optional Qwen3-Reranker-8B reranking via SiliconFlow when `SILICON_FLOW_API_KEY` is set.
- **Synthesis & Health Checks**: Generate long-form markdown articles from retrieved knowledge (`pci synthesize`) and run LLM-driven knowledge-gap reports (`pci checkup`).
- **Topic Clustering & Deduplication**: Browse tags, cluster them into high-level topics with AI (`pci topics --cluster`), and detect duplicates across URL, title, and embedding-similarity tiers (`pci dedupe`).
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
   # Optional — inbox folder used by `pci ingest` (e.g. a local Google Drive path)
   PCI_INBOX_DIR=/path/to/your/inbox
   # Optional — enables Qwen3-Reranker-8B reranking on LightRAG retrieval
   SILICON_FLOW_API_KEY=your_siliconflow_key_here
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

- **Ingest an inbox folder**
  Process URLs listed in `<inbox>/inbox.txt` and any PDF/Markdown/text files in the inbox directory. Successful files are moved to `<inbox>/archived/`, and succeeded URLs are removed from `inbox.txt`. Ideal for a Google Drive / Dropbox folder you drop items into from your phone.
  ```bash
  pci ingest                        # uses $PCI_INBOX_DIR
  pci ingest --inbox /path/to/dir
  pci ingest --dry-run
  ```

- **Ask for an answer**
  Generate a paragraph-form answer with LightRAG. Optionally save the answer as Markdown and re-ingest it back into the index.
  ```bash
  pci ask "your question here"
  pci ask "your question here" --mode mix
  pci ask "your question here" --references
  pci ask "your question here" --save-to answer.md --ingest
  ```

- **Synthesize an article**
  Generate a comprehensive markdown article about a topic using retrieved context. Saves to `syntheses/` by default.
  ```bash
  pci synthesize "your topic here"
  pci synthesize "your topic here" --mode hybrid --limit 30
  pci synthesize "your topic here" --save-dir notes/ --ingest
  ```

- **Knowledge base health check**
  Ask the LLM to flag gaps and suggest follow-up research based on recent (or random) items.
  ```bash
  pci checkup
  pci checkup --limit 100
  pci checkup --random --limit 30
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

### Organization Commands

- **Browse topics and tags**
  List the most common tags, filter documents by a specific tag, or cluster tags into high-level AI-generated topics. Cluster results are cached locally (`.pci_topic_clusters.json`); pass `--refresh` to rebuild.
  ```bash
  pci topics                               # top 30 tags by document count
  pci topics --limit 50
  pci topics --type pdf                    # only tags from PDF documents
  pci topics "ai agents"                   # list documents with a tag
  pci topics --cluster                     # AI-grouped topic clusters
  pci topics --cluster "Macro & Markets"   # browse documents in a cluster
  pci topics --cluster --refresh
  ```

- **Detect duplicates**
  Layered duplicate detection across URL normalization, title fuzzy matching, and embedding cosine similarity.
  ```bash
  pci dedupe                       # run all three tiers
  pci dedupe --url-only
  pci dedupe --title-only --threshold 0.85
  pci dedupe --content-only --threshold 0.95
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
- **Reranking (optional)**: SiliconFlow `Qwen/Qwen3-Reranker-8B`, enabled when `SILICON_FLOW_API_KEY` is set
- **LLM**: `anthropic` (Claude) — Haiku for indexing/extraction and tag clustering, Sonnet for query-time retrieval reasoning
- **Extraction**: `trafilatura` (web), `yt-dlp` (YouTube)
