# Content SPA 

**Content SPA** (Searchable Personal Archive) is a CLI tool and local database for indexing, embedding, and searching your personal content library. It acts as an intelligent "read it later" / "watch it later" engine that ingests web articles, YouTube videos/playlists, and PDFs, storing them in a local vector database for AI-powered semantic search.

## Features

- **Multi-Format Ingestion**: Feed it single URLs, Chrome bookmark HTML exports, YouTube playlists, or local PDFs.
- **Smart Extraction**: Uses `trafilatura` for clean article extraction and `yt-dlp` for YouTube transcripts.
- **AI Processing**: Summarizes content using Anthropic LLMs and generates local embeddings using `sentence-transformers`.
- **Semantic Search**: Fast, local vector searches using `sqlite-vec` to find exactly what you're looking for based on meaning, not just exact keyword matches.
- **Local First**: All data is stored in a local SQLite database (`pci.db`), keeping your personal archive private.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/young-algo/content-spa.git
   cd content-spa
   ```

2. **Install dependencies:**
    The project uses `uv` for dependency management. If you don't have it, install it, then run:
   ```bash
   uv sync
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory based on your needs. You will need an Anthropic API key for summarization.
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   PCI_DB_PATH=pci.db
   ```

## CLI Usage

The tool is accessible via the `pci` command once installed, or by running the module directly. 

### Core Commands

* **Initialize Database:**
  Creates the local SQLite vector database.
  ```bash
  pci init
  ```

* **Add a Single Item:**
  Ingests a URL (article or YouTube video) or a local PDF file, extracts the text/transcript, generates a summary, creates an embedding, and saves it.
  ```bash
  pci add <url_or_path>
  ```

* **Search:**
  Search your personal index. Uses semantic vector search by default. Pass `--no-semantic` for standard keyword search.
  ```bash
  pci search "your query here"
  ```

### Bulk Import Commands

* **Import Chrome Bookmarks:**
  Parses an exported Chrome bookmarks HTML file and ingests all links.
  ```bash
  pci import-bookmarks path/to/bookmarks.html
  ```

* **Import YouTube Playlist:**
  Downloads transcripts for all videos in a public (or private, with cookies) YouTube playlist.
  ```bash
  pci import-playlist <playlist_url>
  ```

* **Import PDFs:**
  Ingest a single PDF or an entire directory of PDFs.
  ```bash
  pci import-pdf path/to/document.pdf
  pci import-pdf-folder path/to/folder/
  ```

* **Export:**
  Export your entire index to a CSV file for backup or external analysis.
  ```bash
  pci export-csv [optional_output_path.csv]
  ```

## Architecture

- **CLI Framework**: `typer`
- **Database**: `sqlite3` + `sqlite-vec` (for vector storage) + `sqlean.py`
- **Embeddings**: `sentence-transformers` (runs locally, e.g., `all-MiniLM-L6-v2`)
- **Summarization**: `anthropic` (Claude)
- **Extraction**: `trafilatura` (Web), `yt-dlp` (YouTube)
