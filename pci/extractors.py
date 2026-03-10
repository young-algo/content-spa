import os
import re
import json
import subprocess
import tempfile
import urllib.request
from urllib.error import URLError
from urllib.parse import urlparse
import trafilatura
import webvtt
from pypdf import PdfReader

class ExtractionError(Exception):
    """Base exception for content extraction failures."""
    pass

class RateLimitError(ExtractionError):
    """Raised when extraction fails due to rate limiting (e.g. HTTP 429)."""
    pass

def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    return "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc

def extract_youtube(url: str) -> dict:
    video_id = None
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        video_id = match.group(1)

    if not video_id:
        raise ExtractionError(f"Could not parse valid YouTube video ID from URL: {url}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs", "en",
                "--sub-format", "vtt",
                "--cookies-from-browser", "chrome",
                "--js-runtimes", "node",
                "--remote-components", "ejs:npm",
                "--dump-json",
                "--no-simulate",
                "-o", f"{tmpdir}/%(id)s.%(ext)s",
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                if "HTTP Error 429: Too Many Requests" in result.stderr:
                    raise RateLimitError("YouTube API rate limit exceeded (HTTP 429). You are temporarily IP-banned from fetching closed captions.")
                else:
                    raise ExtractionError(f"yt-dlp failed: {result.stderr}")

            try:
                # The last line of stdout is the JSON dump
                yt_data = json.loads(result.stdout.strip().split('\n')[-1])
                title = yt_data.get('title', 'Unknown Title')
            except (json.JSONDecodeError, IndexError):
                title = "Unknown Title"

            vtt_file = None
            for file in os.listdir(tmpdir):
                if file.endswith(".vtt"):
                    vtt_file = os.path.join(tmpdir, file)
                    break

            if vtt_file:
                # Parse cleanly with webvtt-py
                transcript_text = ""
                try:
                    vtt = webvtt.read(vtt_file)
                    lines = []
                    last_line = None
                    for caption in vtt:
                        # Cleaning multiple lines within a single cue and stripping timestamps
                        text = caption.text.strip().replace('\n', ' ')
                        # Avoid appending exact duplicates from overlapping cues
                        if text and text != last_line:
                            lines.append(text)
                            last_line = text
                    transcript_text = " ".join(lines)
                except Exception as e:
                    raise ExtractionError(f"Failed to parse VTT subtitle file: {e}")
            else:
                transcript_text = ""

    except subprocess.TimeoutExpired:
        raise ExtractionError("Extraction timed out after 120 seconds while calling yt-dlp.")
    except RateLimitError:
        raise
    except Exception as e:
        if not isinstance(e, ExtractionError):
            raise ExtractionError(f"Unexpected error extracting YouTube video: {e}")
        raise

    return {"title": title, "content": transcript_text, "source_type": "youtube"}

def extract_article(url: str) -> dict:
    try:
        # Enforce a strict 30 second timeout on URL opening
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        raise ExtractionError(f"Failed to download article at {url}: {e}")

    if not html:
        raise ExtractionError(f"Retrieved empty HTML from {url}")

    result = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not result:
        raise ExtractionError("Trafilatura could not extract any meaningful text content from the article.")

    title = url
    try:
        from trafilatura.metadata import extract_metadata
        meta = extract_metadata(html)
        if meta and meta.title:
            title = meta.title
    except:
        pass

    return {"title": title, "content": result, "source_type": "article"}

def extract_pdf(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise ExtractionError(f"PDF file not found: {file_path}")
        
    try:
        reader = PdfReader(file_path)
        text_content = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
                
        content = "\n".join(text_content)
        
        if not content.strip():
            raise ExtractionError(f"Could not extract any text from PDF: {file_path}")
            
        title = os.path.basename(file_path)
        # Try to get title from metadata if available
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title
            
        return {"title": title, "content": content, "source_type": "pdf"}
    except Exception as e:
        if not isinstance(e, ExtractionError):
            raise ExtractionError(f"Failed to extract text from PDF {file_path}: {e}")
        raise
