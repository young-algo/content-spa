# Topics Browsing & Duplicate Detection

## Overview

Two new CLI commands for the Personal Content Index: `pci topics` for browsing/clustering tags, and `pci dedupe` for layered duplicate detection. Both are read-only reporting tools.

## Feature 1: `pci topics`

### Raw Tag Mode (default)

`pci topics` displays the most common tags across all documents, ranked by document count.

```
pci topics                     # top 30 tags by count
pci topics --limit 50          # top 50
pci topics --type pdf          # only tags from PDF documents
pci topics "ai agents"         # list documents with this tag
```

**Implementation:**
- Query all non-null `tags` from the `documents` table
- Split comma-separated tags, normalize (lowercase, strip whitespace)
- Count occurrences, display as a Rich table sorted by count descending
- When a tag argument is provided, query documents whose `tags` column contains that tag (case-insensitive) and display in the standard document table format (ID, title, type, status, date)

**DB layer:** Add `get_documents_by_tag(tag: str, limit: int) -> List[Row]` to `db.py`. Uses `LIKE '%tag%'` with case-insensitive matching. No new tables needed — tags are already stored as comma-separated strings in the `documents.tags` column.

### Clustered Mode

`pci topics --cluster` groups the top ~200 tags into high-level topic clusters using an LLM call.

```
pci topics --cluster            # show clustered topics
pci topics --cluster "AI/ML"    # list documents in that cluster
pci topics --cluster --refresh  # force re-clustering
```

**Implementation:**
- Collect the top 200 tags by count
- Send to Claude Haiku with a prompt: "Group these tags into 10-20 high-level topic categories. Return JSON: `[{"name": "Topic Name", "tags": ["tag1", "tag2", ...]}]`"
- Display as a Rich table: topic name, sample tags, document count
- Cache the result to `.pci_topic_clusters.json` (alongside `pci.db`)
- `--refresh` flag deletes the cache and re-runs clustering
- When a cluster name argument is provided, look up its tags from the cache, query documents matching any of those tags, and display

**Cache format:**
```json
{
  "created_at": "2026-04-06T12:00:00",
  "clusters": [
    {"name": "Golf", "tags": ["golf instruction", "swing mechanics", "impact position", ...]},
    {"name": "AI/ML", "tags": ["ai agents", "machine learning", "llm", ...]}
  ]
}
```

**Cache invalidation:** The cache is valid until `--refresh` is passed. No automatic invalidation — the user controls when to re-cluster. A stale cache is better than surprise API costs.

## Feature 2: `pci dedupe`

Report-only duplicate detection across three tiers. Output is a Rich table per tier. No documents are modified or deleted.

```
pci dedupe                  # run all three tiers
pci dedupe --url-only       # tier 1 only
pci dedupe --title-only     # tier 2 only
pci dedupe --content-only   # tier 3 only
pci dedupe --threshold 0.9  # adjust similarity cutoff for tiers 2 and 3
```

### Tier 1: URL Duplicates

Normalize all URLs by:
- Lowercasing
- Stripping protocol (`http://`, `https://`)
- Stripping trailing slashes
- Stripping common query parameters (`utm_*`, `ref`, `source`, `fbclid`, `gclid`)

Group by normalized URL. Report groups with 2+ documents.

**Implementation:** Pure Python over `get_all_documents()`. No API cost.

### Tier 2: Title Similarity

Compare all document titles pairwise using `difflib.SequenceMatcher.ratio()`. Flag pairs above threshold (default 0.85).

**Implementation:**
- Load all (id, title) pairs
- O(n^2) comparison — at 1,600 docs this is ~1.3M comparisons, but `SequenceMatcher` on short strings is fast (sub-second)
- Group connected pairs into clusters (if A~B and B~C, report {A, B, C} together)
- Display: doc IDs, titles, similarity percentage

### Tier 3: Content/Embedding Similarity

Use the existing `vec_documents` table to find documents with high cosine similarity.

**Implementation:**
- Use `sqlite-vec` to query pairwise distances. For each document, find its nearest neighbors above the threshold (default 0.92)
- Deduplicate pairs (report A↔B once, not both directions)
- Display: doc IDs, titles, similarity percentage

**Performance note:** sqlite-vec supports KNN queries. Rather than full pairwise comparison, iterate over each document and query its top-K nearest neighbors, filtering by threshold. This is O(n * K) rather than O(n^2).

### Output Format

```
═══ URL Matches (2 groups) ═══
 Doc ID   Title                                          URL
 142      The AI Bubble                                  example.com/ai-bubble
 891      The AI Bubble                                  example.com/ai-bubble?ref=tw

═══ Similar Titles (3 pairs, threshold: 85%) ═══
 Pair     Doc A                          Doc B                          Match
 1        #203 "The AI Bubble..."        #1401 "The AI Bubble Is..."    91%
 2        #55 "How LLMs Work"            #988 "How LLMs Work - Guide"   87%

═══ Content Overlap (1 pair, threshold: 92%) ═══
 Pair     Doc A                          Doc B                          Sim
 1        #55 "How LLMs Work"            #988 "Understanding LLMs"      94%
```

Each tier shows its count in the header. If a tier finds nothing, it prints "No duplicates found."

## Files to Modify

- `pci/db.py` — add `get_documents_by_tag()`, `get_all_document_titles()`, `get_all_urls()`
- `pci/cli.py` — add `topics` and `dedupe` commands
- `pci/llm.py` — add `cluster_tags()` function for the LLM clustering call

## Out of Scope

- Automatic deletion or merging of duplicates
- Scheduled/automatic duplicate detection
- Tag normalization or deduplication in the database itself
