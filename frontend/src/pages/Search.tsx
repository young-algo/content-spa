import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useSearch } from "../hooks/useSearch";
import SearchBar from "../components/SearchBar";
import DocumentCard from "../components/DocumentCard";
import SegmentedControl from "../components/SegmentedControl";
import { AlertCircle, Search as SearchIcon } from "lucide-react";

const SOURCE_TYPES = ["", "article", "youtube", "pdf", "markdown", "text"];

function CardSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="rounded-md border border-ink-border bg-pure p-4 animate-pulse">
          <div className="h-3.5 w-2/3 rounded bg-muted" />
          <div className="mt-2 h-3 w-full rounded bg-muted" />
          <div className="mt-1.5 h-3 w-1/2 rounded bg-muted" />
        </div>
      ))}
    </div>
  );
}

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Filter state lives in the URL so refresh / share / forward preserves it.
  const query = searchParams.get("q") || "";
  const semantic = searchParams.get("semantic") !== "keyword"; // default: semantic
  const sourceType = searchParams.get("source_type") || "";

  // inputValue is the live text in the box; sync from the URL on external nav
  // (e.g. clicking a topic that deep-links to /search?q=…).
  const [inputValue, setInputValue] = useState(query);
  useEffect(() => {
    setInputValue(query);
  }, [query]);

  const { data, isLoading, error } = useSearch({
    q: query,
    semantic,
    source_type: sourceType || undefined,
    limit: 50,
  });

  /**
   * Merge a partial update into the URL, preserving unrelated keys. Filter
   * toggles replace the current entry (default) so they don't flood history;
   * only a new query submit pushes a navigable entry.
   */
  const updateParams = (
    updates: Record<string, string | null>,
    { replace = true }: { replace?: boolean } = {},
  ) => {
    const next = new URLSearchParams(searchParams);
    Object.entries(updates).forEach(([k, v]) => {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    });
    setSearchParams(next, { replace });
  };

  const handleSearch = () => {
    if (inputValue.trim()) {
      updateParams({ q: inputValue.trim() }, { replace: false });
    }
  };

  // Clearing the box clears the active query so stale results don't linger and
  // the empty-state prompt returns.
  const handleInputChange = (value: string) => {
    setInputValue(value);
    if (!value && query) updateParams({ q: null });
  };

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Search</h1>
        <p className="text-xs text-ink-muted">
          Find exact matches, keywords, or semantic concepts across your archive.
        </p>
      </div>

      <SearchBar
        value={inputValue}
        onChange={handleInputChange}
        onSearch={handleSearch}
        placeholder="Search your content… (press / to focus)"
        autoFocus
      />

      <div className="flex flex-wrap items-center gap-3">
        <SegmentedControl
          value={semantic ? "semantic" : "keyword"}
          onChange={(v) => updateParams({ semantic: v === "keyword" ? "keyword" : null })}
          options={[
            { value: "semantic", label: "Semantic" },
            { value: "keyword", label: "Keyword" },
          ]}
        />

        <select
          value={sourceType}
          onChange={(e) => updateParams({ source_type: e.target.value || null })}
          className="rounded-md border border-ink-border bg-pure px-3 py-1.5 text-xs text-ink outline-none transition-colors focus:border-cobalt focus:ring-1 focus:ring-primary"
        >
          <option value="">All types</option>
          {SOURCE_TYPES.filter(Boolean).map((t) => (
            <option key={t} value={t}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {!query && (
        <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
          <SearchIcon className="mx-auto h-5 w-5 text-ink-muted" />
          <p className="mt-2 text-xs text-ink-muted font-medium">Type a query and press Enter to search.</p>
          <p className="mt-1 text-[11px] text-ink-muted">Semantic mode finds concepts; keyword mode finds exact terms.</p>
        </div>
      )}

      {query && isLoading && <CardSkeleton count={3} />}

      {query && error && (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-xs font-semibold">Couldn't run the search. The index may need to be rebuilt.</span>
          </div>
        </div>
      )}

      {query && data && (
        <div>
          <p className="mb-4 text-xs font-mono text-ink-muted">
            {data.total} result{data.total !== 1 ? "s" : ""} · {data.semantic ? "semantic" : "keyword"}
          </p>

          {data.results.length > 0 ? (
            <div className="space-y-3">
              {data.results.map((result) => (
                <DocumentCard key={result.id} doc={result} />
              ))}
            </div>
          ) : (
            <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
              <p className="text-xs text-ink-muted font-medium">No results found for “{query}”.</p>
              <p className="mt-1 text-[11px] text-ink-muted">
                Try switching to {semantic ? "keyword" : "semantic"} mode, clearing the type filter, or rephrasing.
              </p>
              <Link
                to="/library"
                className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-colors hover:bg-cobalt-light/70 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                Browse the library instead
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
