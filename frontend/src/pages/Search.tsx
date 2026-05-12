import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useSearch } from "../hooks/useSearch";
import SearchBar from "../components/SearchBar";
import DocumentCard from "../components/DocumentCard";
import { Loader2 } from "lucide-react";

const SOURCE_TYPES = ["", "article", "youtube", "pdf", "markdown", "text"];

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [inputValue, setInputValue] = useState(searchParams.get("q") || "");
  const [semantic, setSemantic] = useState(true);
  const [sourceType, setSourceType] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const query = searchParams.get("q") || "";

  const { data, isLoading, error } = useSearch({
    q: query,
    semantic,
    source_type: sourceType || undefined,
    limit: 50,
  });

  const handleSearch = () => {
    if (inputValue.trim()) {
      setSearchParams({ q: inputValue.trim() });
      setSubmitted(true);
    }
  };

  const handleSourceTypeChange = (type: string) => {
    setSourceType(type);
    setSearchParams({ q: inputValue.trim() });
  };

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Search</h1>

      <div className="mt-6">
        <SearchBar
          value={inputValue}
          onChange={(v) => {
            setInputValue(v);
            if (!v) setSubmitted(false);
          }}
          onSearch={handleSearch}
          placeholder="Search your content... (press / to focus)"
          autoFocus
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-border bg-card p-0.5">
          <button
            onClick={() => setSemantic(true)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              semantic
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Semantic
          </button>
          <button
            onClick={() => setSemantic(false)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              !semantic
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Keyword
          </button>
        </div>

        <select
          value={sourceType}
          onChange={(e) => handleSourceTypeChange(e.target.value)}
          className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs outline-none focus:border-ring"
        >
          <option value="">All types</option>
          {SOURCE_TYPES.filter(Boolean).map((t) => (
            <option key={t} value={t}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {!submitted && !query && (
        <div className="mt-16 text-center">
          <p className="text-sm text-muted-foreground">
            Type a query and press Enter to search
          </p>
        </div>
      )}

      {isLoading && (
        <div className="mt-8 flex items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        </div>
      )}

      {error && (
        <div className="mt-8 rounded-lg border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
          Search failed. The index may need to be rebuilt.
        </div>
      )}

      {data && (
        <div className="mt-4">
          <p className="mb-4 text-xs text-muted-foreground">
            {data.total} result{data.total !== 1 ? "s" : ""} · {data.semantic ? "semantic" : "keyword"}
          </p>

          <div className="space-y-3">
            {data.results.map((result) => (
              <DocumentCard key={result.id} doc={result} />
            ))}
          </div>

          {data.results.length === 0 && (
            <div className="mt-16 text-center">
              <p className="text-sm text-muted-foreground">No results found for "{query}"</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
