import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useDocuments } from "../hooks/useDocuments";
import DocumentRow from "../components/DocumentRow";
import SearchBar from "../components/SearchBar";
import { ChevronLeft, ChevronRight } from "lucide-react";

const SOURCE_TYPES = ["", "article", "youtube", "pdf", "markdown", "text"];

export default function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [page, setPage] = useState(1);
  const [isRead, setIsRead] = useState<boolean | undefined>(undefined);
  const [sourceType, setSourceType] = useState<string>(searchParams.get("source_type") || "");
  const [sort, setSort] = useState("newest");
  const [searchText, setSearchText] = useState(searchParams.get("tag") || "");

  const params = {
    page,
    per_page: 30,
    is_read: isRead,
    source_type: sourceType || undefined,
    sort,
    tag: searchText || undefined,
  };

  const { data, isLoading } = useDocuments(params);

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {data ? `${data.total} documents` : "Loading..."}
      </p>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-border bg-card p-0.5">
          {(["newest", "oldest", "title"] as const).map((s) => (
            <button
              key={s}
              onClick={() => { setSort(s); setPage(1); }}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                sort === s
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {s === "newest" ? "Newest" : s === "oldest" ? "Oldest" : "Title"}
            </button>
          ))}
        </div>

        <div className="flex rounded-lg border border-border bg-card p-0.5">
          {([
            { value: undefined, label: "All" },
            { value: false, label: "Unread" },
            { value: true, label: "Read" },
          ] as const).map((f) => (
            <button
              key={String(f.value)}
              onClick={() => { setIsRead(f.value); setPage(1); }}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                isRead === f.value
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        <select
          value={sourceType}
          onChange={(e) => {
            const nextSourceType = e.target.value;
            setSourceType(nextSourceType);
            setSearchParams({
              ...(searchText ? { tag: searchText } : {}),
              ...(nextSourceType ? { source_type: nextSourceType } : {}),
            }, { replace: true });
            setPage(1);
          }}
          className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs outline-none focus:border-ring"
        >
          <option value="">All types</option>
          {SOURCE_TYPES.filter(Boolean).map((t) => (
            <option key={t} value={t}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </option>
          ))}
        </select>

        <div className="ml-auto w-64">
          <SearchBar
            value={searchText}
            onChange={(v) => {
              setSearchText(v);
              setSearchParams({
                ...(v ? { tag: v } : {}),
                ...(sourceType ? { source_type: sourceType } : {}),
              }, { replace: true });
              setPage(1);
            }}
            onSearch={() => setPage(1)}
            placeholder="Filter by tag..."
          />
        </div>
      </div>

      {isLoading ? (
        <div className="mt-8 text-sm text-muted-foreground">Loading...</div>
      ) : data && data.items.length > 0 ? (
        <>
          <div className="mt-4 space-y-2">
            {data.items.map((doc) => (
              <DocumentRow key={doc.id} doc={doc} />
            ))}
          </div>

          {data.pages > 1 && (
            <div className="mt-6 flex items-center justify-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="rounded-md border border-border p-2 text-muted-foreground hover:text-foreground disabled:opacity-30"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="text-sm text-muted-foreground">
                {data.page} / {data.pages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
                disabled={page >= data.pages}
                className="rounded-md border border-border p-2 text-muted-foreground hover:text-foreground disabled:opacity-30"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="mt-16 text-center">
          <p className="text-sm text-muted-foreground">No documents found</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Try adding content via the CLI or the Add page.
          </p>
        </div>
      )}
    </div>
  );
}
