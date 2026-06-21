import { Link, useSearchParams } from "react-router-dom";
import { useDocuments } from "../hooks/useDocuments";
import DocumentRow from "../components/DocumentRow";
import SearchBar from "../components/SearchBar";
import { AlertCircle, ChevronLeft, ChevronRight, PlusCircle } from "lucide-react";

const SOURCE_TYPES = ["", "article", "youtube", "pdf", "markdown", "text"];

/** Parse the read filter from the URL, keeping compat with Home's ?is_read=false links. */
function readIsRead(value: string | null): boolean | undefined {
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}

function ListSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 px-3.5 py-3 animate-pulse">
          <div className="h-4 w-4 rounded bg-muted shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-3.5 w-2/3 rounded bg-muted" />
            <div className="h-3 w-1/3 rounded bg-muted" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Single source of truth: the URL. Refresh / share / forward preserves state.
  const sort = searchParams.get("sort") || "newest";
  const isRead = readIsRead(searchParams.get("is_read"));
  const sourceType = searchParams.get("source_type") || "";
  const searchText = searchParams.get("tag") || "";
  const page = Number(searchParams.get("page")) || 1;

  /** Merge a partial update into the URL, preserving unrelated keys, resetting to page 1. */
  const updateParams = (updates: Record<string, string | null>, resetPage = true) => {
    const next = new URLSearchParams(searchParams);
    Object.entries(updates).forEach(([k, v]) => {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    });
    if (resetPage) {
      if (page !== 1) next.set("page", "1");
      else next.delete("page");
    }
    setSearchParams(next, { replace: true });
  };

  const { data, isLoading, isError, refetch } = useDocuments({
    page,
    per_page: 30,
    is_read: isRead,
    source_type: sourceType || undefined,
    sort,
    tag: searchText || undefined,
  });

  const filterBtn = (active: boolean) =>
    `rounded px-3 py-1.5 text-xs font-medium transition-colors focus:outline-none ${
      active
        ? "bg-cobalt-light/60 text-cobalt font-semibold"
        : "text-ink-muted hover:text-ink"
    }`;

  const setPageParam = (next: number) => {
    const params = new URLSearchParams(searchParams);
    if (next <= 1) params.delete("page");
    else params.set("page", String(next));
    setSearchParams(params, { replace: true });
  };

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Library</h1>
        <p className="text-xs text-ink-muted">
          {data ? `${data.total} document${data.total === 1 ? "" : "s"} in your archive` : "Loading your archive…"}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-md border border-ink-border bg-paper p-0.5">
          {(["newest", "oldest", "title"] as const).map((s) => (
            <button
              key={s}
              onClick={() => updateParams({ sort: s === "newest" ? null : s })}
              className={filterBtn(sort === s)}
            >
              {s === "newest" ? "Newest" : s === "oldest" ? "Oldest" : "Title"}
            </button>
          ))}
        </div>

        <div className="flex rounded-md border border-ink-border bg-paper p-0.5">
          {([
            { value: undefined, label: "All", param: null },
            { value: false, label: "Unread", param: "false" },
            { value: true, label: "Read", param: "true" },
          ] as const).map((f) => (
            <button
              key={f.label}
              onClick={() => updateParams({ is_read: f.param })}
              className={filterBtn(isRead === f.value)}
            >
              {f.label}
            </button>
          ))}
        </div>

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

        <div className="ml-auto w-64">
          <SearchBar
            value={searchText}
            onChange={(v) => updateParams({ tag: v || null })}
            onSearch={() => updateParams({ tag: searchText || null })}
            placeholder="Filter by tag..."
          />
        </div>
      </div>

      {isLoading ? (
        <ListSkeleton count={6} />
      ) : isError ? (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4 text-center">
          <div className="flex items-center justify-center gap-2 text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-xs font-semibold">Failed to load the library.</span>
          </div>
          <button
            onClick={() => refetch()}
            className="mt-2.5 inline-flex items-center gap-1 rounded border border-red-200 bg-pure px-2.5 py-1 text-xs font-semibold text-red-700 transition-colors hover:bg-red-50 focus:outline-none focus:ring-1 focus:ring-red-500"
          >
            Retry connection
          </button>
        </div>
      ) : data && data.items.length > 0 ? (
        <>
          <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
            {data.items.map((doc) => (
              <DocumentRow key={doc.id} doc={doc} />
            ))}
          </div>

          {data.pages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={() => setPageParam(Math.max(1, page - 1))}
                disabled={page <= 1}
                aria-label="Previous page"
                className="rounded-md border border-ink-border p-2 text-ink-muted transition-colors hover:bg-accent hover:text-ink disabled:opacity-30 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="text-xs font-mono text-ink-muted">
                {data.page} / {data.pages}
              </span>
              <button
                onClick={() => setPageParam(Math.min(data.pages, page + 1))}
                disabled={page >= data.pages}
                aria-label="Next page"
                className="rounded-md border border-ink-border p-2 text-ink-muted transition-colors hover:bg-accent hover:text-ink disabled:opacity-30 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
          <p className="text-xs text-ink-muted font-medium">No documents match these filters.</p>
          <p className="mt-1 text-[11px] text-ink-muted">Try clearing a filter, or ingest something new.</p>
          <Link
            to="/add"
            className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-colors hover:bg-cobalt-light/70 focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            Add a source
          </Link>
        </div>
      )}
    </div>
  );
}
