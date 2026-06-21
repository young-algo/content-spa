import { Link, useSearchParams } from "react-router-dom";
import { useDocuments } from "../hooks/useDocuments";
import DocumentRow from "../components/DocumentRow";
import SearchBar from "../components/SearchBar";
import SegmentedControl from "../components/SegmentedControl";
import { RowSkeleton } from "../components/Skeleton";
import { AlertCircle, ChevronLeft, ChevronRight, PlusCircle } from "lucide-react";

const SOURCE_TYPES = ["", "article", "youtube", "pdf", "markdown", "text"];

type ReadFilter = "all" | "unread" | "read";

export default function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Single source of truth: the URL. Refresh / share / forward preserves state.
  const sort = searchParams.get("sort") || "newest";
  const readParam = searchParams.get("is_read");
  const readValue: ReadFilter =
    readParam === "true" ? "read" : readParam === "false" ? "unread" : "all";
  const isRead: boolean | undefined =
    readValue === "read" ? true : readValue === "unread" ? false : undefined;
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
        <SegmentedControl
          value={sort}
          onChange={(s) => updateParams({ sort: s === "newest" ? null : s })}
          options={[
            { value: "newest", label: "Newest" },
            { value: "oldest", label: "Oldest" },
            { value: "title", label: "Title" },
          ]}
        />

        <SegmentedControl
          value={readValue}
          onChange={(v) =>
            updateParams({ is_read: v === "all" ? null : v === "read" ? "true" : "false" })
          }
          options={[
            { value: "all", label: "All" },
            { value: "unread", label: "Unread" },
            { value: "read", label: "Read" },
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
        <RowSkeleton count={6} />
      ) : isError ? (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4 text-center">
          <div className="flex items-center justify-center gap-2 text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-xs font-semibold">Couldn't load the library.</span>
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
