import { useParams, useNavigate, Link } from "react-router-dom";
import { ExternalLink, Trash2, CheckCircle, Circle, ArrowLeft, AlertCircle } from "lucide-react";
import { useDocument, useUpdateDocument, useDeleteDocument } from "../hooks/useDocuments";
import TagBadge from "../components/TagBadge";
import { cn, sourceTypeLabel, sourceTypeColor, formatDate } from "../lib/utils";

export default function DocumentPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: doc, isLoading, error } = useDocument(Number(id));
  const updateDoc = useUpdateDocument();
  const deleteDoc = useDeleteDocument();

  if (isLoading) {
    return (
      <div className="space-y-4 animate-fade-in">
        <div className="h-3.5 w-24 animate-pulse rounded bg-muted" />
        <div className="h-6 w-2/3 animate-pulse rounded bg-muted" />
        <div className="h-40 animate-pulse rounded-md border border-ink-border bg-paper" />
        <div className="h-64 animate-pulse rounded-md border border-ink-border bg-paper" />
      </div>
    );
  }

  if (error || !doc) {
    return (
      <div className="animate-fade-in">
        <Link
          to="/library"
          className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Library
        </Link>
        <div className="mt-8 rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
          <AlertCircle className="mx-auto h-5 w-5 text-ink-muted" />
          <p className="mt-2 text-xs text-ink-muted font-medium">Document not found.</p>
          <p className="mt-1 text-[11px] text-ink-muted">It may have been deleted, or the link is stale.</p>
          <Link
            to="/library"
            className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-colors hover:bg-cobalt-light/70 focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to library
          </Link>
        </div>
      </div>
    );
  }

  const tags = doc.tags
    ? doc.tags.split(",").map((t) => t.trim()).filter(Boolean)
    : [];

  const handleToggleRead = () => {
    updateDoc.mutate({ id: doc.id, data: { is_read: !doc.is_read } });
  };

  const handleDelete = () => {
    if (confirm("Delete this document? This cannot be undone.")) {
      deleteDoc.mutate(doc.id, {
        onSuccess: () => navigate("/library"),
      });
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Link
        to="/library"
        className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Library
      </Link>

      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-bold tracking-tight text-ink">
            {doc.title || doc.url || "Untitled"}
          </h1>

          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "inline-flex items-center rounded border px-1.5 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-wider",
                sourceTypeColor(doc.source_type),
              )}
            >
              {sourceTypeLabel(doc.source_type)}
            </span>

            <button
              onClick={handleToggleRead}
              className={cn(
                "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-medium transition-colors focus:outline-none focus:ring-1 focus:ring-primary",
                doc.is_read
                  ? "border-emerald-200 bg-emerald-50/50 text-emerald-700"
                  : "border-amber-200 bg-amber-50/50 text-amber-700",
              )}
            >
              {doc.is_read ? (
                <>
                  <CheckCircle className="h-3 w-3" />
                  Read
                </>
              ) : (
                <>
                  <Circle className="h-3 w-3" />
                  Unread
                </>
              )}
            </button>

            {doc.created_at && (
              <span className="text-[11px] text-ink-muted">Added {formatDate(doc.created_at)}</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {doc.url && (
            <a
              href={doc.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded p-2 text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
              title="Open in browser"
              aria-label="Open in browser"
            >
              <ExternalLink className="h-4 w-4" />
            </a>
          )}
          <button
            onClick={handleDelete}
            className="rounded p-2 text-ink-muted transition-colors hover:bg-red-50 hover:text-red-600 focus:outline-none focus:ring-1 focus:ring-primary"
            title="Delete"
            aria-label="Delete document"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
              <TagBadge tag={tag} />
            </Link>
          ))}
        </div>
      )}

      {doc.summary && (
        <div className="rounded-md border border-ink-border bg-paper p-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
            Summary
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-ink">{doc.summary}</p>
        </div>
      )}

      {doc.content && (
        <div className="rounded-md border border-ink-border bg-pure p-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
            Content
          </h2>
          <div className="mt-2 max-h-96 overflow-y-auto pr-2">
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-ink-muted">
              {doc.content.length > 10000
                ? doc.content.slice(0, 10000) + "\n\n… (truncated)"
                : doc.content}
            </pre>
          </div>
        </div>
      )}

      {doc.url && (
        <div className="rounded-md border border-ink-border bg-paper p-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
            Source URL
          </h2>
          <a
            href={doc.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 block break-all text-sm text-cobalt transition-colors hover:underline focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {doc.url}
          </a>
        </div>
      )}
    </div>
  );
}
