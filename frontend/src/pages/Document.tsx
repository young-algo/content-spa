import { useParams, useNavigate, Link } from "react-router-dom";
import { ExternalLink, Trash2, CheckCircle, Circle, ArrowLeft } from "lucide-react";
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
    return <div className="text-sm text-muted-foreground">Loading...</div>;
  }

  if (error || !doc) {
    return (
      <div className="text-center">
        <p className="text-sm text-red-400">Document not found</p>
        <Link to="/library" className="mt-2 inline-block text-xs text-primary hover:underline">
          ← Back to library
        </Link>
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
    <div>
      <Link
        to="/library"
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Library
      </Link>

      <div className="mt-4">
        <div className="flex items-start gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-xl font-semibold text-foreground">
              {doc.title || doc.url || "Untitled"}
            </h1>

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span
                className={cn(
                  "inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium",
                  sourceTypeColor(doc.source_type),
                )}
              >
                {sourceTypeLabel(doc.source_type)}
              </span>

              <button
                onClick={handleToggleRead}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium transition-colors",
                  doc.is_read
                    ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                    : "border-amber-500/20 bg-amber-500/10 text-amber-400",
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
                <span className="text-xs text-muted-foreground">
                  Added {formatDate(doc.created_at)}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1">
            {doc.url && (
              <a
                href={doc.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-lg p-2 text-muted-foreground hover:text-foreground hover:bg-accent"
                title="Open in browser"
              >
                <ExternalLink className="h-4 w-4" />
              </a>
            )}
            <button
              onClick={handleDelete}
              className="rounded-lg p-2 text-muted-foreground hover:text-red-400 hover:bg-red-500/10"
              title="Delete"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {tags.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
              <TagBadge tag={tag} />
            </Link>
          ))}
        </div>
      )}

      {doc.summary && (
        <div className="mt-6 rounded-lg border border-border bg-card p-4">
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Summary
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-foreground">{doc.summary}</p>
        </div>
      )}

      {doc.content && (
        <div className="mt-4 rounded-lg border border-border bg-card p-4">
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Content
          </h2>
          <div className="mt-2 max-h-96 overflow-y-auto">
            <pre className="whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground font-sans">
              {doc.content.length > 10000
                ? doc.content.slice(0, 10000) + "\n\n... (truncated)"
                : doc.content}
            </pre>
          </div>
        </div>
      )}

      {doc.url && (
        <div className="mt-4 rounded-lg border border-border bg-card p-4">
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            URL
          </h2>
          <a
            href={doc.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 block text-sm text-primary hover:underline break-all"
          >
            {doc.url}
          </a>
        </div>
      )}
    </div>
  );
}
