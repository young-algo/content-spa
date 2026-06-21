import { Link } from "react-router-dom";
import { ExternalLink, CheckCircle, Circle, Loader2 } from "lucide-react";
import type { DocumentItem } from "../api/documents";
import { cn, sourceTypeLabel, sourceTypeColor, formatRelativeDate } from "../lib/utils";
import { useUpdateDocument } from "../hooks/useDocuments";

interface DocumentRowProps {
  doc: DocumentItem;
  /**
   * Optional controlled toggle. When provided (e.g. Home's mark-read + undo
   * affordance), the row delegates the mutation to the caller and uses the
   * caller's `isMutating`. Otherwise it manages the toggle itself.
   */
  onToggleRead?: (id: number, currentRead: boolean, title: string) => void;
  isMutating?: boolean;
}

/**
 * Shared scan row used by Library, Topic detail, and Home — same density, same
 * title → source/date → tags order — so users move between Browse, Library, and
 * Search without relearning the hierarchy. Render inside a connected list:
 *   <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
 */
export default function DocumentRow({ doc, onToggleRead, isMutating }: DocumentRowProps) {
  const updateDoc = useUpdateDocument();
  const mutating = onToggleRead
    ? !!isMutating
    : updateDoc.isPending && updateDoc.variables?.id === doc.id;
  const tags = doc.tags ? doc.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];

  const toggleRead = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onToggleRead) {
      onToggleRead(doc.id, !!doc.is_read, doc.title || doc.url || "Untitled");
    } else {
      updateDoc.mutate({ id: doc.id, data: { is_read: !doc.is_read } });
    }
  };

  return (
    <div className="group flex items-center gap-3 px-3.5 py-2.5 transition-colors duration-150 hover:bg-accent/40 focus-within:bg-accent/30">
      <button
        onClick={toggleRead}
        disabled={mutating}
        aria-label={doc.is_read ? "Mark unread" : "Mark read"}
        className="shrink-0 rounded text-ink-muted transition-colors hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
      >
        {mutating ? (
          <Loader2 className="h-4 w-4 animate-spin text-cobalt" />
        ) : doc.is_read ? (
          <CheckCircle className="h-4 w-4 text-emerald-600" />
        ) : (
          <Circle className="h-4 w-4 text-ink-muted/40 group-hover:text-ink-muted" />
        )}
      </button>

      <Link
        to={`/documents/${doc.id}`}
        className="min-w-0 flex-1 flex items-center justify-between gap-3 rounded focus:outline-none focus:ring-1 focus:ring-primary"
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "truncate text-[13px] font-medium transition-colors",
                doc.is_read ? "text-ink-muted" : "text-ink group-hover:text-cobalt",
              )}
            >
              {doc.title || doc.url || "Untitled"}
            </span>
            <span
              className={cn(
                "shrink-0 rounded border px-1.5 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-wider",
                sourceTypeColor(doc.source_type),
              )}
            >
              {sourceTypeLabel(doc.source_type)}
            </span>
          </div>

          <div className="mt-0.5 flex items-center gap-2 text-[11px] text-ink-muted">
            {doc.created_at && <span>{formatRelativeDate(doc.created_at)}</span>}
            {tags.length > 0 && (
              <span className="flex items-center gap-1.5 truncate">
                {tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="text-[10px] text-ink-muted">
                    #{tag}
                  </span>
                ))}
              </span>
            )}
          </div>
        </div>
      </Link>

      {doc.url && (
        <a
          href={doc.url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Open source URL"
          className="shrink-0 rounded p-1 text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      )}
    </div>
  );
}
