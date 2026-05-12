import { useNavigate } from "react-router-dom";
import { ExternalLink, CheckCircle, Circle } from "lucide-react";
import type { DocumentItem } from "../api/documents";
import TagBadge from "./TagBadge";
import { cn, sourceTypeLabel, sourceTypeColor, formatRelativeDate } from "../lib/utils";
import { useUpdateDocument } from "../hooks/useDocuments";

interface DocumentRowProps {
  doc: DocumentItem;
  onDelete?: (id: number) => void;
}

export default function DocumentRow({ doc, onDelete }: DocumentRowProps) {
  const navigate = useNavigate();
  const updateDoc = useUpdateDocument();
  const tags = doc.tags ? doc.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];

  const toggleRead = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    updateDoc.mutate({ id: doc.id, data: { is_read: !doc.is_read } });
  };

  const openDocument = () => navigate(`/documents/${doc.id}`);
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openDocument();
    }
  };

  return (
    <div
      role="link"
      tabIndex={0}
      onClick={openDocument}
      onKeyDown={handleKeyDown}
      className="flex cursor-pointer items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 transition-colors hover:border-ring/50 hover:bg-card/80 focus:outline-none focus:ring-1 focus:ring-ring"
    >
      <button
        onClick={toggleRead}
        className="shrink-0 rounded text-muted-foreground hover:text-primary"
        title={doc.is_read ? "Mark unread" : "Mark read"}
      >
        {doc.is_read ? (
          <CheckCircle className="h-4 w-4 text-emerald-500" />
        ) : (
          <Circle className="h-4 w-4" />
        )}
      </button>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <h4
            className={cn(
              "text-sm font-medium truncate",
              doc.is_read ? "text-muted-foreground" : "text-foreground",
            )}
          >
            {doc.title || doc.url || "Untitled"}
          </h4>
          <span
            className={cn(
              "shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-medium",
              sourceTypeColor(doc.source_type),
            )}
          >
            {sourceTypeLabel(doc.source_type)}
          </span>
        </div>

        <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
          {doc.created_at && <span>{formatRelativeDate(doc.created_at)}</span>}
          {tags.length > 0 && (
            <span className="flex items-center gap-1">
              {tags.slice(0, 3).map((tag) => (
                <span key={tag} className="text-[10px] text-muted-foreground/70">
                  #{tag}
                </span>
              ))}
            </span>
          )}
        </div>
      </div>

      {doc.url && (
        <a
          href={doc.url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="shrink-0 rounded p-1 text-muted-foreground hover:text-foreground hover:bg-accent"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      )}
    </div>
  );
}
