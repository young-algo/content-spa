import { useNavigate } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import type { SearchResult } from "../api/search";
import TagBadge from "./TagBadge";
import { cn, sourceTypeLabel, sourceTypeColor, formatRelativeDate } from "../lib/utils";

interface DocumentCardProps {
  doc: SearchResult;
  className?: string;
}

export default function DocumentCard({ doc, className }: DocumentCardProps) {
  const navigate = useNavigate();
  const tags = doc.tags
    ? String(doc.tags).split(",").map((t: string) => t.trim()).filter(Boolean)
    : [];
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
      className={cn(
        "block cursor-pointer rounded-lg border border-border bg-card p-4 transition-colors hover:border-ring/50 hover:bg-card/80 focus:outline-none focus:ring-1 focus:ring-ring",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium leading-snug text-foreground">
            {doc.title || doc.url || "Untitled"}
          </h3>

          {doc.summary && (
            <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
              {doc.summary}
            </p>
          )}

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span
              className={cn(
                "inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium",
                sourceTypeColor(doc.source_type),
              )}
            >
              {sourceTypeLabel(doc.source_type)}
            </span>

            {doc.score !== null && doc.score !== undefined && (
              <span className="text-[10px] text-muted-foreground">
                {(doc.score * 100).toFixed(0)}% match
              </span>
            )}

            {doc.created_at && (
              <span className="text-[10px] text-muted-foreground">
                {formatRelativeDate(doc.created_at)}
              </span>
            )}
          </div>

          {tags.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {tags.slice(0, 5).map((tag: string) => (
                <TagBadge key={tag} tag={tag} size="sm" />
              ))}
              {tags.length > 5 && (
                <span className="text-[10px] text-muted-foreground">
                  +{tags.length - 5} more
                </span>
              )}
            </div>
          )}
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
    </div>
  );
}
