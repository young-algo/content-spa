import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { ExternalLink, Trash2, CheckCircle, Circle, ArrowLeft, AlertCircle, Undo2, Loader2 } from "lucide-react";
import { useDocument, useUpdateDocument, useDeleteDocument } from "../hooks/useDocuments";
import { openDocumentSource } from "../api/documents";
import TagBadge from "../components/TagBadge";
import MarkdownReader from "../components/MarkdownReader";
import { cn, sourceTypeLabel, sourceTypeColor, formatDate } from "../lib/utils";

const DISPLAY_LIMIT = 30000;

function browserCanOpenDirectly(url: string) {
  return /^(https?:|mailto:)/i.test(url);
}

export default function DocumentPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: doc, isLoading, error } = useDocument(Number(id));
  const updateDoc = useUpdateDocument();
  const deleteDoc = useDeleteDocument();

  // Delayed delete with an inline Undo window — no native confirm(). The
  // pending target id is tracked in a ref so the delete commits the document
  // the user clicked, independent of whichever doc is in view when it fires.
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState(false);
  const [sourceOpenPending, setSourceOpenPending] = useState(false);
  const [sourceOpenError, setSourceOpenError] = useState(false);
  const deleteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingDeleteId = useRef<number | null>(null);

  const performDelete = (targetId: number) => {
    pendingDeleteId.current = null;
    deleteDoc.mutate(targetId, {
      onSuccess: () => navigate("/library"),
      onError: () => {
        setDeletePending(false);
        setDeleteError(true);
      },
    });
  };

  // Commit any still-pending delete — only an explicit Undo cancels it. Called
  // when leaving the page or switching to another document so the "Deleting…"
  // banner never lies about what happened.
  const flushPendingDelete = () => {
    if (deleteTimer.current) {
      clearTimeout(deleteTimer.current);
      deleteTimer.current = null;
    }
    if (pendingDeleteId.current !== null) {
      const targetId = pendingDeleteId.current;
      pendingDeleteId.current = null;
      deleteDoc.mutate(targetId);
    }
  };

  // Reset the banner for each document in view, and commit a leftover pending
  // delete from the previous one on id change / unmount.
  useEffect(() => {
    setDeletePending(false);
    setDeleteError(false);
    return () => flushPendingDelete();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const handleDeleteClick = () => {
    if (!doc) return;
    pendingDeleteId.current = doc.id;
    setDeleteError(false);
    setDeletePending(true);
    if (deleteTimer.current) clearTimeout(deleteTimer.current);
    deleteTimer.current = setTimeout(() => performDelete(doc.id), 4500);
  };

  const handleUndoDelete = () => {
    if (deleteTimer.current) clearTimeout(deleteTimer.current);
    deleteTimer.current = null;
    pendingDeleteId.current = null;
    setDeletePending(false);
  };

  const handleOpenSource = async () => {
    if (!doc) return;
    setSourceOpenPending(true);
    setSourceOpenError(false);
    try {
      await openDocumentSource(doc.id);
    } catch {
      setSourceOpenError(true);
    } finally {
      setSourceOpenPending(false);
    }
  };

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

  const wordCount = doc.content ? doc.content.trim().split(/\s+/).filter(Boolean).length : 0;
  const readingTime = Math.ceil(wordCount / 200);
  const isTruncated = doc.content ? doc.content.length > DISPLAY_LIMIT : false;
  const displayContent = isTruncated && doc.content ? doc.content.slice(0, DISPLAY_LIMIT) : (doc.content || "");

  const handleToggleRead = () => {
    updateDoc.mutate({ id: doc.id, data: { is_read: !doc.is_read } });
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

      {(deletePending || deleteError) && (
        <div
          role="status"
          aria-live="polite"
          className={
            deleteError
              ? "flex items-center justify-between gap-3 rounded-md border border-red-200 bg-red-50/40 px-4 py-2.5 animate-fade-in"
              : "flex items-center justify-between gap-3 rounded-md border border-amber-200 bg-amber-50/50 px-4 py-2.5 animate-fade-in"
          }
        >
          {deleteError ? (
            <>
              <span className="flex items-center gap-2 text-xs text-red-700">
                <AlertCircle className="h-4 w-4 shrink-0" />
                Couldn't delete this document. Try again.
              </span>
              <button
                onClick={handleDeleteClick}
                className="rounded border border-red-200 bg-pure px-2.5 py-1 text-xs font-semibold text-red-700 transition-colors hover:bg-red-50 focus:outline-none focus:ring-1 focus:ring-red-500"
              >
                Retry
              </button>
            </>
          ) : (
            <>
              <span className="flex items-center gap-2 text-xs text-amber-700">
                <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                Deleting “{doc.title || doc.url || "Untitled"}”…
              </span>
              <button
                onClick={handleUndoDelete}
                className="inline-flex items-center gap-1 rounded border border-amber-200 bg-pure px-2.5 py-1 text-xs font-semibold text-amber-700 transition-colors hover:bg-amber-50 focus:outline-none focus:ring-1 focus:ring-amber-500"
              >
                <Undo2 className="h-3.5 w-3.5" />
                Undo
              </button>
            </>
          )}
        </div>
      )}

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
          {doc.url && browserCanOpenDirectly(doc.url) && (
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
          {doc.url && !browserCanOpenDirectly(doc.url) && (
            <button
              onClick={handleOpenSource}
              disabled={sourceOpenPending}
              className="rounded p-2 text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
              title="Open source"
              aria-label="Open source"
            >
              {sourceOpenPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <ExternalLink className="h-4 w-4" />}
            </button>
          )}
          <button
            onClick={handleDeleteClick}
            disabled={deletePending || deleteDoc.isPending}
            className="rounded p-2 text-ink-muted transition-colors hover:bg-red-50 hover:text-red-600 focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
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

      {sourceOpenError && (
        <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50/40 px-4 py-2.5 text-xs text-red-700 animate-fade-in">
          <AlertCircle className="h-4 w-4 shrink-0" />
          Couldn't open the source URL from the local app.
        </div>
      )}

      {doc.content ? (
        <div className="rounded-md border border-ink-border bg-pure overflow-hidden">
          <div className="flex items-center justify-between border-b border-ink-border bg-paper/50 px-4 py-2.5">
            <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-ink-muted">
              <span>Reader</span>
              <span className="text-ink-border">•</span>
              <span className="font-normal normal-case text-ink-muted/80">{wordCount.toLocaleString()} words</span>
              <span className="text-ink-border">•</span>
              <span className="font-normal normal-case text-ink-muted/80">{readingTime} min read</span>
            </div>
            {isTruncated && (
              <span className="inline-flex items-center gap-1 rounded bg-amber-50 px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wider text-amber-700 border border-amber-200">
                Truncated
              </span>
            )}
          </div>
          <div className="p-6 md:p-8">
            <MarkdownReader content={displayContent} />
          </div>
          {isTruncated && (
            <div className="border-t border-ink-border bg-paper/50 px-6 py-4 flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-4 w-4 text-amber-600 shrink-0" />
              <div className="space-y-1">
                <p className="text-xs font-semibold text-ink">Document Truncated</p>
                <p className="text-xs text-ink-muted leading-relaxed">
                  This document exceeds the display limit of {DISPLAY_LIMIT.toLocaleString()} characters and has been truncated. 
                  {doc.url && (
                    <span>
                      {" "}You can view the complete text by opening the{" "}
                      {browserCanOpenDirectly(doc.url) ? (
                        <a
                          href={doc.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary inline-flex items-center gap-0.5 font-medium"
                        >
                          Source URL
                        </a>
                      ) : (
                        <button
                          type="button"
                          onClick={handleOpenSource}
                          disabled={sourceOpenPending}
                          className="text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary inline-flex items-center gap-0.5 font-medium disabled:opacity-50"
                        >
                          Source URL
                        </button>
                      )}.
                    </span>
                  )}
                </p>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-md border border-ink-border border-dashed bg-paper/20 p-8 text-center">
          <AlertCircle className="mx-auto h-5 w-5 text-ink-muted" />
          <p className="mt-2 text-xs text-ink-muted font-medium">No content available.</p>
          <p className="mt-1 text-[11px] text-ink-muted">This document has no text content stored in the archive.</p>
        </div>
      )}

      {doc.url && (
        <div className="rounded-md border border-ink-border bg-paper p-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
            Source URL
          </h2>
          {browserCanOpenDirectly(doc.url) ? (
            <a
              href={doc.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 block break-all text-sm text-cobalt transition-colors hover:underline focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {doc.url}
            </a>
          ) : (
            <button
              type="button"
              onClick={handleOpenSource}
              disabled={sourceOpenPending}
              className="mt-1 block break-all text-left text-sm text-cobalt transition-colors hover:underline focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
            >
              {doc.url}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
