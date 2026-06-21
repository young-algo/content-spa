import { useMemo, useEffect, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  BookOpen,
  Clock,
  FileText,
  Hash,
  MessageSquare,
  PlusCircle,
  Search,
  AlertCircle,
  CheckCircle,
  Loader2,
  Circle,
  ExternalLink,
  Command,
  Activity,
  Tag,
  Database,
} from "lucide-react";
import { fetchDocuments, type DocumentItem } from "../api/documents";
import { fetchStats } from "../api/system";
import { fetchTopics, type TopicCluster } from "../api/system";
import { useUpdateDocument } from "../hooks/useDocuments";
import { cn, formatRelativeDate, sourceTypeLabel, sourceTypeColor } from "../lib/utils";

const SOURCE_TYPES = ["article", "youtube", "pdf", "markdown", "text"] as const;

function slugify(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function topicPath(cluster: TopicCluster) {
  return `/topics/${slugify(cluster.name)}?name=${encodeURIComponent(cluster.name)}`;
}

function getDocumentTitle(doc: Partial<DocumentItem> | null | undefined) {
  return String(doc?.title || doc?.url || "Untitled");
}

function LaneSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 p-3.5 animate-pulse">
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

function LaneError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-md border border-red-200 bg-red-50/30 p-4 text-center">
      <div className="flex items-center justify-center gap-2 text-red-700">
        <AlertCircle className="h-4 w-4 shrink-0" />
        <span className="text-xs font-semibold">{message}</span>
      </div>
      <button
        onClick={onRetry}
        className="mt-2.5 inline-flex items-center gap-1 rounded border border-red-200 bg-pure px-2.5 py-1 text-xs font-semibold text-red-700 transition-all duration-150 hover:bg-red-50 focus:outline-none focus:ring-1 focus:ring-red-500"
      >
        Retry connection
      </button>
    </div>
  );
}

interface CompactDocumentRowProps {
  doc: DocumentItem;
  isMutating: boolean;
  onToggleRead: (id: number, currentRead: boolean, title: string) => void;
}

function CompactDocumentRow({ doc, isMutating, onToggleRead }: CompactDocumentRowProps) {
  const tags = doc.tags ? doc.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];

  return (
    <div className="group flex items-center gap-3 px-3.5 py-2.5 transition-colors duration-150 hover:bg-accent/40 focus-within:bg-accent/30">
      {/* Toggle Read Button */}
      <button
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onToggleRead(doc.id, doc.is_read === 1, doc.title || doc.url || "Untitled");
        }}
        disabled={isMutating}
        className="shrink-0 rounded text-ink-muted hover:text-cobalt transition-colors focus:outline-none focus:ring-1 focus:ring-primary"
        title={doc.is_read === 1 ? "Mark unread" : "Mark read"}
      >
        {isMutating ? (
          <Loader2 className="h-4 w-4 animate-spin text-cobalt" />
        ) : doc.is_read === 1 ? (
          <CheckCircle className="h-4 w-4 text-emerald-600" />
        ) : (
          <Circle className="h-4 w-4 text-ink-muted/40 group-hover:text-ink-muted" />
        )}
      </button>

      {/* Main Link to Document */}
      <Link
        to={`/documents/${doc.id}`}
        className="min-w-0 flex-1 flex items-center justify-between gap-3 rounded focus:outline-none focus:ring-1 focus:ring-primary"
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-medium text-ink truncate group-hover:text-cobalt transition-colors">
              {doc.title || doc.url || "Untitled"}
            </span>
            <span
              className={cn(
                "shrink-0 rounded border px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider font-semibold",
                sourceTypeColor(doc.source_type),
              )}
            >
              {sourceTypeLabel(doc.source_type)}
            </span>
          </div>

          <div className="mt-0.5 flex items-center gap-2 text-[11px] text-ink-muted font-sans">
            {doc.created_at && <span>{formatRelativeDate(doc.created_at)}</span>}
            {tags.length > 0 && (
              <span className="flex items-center gap-1.5 truncate">
                <span className="text-ink-border">|</span>
                {tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="text-[10px] text-ink-muted hover:text-ink transition-colors">
                    #{tag}
                  </span>
                ))}
              </span>
            )}
          </div>
        </div>
      </Link>

      {/* External Link */}
      {doc.url && (
        <a
          href={doc.url}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 rounded p-1 text-ink-muted hover:text-ink hover:bg-accent focus:outline-none focus:ring-1 focus:ring-primary"
          title="Open source URL"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      )}
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const updateDoc = useUpdateDocument();

  const { data: stats, isLoading: statsLoading, isError: statsError, refetch: refetchStats } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
  });

  const { data: topics, isLoading: topicsLoading, isError: topicsError, refetch: refetchTopics } = useQuery({
    queryKey: ["topics", true],
    queryFn: () => fetchTopics(true),
  });

  const { data: recent, isLoading: recentLoading, isError: recentError, refetch: refetchRecent } = useQuery({
    queryKey: ["documents", { page: 1, per_page: 6, sort: "newest" }],
    queryFn: () => fetchDocuments({ page: 1, per_page: 6, sort: "newest" }),
  });

  const { data: unreadNewest, isLoading: unreadLoading, isError: unreadError, refetch: refetchUnread } = useQuery({
    queryKey: ["documents", { page: 1, per_page: 4, is_read: false, sort: "newest" }],
    queryFn: () => fetchDocuments({ page: 1, per_page: 4, is_read: false, sort: "newest" }),
  });

  const resolvedSourceTypes = useMemo(() => {
    if (statsLoading || statsError || !stats?.by_source_type) {
      return SOURCE_TYPES;
    }
    const statsTypes = stats.by_source_type.map(item => String(item.source_type || "")).filter(Boolean);
    const knownSet = new Set<string>(SOURCE_TYPES);
    
    const presentKnown = SOURCE_TYPES.filter(type => statsTypes.includes(type));
    const unknownTypes = statsTypes.filter(type => !knownSet.has(type));
    
    return [...presentKnown, ...unknownTypes];
  }, [stats?.by_source_type, statsLoading, statsError]);

  const sourceQueries = useQueries({
    queries: resolvedSourceTypes.map((sourceType) => ({
      queryKey: ["documents", { source_type: sourceType, per_page: 3, sort: "newest" }],
      queryFn: () => fetchDocuments({ source_type: sourceType, per_page: 3, sort: "newest" }),
    })),
  });

  const clusters = topics?.clusters ?? [];
  const topTags = topics?.tags.slice(0, 18) ?? [];
  const sourceCounts = useMemo(() => {
    const counts = new Map<string, number>();
    stats?.by_source_type.forEach((item) => {
      counts.set(String(item.source_type || ""), Number(item.count || 0));
    });
    return counts;
  }, [stats?.by_source_type]);

  const oldestUnread = stats?.oldest_unread as DocumentItem | null;

  const [lastMarkedRead, setLastMarkedRead] = useState<{ id: number; title: string } | null>(null);

  const handleToggleRead = (id: number, currentRead: boolean, title?: string) => {
    updateDoc.mutate(
      { id, data: { is_read: !currentRead } },
      {
        onSuccess: (updatedDoc) => {
          if (!currentRead) {
            setLastMarkedRead({ id, title: title || updatedDoc?.title || "Untitled" });
          } else {
            if (lastMarkedRead?.id === id) {
              setLastMarkedRead(null);
            }
          }
        },
      }
    );
  };

  const handleUndoMarkRead = (id: number) => {
    updateDoc.mutate(
      { id, data: { is_read: false } },
      {
        onSuccess: () => {
          setLastMarkedRead(null);
        },
      }
    );
  };

  // Keyboard Shortcuts for Raycast-like command console speed
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;
      if (document.body.hasAttribute("data-overlay-open")) return;
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;

      const target = e.target as HTMLElement | null;
      if (!target) return;

      const isEditable =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable ||
        target.getAttribute("contenteditable") === "true" ||
        target.getAttribute("role") === "textbox";

      const isInteractive =
        target.tagName === "BUTTON" ||
        target.tagName === "A" ||
        target.closest("button") !== null ||
        target.closest("a") !== null;

      if (isEditable || isInteractive) {
        return;
      }

      const key = e.key.toLowerCase();

      if (key === "s") {
        e.preventDefault();
        navigate("/search");
      } else if (key === "a") {
        e.preventDefault();
        navigate("/ask");
      } else if (key === "i") {
        e.preventDefault();
        navigate("/add");
      } else if (key === "t") {
        e.preventDefault();
        navigate("/topics");
      } else if (key === "r" && oldestUnread) {
        e.preventDefault();
        navigate(`/documents/${oldestUnread.id}`);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate, oldestUnread]);

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      {/* 1. Page Header */}
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold tracking-tight text-ink">
            Archive
          </h1>
        </div>
        <p className="text-xs text-ink-muted">
          Resume, retrieve, and route new material from one place.
        </p>
      </div>

      {/* 2. Quick Actions */}
      <div className="rounded-md border border-ink-border bg-paper p-4 md:p-5">
        <div className="flex items-center gap-3 border-b border-ink-border/60 pb-3 mb-2.5">
          <Command className="h-4 w-4 text-ink-muted shrink-0" />
          <div className="text-[13px] text-ink font-semibold flex-1">
            Quick Actions
          </div>
          <button
            onClick={() => window.dispatchEvent(new CustomEvent("ci:open-palette"))}
            className="inline-flex items-center gap-1.5 rounded border border-ink-border bg-pure px-2 py-1 text-[10px] font-mono text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
            title="Open command palette"
          >
            <Command className="h-3 w-3" />
            ⌘K
          </button>
        </div>

        <div className="space-y-0.5">
          {/* Action 1: Resume Reading */}
          <Link
            to={oldestUnread ? `/documents/${oldestUnread.id}` : "/library?is_read=false"}
            className="flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors duration-150 hover:bg-accent/60 group focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <div className="flex items-center gap-3 min-w-0">
              <Clock className="h-4 w-4 text-ink-muted shrink-0 group-hover:text-cobalt transition-colors duration-150" />
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center">
                <span className="text-xs font-semibold text-ink group-hover:text-cobalt transition-colors">Resume Reading</span>
                <span className="text-[11px] text-ink-muted truncate sm:ml-2">
                  {oldestUnread ? `— ${getDocumentTitle(oldestUnread)}` : "— Nothing pending, all caught up!"}
                </span>
              </div>
            </div>
            {oldestUnread ? (
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Resume</span>
                <kbd className="h-5 w-5 flex items-center justify-center rounded border border-ink-border bg-pure text-[10px] font-mono text-ink-muted">
                  R
                </kbd>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Queue empty</span>
              </div>
            )}
          </Link>

          {/* Action 2: Search Archive */}
          <Link
            to="/search"
            className="flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors duration-150 hover:bg-accent/60 group focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <div className="flex items-center gap-3 min-w-0">
              <Search className="h-4 w-4 text-ink-muted shrink-0 group-hover:text-cobalt transition-colors duration-150" />
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center">
                <span className="text-xs font-semibold text-ink group-hover:text-cobalt transition-colors">Search Archive</span>
                <span className="text-[11px] text-ink-muted truncate sm:ml-2">
                  — Find exact matches, keywords, or tags
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Retrieve</span>
              <kbd className="h-5 w-5 flex items-center justify-center rounded border border-ink-border bg-pure text-[10px] font-mono text-ink-muted">
                S
              </kbd>
            </div>
          </Link>

          {/* Action 3: Ask with Context */}
          <Link
            to="/ask"
            className="flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors duration-150 hover:bg-accent/60 group focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <div className="flex items-center gap-3 min-w-0">
              <MessageSquare className="h-4 w-4 text-ink-muted shrink-0 group-hover:text-cobalt transition-colors duration-150" />
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center">
                <span className="text-xs font-semibold text-ink group-hover:text-cobalt transition-colors">Ask with Context</span>
                <span className="text-[11px] text-ink-muted truncate sm:ml-2">
                  — Synthesize concepts and ask questions with AI
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Synthesize</span>
              <kbd className="h-5 w-5 flex items-center justify-center rounded border border-ink-border bg-pure text-[10px] font-mono text-ink-muted">
                A
              </kbd>
            </div>
          </Link>

          {/* Action 4: Add Source */}
          <Link
            to="/add"
            className="flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors duration-150 hover:bg-accent/60 group focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <div className="flex items-center gap-3 min-w-0">
              <PlusCircle className="h-4 w-4 text-ink-muted shrink-0 group-hover:text-cobalt transition-colors duration-150" />
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center">
                <span className="text-xs font-semibold text-ink group-hover:text-cobalt transition-colors">Add Source</span>
                <span className="text-[11px] text-ink-muted truncate sm:ml-2">
                  — Ingest new articles, YouTube transcripts, PDFs, or markdown
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Ingest</span>
              <kbd className="h-5 w-5 flex items-center justify-center rounded border border-ink-border bg-pure text-[10px] font-mono text-ink-muted">
                I
              </kbd>
            </div>
          </Link>

          {/* Action 5: Explore Topics */}
          <Link
            to="/topics"
            className="flex items-center justify-between gap-3 rounded-md px-3 py-2.5 transition-colors duration-150 hover:bg-accent/60 group focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <div className="flex items-center gap-3 min-w-0">
              <Hash className="h-4 w-4 text-ink-muted shrink-0 group-hover:text-cobalt transition-colors duration-150" />
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center">
                <span className="text-xs font-semibold text-ink group-hover:text-cobalt transition-colors">Explore Topics</span>
                <span className="text-[11px] text-ink-muted truncate sm:ml-2">
                  — Browse automatically generated topic clusters and active tags
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] text-ink-muted hidden sm:inline font-mono">Browse</span>
              <kbd className="h-5 w-5 flex items-center justify-center rounded border border-ink-border bg-pure text-[10px] font-mono text-ink-muted">
                T
              </kbd>
            </div>
          </Link>
        </div>
      </div>

      {/* 3. Responsive Workbench Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-6 md:gap-8 items-start">
        
        {/* Left Column: Primary Work Lane (Queue & Recent) */}
        <div className="space-y-6 md:space-y-8 min-w-0">
          
          {/* Section: Reading Queue */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 text-amber-500" />
                Reading Queue
              </h2>
              <Link to="/library?is_read=false" className="rounded px-1.5 py-0.5 text-xs font-mono text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary">
                View all unread
              </Link>
            </div>

            {/* Inline Undo/Recovery Affordance */}
            {lastMarkedRead && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-ink-border bg-paper px-3.5 py-2 text-xs text-ink-muted animate-fade-in">
                <span className="truncate">
                  Marked <span className="font-semibold text-ink">“{lastMarkedRead.title}”</span> as read.
                </span>
                <div className="flex items-center gap-2.5 shrink-0">
                  <button
                    onClick={() => handleUndoMarkRead(lastMarkedRead.id)}
                    className="font-mono text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary font-semibold"
                  >
                    Undo
                  </button>
                  <span className="text-ink-border">|</span>
                  <button
                    onClick={() => setLastMarkedRead(null)}
                    className="font-mono text-ink-muted hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* Oldest Unread (Promoted Session Card) */}
            {statsLoading ? (
              <div className="h-28 animate-pulse rounded-md border border-ink-border bg-paper" />
            ) : statsError ? null : oldestUnread ? (
              <div className="relative overflow-hidden rounded-md border border-cobalt/25 bg-cobalt-light/30 p-4">
                <div className="flex items-center justify-between">
                  <span className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-cobalt font-mono">
                    <BookOpen className="h-3 w-3 text-cobalt" />
                    Resume Session
                  </span>
                  <span className="text-[11px] text-ink-muted font-mono">
                    {oldestUnread.created_at && formatRelativeDate(oldestUnread.created_at)}
                  </span>
                </div>
                <h3 className="mt-2 text-sm font-semibold text-ink line-clamp-1">
                  {getDocumentTitle(oldestUnread)}
                </h3>
                {oldestUnread.tags && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {oldestUnread.tags.split(",").map(t => t.trim()).filter(Boolean).slice(0, 3).map(tag => (
                      <span key={tag} className="rounded border border-ink-border bg-pure px-2 py-0.5 text-[10px] text-ink-muted font-mono">
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/documents/${oldestUnread.id}`}
                      className="inline-flex items-center gap-1.5 rounded bg-cobalt px-3 py-1.5 text-xs font-semibold text-pure transition-all duration-150 hover:bg-cobalt/90 focus:outline-none focus:ring-1 focus:ring-primary"
                    >
                      Resume Reading
                      <ArrowRight className="h-3 w-3" />
                    </Link>
                    <button
                      onClick={() => handleToggleRead(oldestUnread.id, false, getDocumentTitle(oldestUnread))}
                      disabled={updateDoc.isPending && updateDoc.variables?.id === oldestUnread.id}
                      className="inline-flex items-center gap-1.5 rounded border border-ink-border bg-pure px-3 py-1.5 text-xs font-semibold text-ink-muted transition-all duration-150 hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
                    >
                      {updateDoc.isPending && updateDoc.variables?.id === oldestUnread.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" />
                      ) : (
                        <CheckCircle className="h-3.5 w-3.5 text-emerald-600" />
                      )}
                      Mark read
                    </button>
                  </div>
                  <span className={cn(
                    "text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border",
                    sourceTypeColor(oldestUnread.source_type)
                  )}>
                    {sourceTypeLabel(oldestUnread.source_type)}
                  </span>
                </div>
              </div>
            ) : null}

            {/* Unread List */}
            {unreadLoading ? (
              <LaneSkeleton count={3} />
            ) : unreadError ? (
              <LaneError message="Failed to load reading queue." onRetry={() => refetchUnread()} />
            ) : unreadNewest?.items && unreadNewest.items.length > 0 ? (
              <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
                {unreadNewest.items.map((doc) => (
                  <CompactDocumentRow
                    key={doc.id}
                    doc={doc}
                    isMutating={updateDoc.isPending && updateDoc.variables?.id === doc.id}
                    onToggleRead={handleToggleRead}
                  />
                ))}
              </div>
            ) : (
              <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
                <p className="text-xs text-ink-muted font-medium">All caught up! No unread documents in your queue.</p>
                <Link
                  to="/add"
                  className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-all duration-150 hover:bg-cobalt-light/80 focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  Ingest a new URL
                </Link>
              </div>
            )}
          </div>

          {/* Section: Recent Additions */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-ink flex items-center gap-1.5">
                <FileText className="h-3.5 w-3.5 text-ink-muted" />
                Recent additions
              </h2>
              <Link to="/library" className="rounded px-1.5 py-0.5 text-xs font-mono text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary">
                View full library
              </Link>
            </div>

            {recentLoading ? (
              <LaneSkeleton count={4} />
            ) : recentError ? (
              <LaneError message="Failed to load recent additions." onRetry={() => refetchRecent()} />
            ) : recent?.items && recent.items.length > 0 ? (
              <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
                {recent.items.map((doc) => (
                  <CompactDocumentRow
                    key={doc.id}
                    doc={doc}
                    isMutating={updateDoc.isPending && updateDoc.variables?.id === doc.id}
                    onToggleRead={handleToggleRead}
                  />
                ))}
              </div>
            ) : (
              <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
                <p className="text-xs text-ink-muted font-medium">No documents in the archive yet.</p>
                <Link
                  to="/add"
                  className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-all duration-150 hover:bg-cobalt-light/80 focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  Add your first source
                </Link>
              </div>
            )}
          </div>

        </div>

        {/* Right Column: Secondary Exploration Lane (Telemetry, Topics & Source Index) */}
        <div className="space-y-6 md:space-y-8">
          
          {/* Section: Telemetry */}
          <div className="rounded-md border border-ink-border bg-paper p-4">
            <div className="flex items-center gap-2 border-b border-ink-border/60 pb-2 mb-3">
              <Activity className="h-4 w-4 text-cyan" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-ink font-mono">
                Archive Telemetry
              </h3>
            </div>

            {statsLoading ? (
              <div className="space-y-2 animate-pulse">
                <div className="h-4 bg-muted rounded w-2/3" />
                <div className="h-4 bg-muted rounded w-1/2" />
                <div className="h-4 bg-muted rounded w-3/4" />
              </div>
            ) : statsError ? (
              <div className="text-xs text-red-600 flex items-center gap-1 font-mono">
                <AlertCircle className="h-3.5 w-3.5" /> Telemetry offline
              </div>
            ) : (
              <div className="space-y-2.5 text-xs font-mono">
                <div className="flex items-center justify-between border-b border-ink-border/30 pb-1.5">
                  <span className="text-ink-muted">TOTAL ARCHIVE</span>
                  <span className="text-ink font-semibold">{stats?.total ?? 0} items</span>
                </div>
                <div className="flex items-center justify-between border-b border-ink-border/30 pb-1.5">
                  <span className="text-ink-muted">PENDING QUEUE</span>
                  <span className="text-amber-600 font-bold">{stats?.unread_count ?? 0} unread</span>
                </div>
                <div className="flex items-center justify-between border-b border-ink-border/30 pb-1.5">
                  <span className="text-ink-muted">TOPIC CLUSTERS</span>
                  <span className="text-ink font-semibold">{clusters.length} clusters</span>
                </div>
                <div className="flex items-center justify-between border-b border-ink-border/30 pb-1.5">
                  <span className="text-ink-muted">ACTIVE TAGS</span>
                  <span className="text-ink font-semibold">{stats?.top_tags?.length || topTags.length || 0} active</span>
                </div>
                {topics?.cluster_created_at && (
                  <div className="flex items-center justify-between">
                    <span className="text-ink-muted">INDEX UPDATED</span>
                    <span className="text-ink font-semibold">
                      {new Date(topics.cluster_created_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        year: "numeric"
                      })}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Section: Topic Clusters */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono flex items-center gap-1.5">
                <BookOpen className="h-3.5 w-3.5 text-cobalt" />
                Topic Clusters
              </h2>
              <Link to="/topics" className="rounded px-1.5 py-0.5 text-xs font-mono text-cobalt hover:underline focus:outline-none focus:ring-1 focus:ring-primary">
                All topics
              </Link>
            </div>

            {topicsLoading ? (
              <LaneSkeleton count={2} />
            ) : topicsError ? (
              <LaneError message="Failed to load topic clusters." onRetry={() => refetchTopics()} />
            ) : clusters.length > 0 ? (
              <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
                {clusters.slice(0, 4).map((cluster) => {
                  return (
                    <div
                      key={cluster.name}
                      className="p-3"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <Link
                            to={topicPath(cluster)}
                            className="flex items-center gap-1.5 rounded text-[13px] font-semibold text-ink transition-colors hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                          >
                            <Hash className="h-3.5 w-3.5 text-cobalt shrink-0" />
                            <span className="truncate">{cluster.name}</span>
                          </Link>
                          {cluster.description && (
                            <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-ink-muted">
                              {cluster.description}
                            </p>
                          )}
                        </div>
                        
                        <div className="flex items-center gap-1.5 shrink-0">
                          <Link
                            to={topicPath(cluster)}
                            className="rounded border border-ink-border bg-pure px-2.5 py-1 text-[11px] font-semibold text-ink transition-all duration-150 hover:bg-accent focus:outline-none focus:ring-1 focus:ring-primary"
                          >
                            Open
                          </Link>
                          <Link
                            to={`/ask?topic=${encodeURIComponent(cluster.name)}&mode=ask`}
                            className="rounded border border-ink-border bg-pure p-1 text-ink-muted transition-all duration-150 hover:bg-accent hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                            title="Ask AI with this topic context"
                          >
                            <MessageSquare className="h-3.5 w-3.5" />
                          </Link>
                        </div>
                      </div>

                      {/* Tags list inside topic cluster */}
                      <div className="mt-2.5 flex flex-wrap gap-1">
                        {cluster.tags.slice(0, 3).map((tag) => (
                          <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`} className="rounded-full focus:outline-none focus:ring-1 focus:ring-primary">
                            <span className="inline-flex items-center rounded-full border border-ink-border bg-pure px-2 py-0.5 text-[10px] text-ink-muted transition-colors hover:text-cobalt hover:border-cobalt/20">
                              #{tag}
                            </span>
                          </Link>
                        ))}
                        {cluster.tags.length > 3 && (
                          <span className="inline-flex items-center rounded-full border border-ink-border bg-paper px-1.5 py-0.5 text-[9px] font-mono text-ink-muted">
                            +{cluster.tags.length - 3}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
                <p className="text-xs text-ink-muted font-medium">No topic clusters generated yet.</p>
                <p className="text-[11px] text-ink-muted mt-1">Topic clusters will appear once you ingest documents.</p>
              </div>
            )}
          </div>

          {/* Active Tags section */}
          {!topicsLoading && !topicsError && topTags.length > 0 && (
            <div className="rounded-md border border-ink-border bg-pure p-4">
              <div className="flex items-center gap-1.5 mb-2.5">
                <Tag className="h-3.5 w-3.5 text-ink-muted" />
                <h3 className="text-[10px] font-bold uppercase tracking-wider text-ink-muted font-mono">
                  Active Tags
                </h3>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {topTags.slice(0, 12).map((item) => (
                  <Link
                    key={item.tag}
                    to={`/library?tag=${encodeURIComponent(item.tag)}`}
                    className="inline-flex items-center rounded-full border border-ink-border bg-pure px-2 py-0.5 text-[11px] text-ink-muted transition-all duration-150 hover:border-cobalt/40 hover:bg-cobalt-light/30 hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    #{item.tag}
                    <span className="ml-1 text-[9px] font-mono text-ink-muted">
                      {item.count}
                    </span>
                  </Link>
                ))}
              </div>
            </div>
          )}

          {/* Section: Source Index */}
          <div className="space-y-3">
            <div className="flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5 text-ink-muted" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
                Source Index
              </h2>
            </div>

            <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
              {resolvedSourceTypes.map((sourceType, index) => {
                const query = sourceQueries[index];
                if (!query) return null;
                const count = sourceCounts.get(sourceType) ?? query.data?.total ?? 0;
                const isLoading = query.isLoading;
                const isError = query.isError;
                const refetch = query.refetch;

                return (
                  <div key={sourceType} className="p-3">
                    <div className="flex items-center justify-between gap-2">
                      <Link
                        to={`/library?source_type=${sourceType}`}
                        className="flex items-center gap-2 rounded transition-colors hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                      >
                        <span
                          className={cn(
                            "rounded border px-1.5 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider",
                            sourceTypeColor(sourceType),
                          )}
                        >
                          {sourceTypeLabel(sourceType)}
                        </span>
                      </Link>
                      <span className="text-[11px] font-mono text-ink-muted">
                        {isLoading ? "..." : `${count} item${count === 1 ? "" : "s"}`}
                      </span>
                    </div>

                    {/* Inline list of 2 latest items, extremely compact */}
                    {!isLoading && !isError && query.data?.items && query.data.items.length > 0 && (
                      <div className="mt-2 space-y-1.5 border-l border-ink-border pl-2.5 ml-1">
                        {query.data.items.slice(0, 2).map((doc) => (
                          <Link
                            key={doc.id}
                            to={`/documents/${doc.id}`}
                            className="block truncate rounded text-[11px] text-ink-muted transition-colors hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                          >
                            {doc.title || doc.url || "Untitled"}
                          </Link>
                        ))}
                      </div>
                    )}

                    {isError && (
                      <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px] text-red-600">
                        <span>Offline</span>
                        <button
                          onClick={() => refetch()}
                          className="rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] text-red-700 transition-all duration-150 hover:bg-red-100"
                        >
                          Retry
                        </button>
                      </div>
                    )}

                    {isLoading && (
                      <div className="mt-2 animate-pulse space-y-1 border-l border-ink-border pl-2.5 ml-1">
                        <div className="h-3.5 w-3/4 rounded bg-muted" />
                        <div className="h-3.5 w-1/2 rounded bg-muted" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
