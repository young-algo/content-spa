import { useMemo } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  BookOpen,
  Clock,
  FileText,
  Hash,
  MessageSquare,
  PlusCircle,
  Search,
  Sparkles,
} from "lucide-react";
import { fetchDocuments, type DocumentItem } from "../api/documents";
import { fetchStats } from "../api/system";
import { fetchTopics, type TopicCluster } from "../api/system";
import DocumentRow from "../components/DocumentRow";
import TagBadge from "../components/TagBadge";
import { cn, sourceTypeColor, sourceTypeLabel } from "../lib/utils";

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

export default function HomePage() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
  });

  const { data: topics, isLoading: topicsLoading } = useQuery({
    queryKey: ["topics", true],
    queryFn: () => fetchTopics(true),
  });

  const { data: recent } = useQuery({
    queryKey: ["documents", { page: 1, per_page: 6, sort: "newest" }],
    queryFn: () => fetchDocuments({ page: 1, per_page: 6, sort: "newest" }),
  });

  const { data: unreadNewest } = useQuery({
    queryKey: ["documents", { page: 1, per_page: 4, is_read: false, sort: "newest" }],
    queryFn: () => fetchDocuments({ page: 1, per_page: 4, is_read: false, sort: "newest" }),
  });

  const sourceQueries = useQueries({
    queries: SOURCE_TYPES.map((sourceType) => ({
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

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Browse</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Explore topics, unread items, and recent additions in your archive.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Link
            to="/search"
            className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <Search className="h-4 w-4" />
            Search
          </Link>
          <Link
            to="/ask"
            className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <MessageSquare className="h-4 w-4" />
            Ask
          </Link>
          <Link
            to="/add"
            className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
          >
            <PlusCircle className="h-4 w-4" />
            Add URL
          </Link>
        </div>
      </div>

      {!statsLoading && stats && (
        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <FileText className="h-3.5 w-3.5" />
              Documents
            </div>
            <p className="mt-1 text-2xl font-semibold">{stats.total}</p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <BookOpen className="h-3.5 w-3.5" />
              Unread
            </div>
            <p className="mt-1 text-2xl font-semibold text-amber-400">{stats.unread_count}</p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Hash className="h-3.5 w-3.5" />
              Tags
            </div>
            <p className="mt-1 text-2xl font-semibold">{stats.top_tags?.length || topTags.length}</p>
          </div>
        </div>
      )}

      <section className="mt-8">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-medium text-muted-foreground">Topic Clusters</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              AI-grouped paths through your tag space
              {topics?.cluster_created_at ? ` · refreshed ${new Date(topics.cluster_created_at).toLocaleDateString()}` : ""}.
            </p>
          </div>
          <Link to="/topics" className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
            All topics
            <ArrowRight className="h-3 w-3" />
          </Link>
        </div>

        {topicsLoading ? (
          <div className="mt-3 text-sm text-muted-foreground">Loading topics...</div>
        ) : clusters.length > 0 ? (
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {clusters.slice(0, 6).map((cluster) => {
              const tagCount = cluster.tags.reduce((total, tag) => {
                const found = topics?.tags.find((item) => item.tag === tag);
                return total + (found?.count ?? 0);
              }, 0);

              return (
                <div key={cluster.name} className="rounded-lg border border-border bg-card p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <Link to={topicPath(cluster)} className="text-sm font-medium hover:text-primary">
                        {cluster.name}
                      </Link>
                      {cluster.description && (
                        <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                          {cluster.description}
                        </p>
                      )}
                      <p className="mt-2 text-[10px] text-muted-foreground">
                        {tagCount || cluster.tags.length} tag match{tagCount === 1 ? "" : "es"}
                      </p>
                    </div>
                    <Link
                      to={topicPath(cluster)}
                      className="shrink-0 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                    >
                      Open
                    </Link>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {cluster.tags.slice(0, 6).map((tag) => (
                      <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
                        <TagBadge tag={tag} size="sm" />
                      </Link>
                    ))}
                    {cluster.tags.length > 6 && (
                      <span className="text-[10px] text-muted-foreground">
                        +{cluster.tags.length - 6}
                      </span>
                    )}
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    <Link
                      to={`/ask?topic=${encodeURIComponent(cluster.name)}&mode=ask`}
                      className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary"
                    >
                      <MessageSquare className="h-3.5 w-3.5" />
                      Ask
                    </Link>
                    <Link
                      to={`/ask?topic=${encodeURIComponent(cluster.name)}&mode=synthesize`}
                      className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      Synthesize
                    </Link>
                    <Link
                      to={`/search?q=${encodeURIComponent(cluster.name)}`}
                      className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary"
                    >
                      <Search className="h-3.5 w-3.5" />
                      Search
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="mt-3 rounded-lg border border-border bg-card p-4">
            <p className="text-sm text-muted-foreground">
              No clusters available yet. Browse raw tags or ingest more content.
            </p>
            {topTags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {topTags.map(({ tag }) => (
                  <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
                    <TagBadge tag={tag} />
                  </Link>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]">
        <section>
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">Recent Additions</h2>
            <Link to="/library" className="text-xs text-primary hover:underline">
              View library
            </Link>
          </div>
          <div className="mt-3 space-y-2">
            {recent?.items.map((doc) => (
              <DocumentRow key={doc.id} doc={doc} />
            ))}
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">Reading Queue</h2>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </div>

          {stats?.oldest_unread && (
            <Link
              to={`/documents/${stats.oldest_unread.id}`}
              className="mt-3 block rounded-lg border border-amber-500/20 bg-amber-500/5 p-4 hover:border-amber-500/40"
            >
              <p className="text-xs font-medium text-amber-400">Oldest unread</p>
              <h3 className="mt-1 line-clamp-2 text-sm font-medium text-foreground">
                {getDocumentTitle(stats.oldest_unread as Partial<DocumentItem>)}
              </h3>
            </Link>
          )}

          <div className="mt-3 space-y-2">
            {unreadNewest?.items.map((doc) => (
              <DocumentRow key={doc.id} doc={doc} />
            ))}
            {unreadNewest?.items.length === 0 && (
              <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
                No unread documents.
              </p>
            )}
          </div>
        </section>
      </div>

      <section className="mt-8">
        <h2 className="text-sm font-medium text-muted-foreground">Browse By Source</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {SOURCE_TYPES.map((sourceType, index) => {
            const query = sourceQueries[index];
            const count = sourceCounts.get(sourceType) ?? query.data?.total ?? 0;
            return (
              <Link
                key={sourceType}
                to={`/library?source_type=${sourceType}`}
                className="rounded-lg border border-border bg-card p-3 hover:border-ring/50"
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={cn(
                      "rounded border px-1.5 py-0.5 text-[10px] font-medium",
                      sourceTypeColor(sourceType),
                    )}
                  >
                    {sourceTypeLabel(sourceType)}
                  </span>
                  <span className="text-xs text-muted-foreground">{count}</span>
                </div>
                <div className="mt-3 space-y-2">
                  {query.data?.items.slice(0, 2).map((doc) => (
                    <p key={doc.id} className="line-clamp-2 text-xs leading-snug text-muted-foreground">
                      {doc.title || doc.url || "Untitled"}
                    </p>
                  ))}
                </div>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
