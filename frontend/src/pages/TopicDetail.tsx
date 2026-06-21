import { useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  ArrowLeft,
  BookOpen,
  FileSearch,
  Library,
  MessageSquare,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { fetchDocuments, type DocumentItem } from "../api/documents";
import { fetchTopics } from "../api/system";
import DocumentRow from "../components/DocumentRow";
import TagBadge from "../components/TagBadge";
import { RowSkeleton } from "../components/Skeleton";
import { cn, formatRelativeDate } from "../lib/utils";

function slugify(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function uniqueDocuments(groups: Array<DocumentItem[] | undefined>) {
  const docs = new Map<number, DocumentItem>();
  groups.flatMap((items) => items ?? []).forEach((doc) => docs.set(doc.id, doc));
  return Array.from(docs.values());
}

export default function TopicDetailPage() {
  const { topicSlug = "" } = useParams<{ topicSlug: string }>();
  const [searchParams] = useSearchParams();
  const requestedName = searchParams.get("name") || "";
  const [refreshVersion, setRefreshVersion] = useState(0);

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["topics", true, refreshVersion],
    queryFn: () => fetchTopics(true, undefined, refreshVersion > 0),
  });

  const cluster = useMemo(() => {
    const clusters = data?.clusters ?? [];
    return clusters.find((item) => slugify(item.name) === topicSlug) ||
      clusters.find((item) => item.name === requestedName);
  }, [data?.clusters, requestedName, topicSlug]);

  const tagQueries = useQueries({
    queries: (cluster?.tags ?? []).slice(0, 12).map((tag) => ({
      queryKey: ["documents", { tag, per_page: 20, sort: "newest" }],
      queryFn: () => fetchDocuments({ tag, per_page: 20, sort: "newest" }),
      enabled: !!cluster,
    })),
  });

  const documents = useMemo(
    () => uniqueDocuments(tagQueries.map((query) => query.data?.items)),
    [tagQueries],
  );
  const recent = documents.slice(0, 6);
  const unread = documents.filter((doc) => !doc.is_read).slice(0, 6);
  const representative = documents
    .filter((doc) => doc.summary)
    .slice(0, 4);
  const primaryTag = cluster?.tags[0] ?? requestedName;
  const topicName = cluster?.name || requestedName || topicSlug.replace(/-/g, " ");
  const loadingDocs = tagQueries.some((query) => query.isLoading);

  if (isLoading) {
    return (
      <div className="space-y-4 animate-fade-in">
        <div className="h-3.5 w-24 animate-pulse rounded bg-muted" />
        <div className="h-6 w-1/2 animate-pulse rounded bg-muted" />
        <RowSkeleton count={3} />
      </div>
    );
  }

  if (!cluster) {
    return (
      <div className="animate-fade-in">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Browse
        </Link>
        <div className="mt-8 rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
          <BookOpen className="mx-auto h-5 w-5 text-ink-muted" />
          <h1 className="mt-2 text-sm font-semibold text-ink">Topic not found</h1>
          <p className="mt-1 text-[11px] text-ink-muted">
            Refresh topic clusters from Browse, or open a tag directly in Library.
          </p>
        </div>
      </div>
    );
  }

  const sectionHeading = "text-xs font-bold uppercase tracking-wider text-ink-muted font-mono";

  const actionLink =
    "inline-flex items-center gap-2 rounded-md border border-ink-border bg-pure px-3 py-2 text-sm text-ink transition-colors hover:border-cobalt/30 hover:bg-accent/30 focus:outline-none focus:ring-1 focus:ring-primary";

  return (
    <div className="space-y-6 animate-fade-in">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Browse
      </Link>

      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-xl font-bold tracking-tight text-ink">{topicName}</h1>
          {cluster.description && (
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-muted">
              {cluster.description}
            </p>
          )}
          <p className="mt-2 text-xs text-ink-muted">
            {documents.length} matched document{documents.length === 1 ? "" : "s"}
            {documents[0]?.created_at ? ` · newest ${formatRelativeDate(documents[0].created_at)}` : ""}
            {data?.cluster_created_at ? ` · clusters refreshed ${new Date(data.cluster_created_at).toLocaleDateString()}` : ""}
          </p>
        </div>

        <button
          onClick={() => setRefreshVersion((value) => value + 1)}
          className="inline-flex shrink-0 items-center gap-2 rounded-md border border-ink-border bg-pure px-3 py-2 text-xs font-medium text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {cluster.tags.map((tag) => (
          <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
            <TagBadge tag={tag} />
          </Link>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Link to={`/ask?topic=${encodeURIComponent(topicName)}&mode=ask`} className={actionLink}>
          <MessageSquare className="h-4 w-4 text-cobalt" />
          Ask about topic
        </Link>
        <Link to={`/ask?topic=${encodeURIComponent(topicName)}&mode=synthesize`} className={actionLink}>
          <Sparkles className="h-4 w-4 text-cobalt" />
          Synthesize topic
        </Link>
        <Link to={`/search?q=${encodeURIComponent(topicName)}`} className={actionLink}>
          <FileSearch className="h-4 w-4 text-cobalt" />
          Search topic
        </Link>
        <Link to={`/library?tag=${encodeURIComponent(primaryTag)}`} className={actionLink}>
          <Library className="h-4 w-4 text-cobalt" />
          Open in Library
        </Link>
      </div>

      {loadingDocs ? (
        <RowSkeleton count={4} />
      ) : (
        <div className="space-y-8">
          {representative.length > 0 && (
            <section className="space-y-3">
              <h2 className={sectionHeading}>Representative Documents</h2>
              <div className="grid gap-3 md:grid-cols-2">
                {representative.map((doc) => (
                  <Link
                    key={doc.id}
                    to={`/documents/${doc.id}`}
                    className="rounded-md border border-ink-border bg-pure p-4 transition-colors hover:border-cobalt/30 hover:bg-accent/30 focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <h3 className="line-clamp-2 text-sm font-medium text-ink">
                      {doc.title || doc.url || "Untitled"}
                    </h3>
                    <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-ink-muted">
                      {doc.summary}
                    </p>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {unread.length > 0 && (
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className={sectionHeading}>Unread In This Topic</h2>
                <BookOpen className="h-4 w-4 text-ink-muted" />
              </div>
              <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
                {unread.map((doc) => (
                  <DocumentRow key={doc.id} doc={doc} />
                ))}
              </div>
            </section>
          )}

          <section className="space-y-3">
            <h2 className={sectionHeading}>Recent In This Topic</h2>
            {recent.length > 0 ? (
              <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
                {recent.map((doc) => (
                  <DocumentRow key={doc.id} doc={doc} />
                ))}
              </div>
            ) : (
              <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
                <p className="text-xs text-ink-muted font-medium">No matching documents for this cluster yet.</p>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
