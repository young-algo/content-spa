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
    return <div className="text-sm text-muted-foreground">Loading topic...</div>;
  }

  if (!cluster) {
    return (
      <div>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Browse
        </Link>
        <div className="mt-8 rounded-lg border border-border bg-card p-5">
          <h1 className="text-lg font-semibold">Topic not found</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Refresh topic clusters from Browse, or open a tag directly in Library.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Browse
      </Link>

      <div className="mt-4 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">{topicName}</h1>
          {cluster.description && (
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
              {cluster.description}
            </p>
          )}
          <p className="mt-2 text-xs text-muted-foreground">
            {documents.length} matched document{documents.length === 1 ? "" : "s"}
            {documents[0]?.created_at ? ` · newest ${formatRelativeDate(documents[0].created_at)}` : ""}
            {data?.cluster_created_at ? ` · clusters refreshed ${new Date(data.cluster_created_at).toLocaleDateString()}` : ""}
          </p>
        </div>

        <button
          onClick={() => {
            setRefreshVersion((value) => value + 1);
          }}
          className="inline-flex shrink-0 items-center gap-2 rounded-md border border-border px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      <div className="mt-5 flex flex-wrap gap-1.5">
        {cluster.tags.map((tag) => (
          <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
            <TagBadge tag={tag} />
          </Link>
        ))}
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Link
          to={`/ask?topic=${encodeURIComponent(topicName)}&mode=ask`}
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm hover:border-ring/50"
        >
          <MessageSquare className="h-4 w-4 text-primary" />
          Ask about topic
        </Link>
        <Link
          to={`/ask?topic=${encodeURIComponent(topicName)}&mode=synthesize`}
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm hover:border-ring/50"
        >
          <Sparkles className="h-4 w-4 text-primary" />
          Synthesize topic
        </Link>
        <Link
          to={`/search?q=${encodeURIComponent(topicName)}`}
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm hover:border-ring/50"
        >
          <FileSearch className="h-4 w-4 text-primary" />
          Search topic
        </Link>
        <Link
          to={`/library?tag=${encodeURIComponent(primaryTag)}`}
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm hover:border-ring/50"
        >
          <Library className="h-4 w-4 text-primary" />
          Open in Library
        </Link>
      </div>

      {loadingDocs ? (
        <div className="mt-8 text-sm text-muted-foreground">Loading documents...</div>
      ) : (
        <div className="mt-8 space-y-8">
          {representative.length > 0 && (
            <section>
              <h2 className="text-sm font-medium text-muted-foreground">Representative Documents</h2>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                {representative.map((doc) => (
                  <Link
                    key={doc.id}
                    to={`/documents/${doc.id}`}
                    className="rounded-lg border border-border bg-card p-4 hover:border-ring/50"
                  >
                    <h3 className="line-clamp-2 text-sm font-medium">{doc.title || doc.url || "Untitled"}</h3>
                    <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                      {doc.summary}
                    </p>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {unread.length > 0 && (
            <section>
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-medium text-muted-foreground">Unread In This Topic</h2>
                <BookOpen className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3 space-y-2">
                {unread.map((doc) => (
                  <DocumentRow key={doc.id} doc={doc} />
                ))}
              </div>
            </section>
          )}

          <section>
            <h2 className="text-sm font-medium text-muted-foreground">Recent In This Topic</h2>
            <div className="mt-3 space-y-2">
              {recent.map((doc) => (
                <DocumentRow key={doc.id} doc={doc} />
              ))}
              {recent.length === 0 && (
                <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
                  No matching documents found for this cluster yet.
                </p>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
