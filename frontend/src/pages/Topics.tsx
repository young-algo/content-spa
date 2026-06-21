import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchTopics } from "../api/system";
import TagBadge from "../components/TagBadge";
import { Hash, PlusCircle, AlertCircle } from "lucide-react";

function ClusterSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="rounded-md border border-ink-border bg-pure p-4 animate-pulse">
          <div className="h-3.5 w-1/3 rounded bg-muted" />
          <div className="mt-2 h-3 w-2/3 rounded bg-muted" />
          <div className="mt-3 flex gap-1.5">
            <div className="h-5 w-12 rounded-full bg-muted" />
            <div className="h-5 w-16 rounded-full bg-muted" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function TopicsPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["topics", true],
    queryFn: () => fetchTopics(true),
  });

  const clusters = (data?.clusters ?? []) as Array<{ name: string; description?: string; tags: string[] }>;

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Topics</h1>
        <p className="text-xs text-ink-muted">Browse automatically generated topic clusters and your active tags.</p>
      </div>

      {isLoading ? (
        <ClusterSkeleton count={3} />
      ) : isError ? (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4 text-center">
          <div className="flex items-center justify-center gap-2 text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-xs font-semibold">Failed to load topics.</span>
          </div>
          <button
            onClick={() => refetch()}
            className="mt-2.5 inline-flex items-center gap-1 rounded border border-red-200 bg-pure px-2.5 py-1 text-xs font-semibold text-red-700 transition-colors hover:bg-red-50 focus:outline-none focus:ring-1 focus:ring-red-500"
          >
            Retry connection
          </button>
        </div>
      ) : data && data.tags.length === 0 && clusters.length === 0 ? (
        <div className="rounded-md border border-ink-border border-dashed p-6 text-center bg-paper/40">
          <Hash className="mx-auto h-5 w-5 text-ink-muted" />
          <p className="mt-2 text-xs text-ink-muted font-medium">No topics yet.</p>
          <p className="mt-1 text-[11px] text-ink-muted">Ingest some content and topic clusters will generate automatically.</p>
          <Link
            to="/add"
            className="mt-2.5 inline-flex items-center gap-1 rounded bg-cobalt-light px-2.5 py-1 text-xs font-semibold text-cobalt transition-colors hover:bg-cobalt-light/70 focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            Add a source
          </Link>
        </div>
      ) : data ? (
        <div className="space-y-8">
          {clusters.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
                Topic Clusters
              </h2>
              <div className="space-y-3">
                {clusters.map((cluster, i) => (
                  <div key={i} className="rounded-md border border-ink-border bg-pure p-4">
                    <h3 className="text-sm font-semibold text-ink">{cluster.name}</h3>
                    {cluster.description && (
                      <p className="mt-1 text-xs leading-relaxed text-ink-muted">{cluster.description}</p>
                    )}
                    <div className="mt-2.5 flex flex-wrap gap-1.5">
                      {cluster.tags.map((tag) => (
                        <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
                          <TagBadge tag={tag} size="sm" />
                        </Link>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
              All Tags ({data.tags.length})
            </h2>
            <div className="flex flex-wrap gap-2">
              {data.tags.map(({ tag, count }) => (
                <Link
                  key={tag}
                  to={`/library?tag=${encodeURIComponent(tag)}`}
                  className="inline-flex items-center rounded-full border border-ink-border bg-pure px-3 py-1 text-xs text-ink-muted transition-colors hover:border-cobalt/30 hover:text-cobalt focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  #{tag}
                  <span className="ml-1.5 text-[10px] font-mono text-ink-muted">{count}</span>
                </Link>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
