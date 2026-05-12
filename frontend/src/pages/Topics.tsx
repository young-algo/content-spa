import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchTopics } from "../api/system";
import TagBadge from "../components/TagBadge";
import { Loader2 } from "lucide-react";

export default function TopicsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["topics", true],
    queryFn: () => fetchTopics(true),
  });

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Topics</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Browse and explore your tag space
      </p>

      {isLoading && (
        <div className="mt-8 flex items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        </div>
      )}

      {data && data.tags.length === 0 && (
        <div className="mt-16 text-center">
          <p className="text-sm text-muted-foreground">
            No tags yet. Ingest some content to get started.
          </p>
        </div>
      )}

      {data && (
        <div className="mt-6 space-y-6">
          {data.clusters && data.clusters.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-muted-foreground mb-3">Topic Clusters</h2>
              <div className="space-y-3">
                {(data.clusters as Array<{ name: string; description?: string; tags: string[] }>).map(
                  (cluster, i) => (
                    <div key={i} className="rounded-lg border border-border bg-card p-4">
                      <h3 className="text-sm font-medium text-foreground">{cluster.name}</h3>
                      {cluster.description && (
                        <p className="mt-1 text-xs text-muted-foreground">{cluster.description}</p>
                      )}
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {cluster.tags.map((tag) => (
                          <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
                            <TagBadge tag={tag} size="sm" />
                          </Link>
                        ))}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          )}

          <div>
            <h2 className="text-sm font-medium text-muted-foreground mb-3">
              All Tags ({data.tags.length})
            </h2>
            <div className="flex flex-wrap gap-2">
              {data.tags.map(({ tag, count }) => (
                <Link key={tag} to={`/library?tag=${encodeURIComponent(tag)}`}>
                  <TagBadge
                    tag={tag}
                    className="px-3 py-1 text-xs"
                  />
                  <span className="ml-1 text-[10px] text-muted-foreground">
                    {count}
                  </span>
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
