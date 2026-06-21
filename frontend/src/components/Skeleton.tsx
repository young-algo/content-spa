/**
 * Shared loading placeholders. RowSkeleton mirrors the connected DocumentRow
 * list (Library, Topic detail, Home lanes) — previously this exact markup was
 * copy-pasted as ListSkeleton / RowSkeleton / LaneSkeleton in three files.
 */
export function RowSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="divide-y divide-ink-border rounded-md border border-ink-border bg-pure overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 px-3.5 py-3 animate-pulse">
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
