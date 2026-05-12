export function cn(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(" ");
}

export function formatDate(date: string | null | undefined): string {
  if (!date) return "";
  const d = new Date(date);
  if (isNaN(d.getTime())) return date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatRelativeDate(date: string | null | undefined): string {
  if (!date) return "";
  const d = new Date(date);
  if (isNaN(d.getTime())) return date;
  const now = Date.now();
  const diff = now - d.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 30) return formatDate(date);
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return "just now";
}

export function sourceTypeLabel(type: string | null | undefined): string {
  switch (type) {
    case "article":
      return "Article";
    case "youtube":
      return "YouTube";
    case "pdf":
      return "PDF";
    case "markdown":
      return "Markdown";
    case "text":
      return "Text";
    default:
      return type || "Unknown";
  }
}

export function sourceTypeColor(type: string | null | undefined): string {
  switch (type) {
    case "article":
      return "bg-blue-500/10 text-blue-400 border-blue-500/20";
    case "youtube":
      return "bg-red-500/10 text-red-400 border-red-500/20";
    case "pdf":
      return "bg-amber-500/10 text-amber-400 border-amber-500/20";
    case "markdown":
      return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
    case "text":
      return "bg-purple-500/10 text-purple-400 border-purple-500/20";
    default:
      return "bg-muted text-muted-foreground border-border";
  }
}
