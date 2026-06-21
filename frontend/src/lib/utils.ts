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
      return "ARTICLE";
    case "youtube":
      return "YT";
    case "pdf":
      return "PDF";
    case "markdown":
      return "MD";
    case "text":
      return "TEXT";
    default:
      return String(type || "UNKNOWN").toUpperCase();
  }
}

export function sourceTypeColor(type: string | null | undefined): string {
  return "bg-paper text-ink-muted border-ink-border";
}
