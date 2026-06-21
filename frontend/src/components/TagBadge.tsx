import { cn } from "../lib/utils";

interface TagBadgeProps {
  tag: string;
  onClick?: () => void;
  className?: string;
  size?: "sm" | "default";
}

export default function TagBadge({ tag, onClick, className, size = "default" }: TagBadgeProps) {
  return (
    <span
      onClick={onClick}
      className={cn(
        "inline-flex items-center rounded-full border border-ink-border bg-pure text-ink-muted transition-colors",
        onClick && "cursor-pointer hover:border-cobalt/30 hover:text-cobalt",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-0.5 text-xs",
        className,
      )}
    >
      #{tag}
    </span>
  );
}
