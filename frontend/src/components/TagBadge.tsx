import { cn } from "../lib/utils";

interface TagBadgeProps {
  tag: string;
  onClick?: () => void;
  className?: string;
  size?: "sm" | "default";
}

export default function TagBadge({ tag, onClick, className, size = "default" }: TagBadgeProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center rounded-full border border-border bg-accent transition-colors hover:bg-accent/70 hover:text-foreground",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-0.5 text-xs",
        onClick && "cursor-pointer",
        className,
      )}
    >
      {tag}
    </button>
  );
}
