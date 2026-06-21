import { useRef, useEffect, type KeyboardEvent, type ChangeEvent } from "react";
import { Search, X } from "lucide-react";
import { cn } from "../lib/utils";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  placeholder?: string;
  className?: string;
  autoFocus?: boolean;
}

export default function SearchBar({
  value,
  onChange,
  onSearch,
  placeholder = "Search your content...",
  className,
  autoFocus,
}: SearchBarProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      onSearch();
    } else if (e.key === "Escape") {
      inputRef.current?.blur();
      if (value) onChange("");
    }
  };

  return (
    <div className={cn("relative", className)}>
      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-muted" />
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        autoFocus={autoFocus}
        aria-label="Search"
        className="w-full rounded-lg border border-ink-border bg-pure py-2.5 pl-10 pr-10 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/70 focus:border-cobalt focus:ring-1 focus:ring-primary"
      />
      {value ? (
        <button
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 rounded text-ink-muted transition-colors hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <X className="h-4 w-4" />
        </button>
      ) : (
        <kbd className="absolute right-3 top-1/2 -translate-y-1/2 rounded border border-ink-border bg-pure px-1.5 py-0.5 text-[10px] font-mono text-ink-muted">
          /
        </kbd>
      )}
    </div>
  );
}
