import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Home,
  Library,
  Search,
  MessageSquare,
  PlusCircle,
  Hash,
  Clock,
  Keyboard,
  type LucideIcon,
} from "lucide-react";
import { cn } from "../lib/utils";

interface Command {
  id: string;
  label: string;
  hint?: string;
  icon: LucideIcon;
  run: () => void;
  group: "Navigate" | "Actions";
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onOpenShortcuts: () => void;
}

export default function CommandPalette({ open, onClose, onOpenShortcuts }: CommandPaletteProps) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      // defer focus until the input mounts
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const commands: Command[] = useMemo(
    () => [
      { id: "nav-home", label: "Go to Home", icon: Home, group: "Navigate", run: () => navigate("/") },
      { id: "nav-library", label: "Go to Library", icon: Library, group: "Navigate", run: () => navigate("/library") },
      { id: "nav-search", label: "Search archive", hint: "S", icon: Search, group: "Actions", run: () => navigate("/search") },
      { id: "nav-ask", label: "Ask with context", hint: "A", icon: MessageSquare, group: "Actions", run: () => navigate("/ask") },
      { id: "nav-add", label: "Ingest a source", hint: "I", icon: PlusCircle, group: "Actions", run: () => navigate("/add") },
      { id: "nav-topics", label: "Explore topics", hint: "T", icon: Hash, group: "Actions", run: () => navigate("/topics") },
      { id: "act-resume", label: "Resume reading (unread queue)", hint: "R", icon: Clock, group: "Actions", run: () => navigate("/library?is_read=false") },
      { id: "act-shortcuts", label: "Show keyboard shortcuts", hint: "?", icon: Keyboard, group: "Actions", run: onOpenShortcuts },
    ],
    [navigate, onOpenShortcuts],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q));
  }, [query, commands]);

  useEffect(() => {
    setActive(0);
  }, [filtered.length]);

  if (!open) return null;

  const runCommand = (cmd: Command | undefined) => {
    if (!cmd) return;
    cmd.run();
    onClose();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => (filtered.length ? (i + 1) % filtered.length : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => (filtered.length ? (i - 1 + filtered.length) % filtered.length : 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      runCommand(filtered[active]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[12vh] px-4 animate-fade-in"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-ink/40 backdrop-blur-sm" aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-lg overflow-hidden rounded-md border border-ink-border bg-pure shadow-xl shadow-ink/10"
      >
        <div className="flex items-center gap-2.5 border-b border-ink-border px-3.5">
          <Search className="h-4 w-4 shrink-0 text-ink-muted" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a command…"
            aria-label="Command"
            className="w-full bg-transparent py-3 text-sm text-ink outline-none placeholder:text-ink-muted/70"
          />
          <kbd className="rounded border border-ink-border bg-paper px-1.5 py-0.5 text-[10px] font-mono text-ink-muted">
            esc
          </kbd>
        </div>

        <div className="max-h-[50vh] overflow-y-auto p-1.5">
          {filtered.length === 0 ? (
            <p className="px-3 py-6 text-center text-xs text-ink-muted">No commands match “{query}”.</p>
          ) : (
            filtered.map((cmd, i) => {
              const Icon = cmd.icon;
              return (
                <button
                  key={cmd.id}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => runCommand(cmd)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded px-3 py-2 text-left text-sm transition-colors focus:outline-none",
                    i === active ? "bg-cobalt-light/60 text-cobalt" : "text-ink hover:bg-accent/50",
                  )}
                >
                  <Icon className={cn("h-4 w-4 shrink-0", i === active ? "text-cobalt" : "text-ink-muted")} />
                  <span className="flex-1 truncate font-medium">{cmd.label}</span>
                  {cmd.hint && (
                    <kbd
                      className={cn(
                        "rounded border px-1.5 py-0.5 text-[10px] font-mono",
                        i === active ? "border-cobalt/30 text-cobalt" : "border-ink-border bg-paper text-ink-muted",
                      )}
                    >
                      {cmd.hint}
                    </kbd>
                  )}
                </button>
              );
            })
          )}
        </div>

        <div className="flex items-center justify-between border-t border-ink-border bg-paper px-3.5 py-2 text-[10px] font-mono text-ink-muted">
          <span>↑↓ navigate · ↵ run</span>
          <span>Command palette</span>
        </div>
      </div>
    </div>
  );
}
