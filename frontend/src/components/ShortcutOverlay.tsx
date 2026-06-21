import { useEffect, useRef } from "react";

interface ShortcutOverlayProps {
  open: boolean;
  onClose: () => void;
}

interface Shortcut {
  keys: string;
  action: string;
}

const GROUPS: { title: string; items: Shortcut[] }[] = [
  {
    title: "Navigate",
    items: [
      { keys: "R", action: "Resume reading (oldest unread)" },
      { keys: "S", action: "Go to Search" },
      { keys: "A", action: "Go to Ask" },
      { keys: "I", action: "Go to Add content" },
      { keys: "T", action: "Go to Topics" },
    ],
  },
  {
    title: "Search & Ask",
    items: [
      { keys: "/", action: "Focus the search box" },
      { keys: "⌘ ↵", action: "Submit a question (Ask)" },
      { keys: "Esc", action: "Clear search / close dialog" },
    ],
  },
  {
    title: "Global",
    items: [
      { keys: "⌘ K", action: "Open the command palette" },
      { keys: "?", action: "Show this shortcut list" },
    ],
  },
];

export default function ShortcutOverlay({ open, onClose }: ShortcutOverlayProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) {
      requestAnimationFrame(() => closeRef.current?.focus());
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          onClose();
        }
      };
      window.addEventListener("keydown", onKey, true);
      return () => window.removeEventListener("keydown", onKey, true);
    }
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center px-4 animate-fade-in"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-ink/40 backdrop-blur-sm" aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-md overflow-hidden rounded-md border border-ink-border bg-pure shadow-xl shadow-ink/10"
      >
        <div className="flex items-center justify-between border-b border-ink-border px-4 py-3">
          <h2 className="text-sm font-semibold text-ink">Keyboard shortcuts</h2>
          <button
            ref={closeRef}
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <span className="text-lg leading-none">×</span>
          </button>
        </div>

        <div className="max-h-[70vh] overflow-y-auto p-2">
          {GROUPS.map((group) => (
            <div key={group.title} className="px-2 py-2">
              <h3 className="px-1 pb-1.5 text-[10px] font-bold uppercase tracking-wider text-ink-muted font-mono">
                {group.title}
              </h3>
              <ul className="space-y-0.5">
                {group.items.map((item) => (
                  <li
                    key={item.action}
                    className="flex items-center justify-between gap-3 rounded px-1 py-1.5"
                  >
                    <span className="text-xs text-ink">{item.action}</span>
                    <kbd className="shrink-0 rounded border border-ink-border bg-paper px-2 py-0.5 text-[11px] font-mono text-ink-muted">
                      {item.keys}
                    </kbd>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-ink-border bg-paper px-4 py-2 text-[10px] font-mono text-ink-muted">
          Press <span className="text-ink">⌘ K</span> for the command palette
        </div>
      </div>
    </div>
  );
}
