import { useEffect, useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";
import ShortcutOverlay from "./ShortcutOverlay";
import { fetchStats } from "../api/system";
import type { DocumentItem } from "../api/documents";
import { isEditableTarget } from "../lib/utils";

export default function Layout() {
  const navigate = useNavigate();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  // For the global "R" (resume reading) shortcut. Shares Home's ["stats"]
  // cache, so this is deduped rather than an extra request on every route.
  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: fetchStats });
  const oldestUnread = (stats?.oldest_unread as DocumentItem | null) ?? null;

  const overlayOpen = paletteOpen || shortcutsOpen;

  // Mark the body so page-level single-key shortcuts (Home's R/S/A/I/T and
  // SearchBar's "/") stand down while a modal is up.
  useEffect(() => {
    if (overlayOpen) document.body.setAttribute("data-overlay-open", "");
    else document.body.removeAttribute("data-overlay-open");
    return () => document.body.removeAttribute("data-overlay-open");
  }, [overlayOpen]);

  // Allow page-level UI (e.g. Home's Quick Actions header) to open the palette.
  useEffect(() => {
    const openPalette = () => {
      setShortcutsOpen(false);
      setPaletteOpen(true);
    };
    window.addEventListener("ci:open-palette", openPalette);
    return () => window.removeEventListener("ci:open-palette", openPalette);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;

      // ⌘K / Ctrl+K toggles the palette from anywhere.
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setShortcutsOpen(false);
        setPaletteOpen((o) => !o);
        return;
      }

      // Close whichever modal is open on Escape (palette handles its own Esc
      // too; this covers the overlay and acts as a backstop).
      if (e.key === "Escape" && (paletteOpen || shortcutsOpen)) {
        setPaletteOpen(false);
        setShortcutsOpen(false);
        return;
      }

      // Single-key shortcuts stand down while a modal is open, while typing, or
      // when focus is inside a button/link.
      if (paletteOpen || shortcutsOpen) return;
      const target = e.target as HTMLElement | null;
      if (isEditableTarget(target) || target?.closest("button") || target?.closest("a")) {
        return;
      }

      // "?" opens the shortcut overlay (Shift is expected to type "?").
      if (e.key === "?") {
        e.preventDefault();
        setShortcutsOpen(true);
        return;
      }

      // Global navigation shortcuts — these live here (not on Home) so they
      // work on every route, matching what the shortcut overlay advertises.
      if (e.repeat || mod || e.altKey || e.shiftKey) return;
      switch (e.key.toLowerCase()) {
        case "s":
          e.preventDefault();
          navigate("/search");
          break;
        case "a":
          e.preventDefault();
          navigate("/ask");
          break;
        case "i":
          e.preventDefault();
          navigate("/add");
          break;
        case "t":
          e.preventDefault();
          navigate("/topics");
          break;
        case "r":
          if (oldestUnread) {
            e.preventDefault();
            navigate(`/documents/${oldestUnread.id}`);
          }
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [paletteOpen, shortcutsOpen, navigate, oldestUnread]);

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-4 md:px-8 py-6 md:py-8 pb-24 md:pb-8">
          <Outlet />
        </div>
      </main>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onOpenShortcuts={() => {
          setPaletteOpen(false);
          setShortcutsOpen(true);
        }}
      />
      <ShortcutOverlay open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  );
}
