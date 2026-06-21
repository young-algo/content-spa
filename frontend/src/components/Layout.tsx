import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";
import ShortcutOverlay from "./ShortcutOverlay";

function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable
  );
}

export default function Layout() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

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

      // "?" opens the shortcut overlay when not typing.
      if (e.key === "?" && !paletteOpen && !shortcutsOpen && !isEditable(e.target)) {
        e.preventDefault();
        setShortcutsOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [paletteOpen, shortcutsOpen]);

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
