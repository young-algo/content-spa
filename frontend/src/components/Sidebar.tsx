import { NavLink, useLocation } from "react-router-dom";
import {
  Library,
  Search,
  MessageSquare,
  PlusCircle,
  Hash,
  Home,
} from "lucide-react";
import { cn } from "../lib/utils";

// Desktop rail destinations (Add stays a rail item on desktop — sidebar
// structure is intentionally preserved).
const navItems = [
  { to: "/", icon: Home, label: "Home" },
  { to: "/library", icon: Library, label: "Library" },
  { to: "/search", icon: Search, label: "Search" },
  { to: "/ask", icon: MessageSquare, label: "Ask" },
  { to: "/add", icon: PlusCircle, label: "Add" },
  { to: "/topics", icon: Hash, label: "Topics" },
];

// Mobile bottom-nav destinations — Add is promoted to a thumb-zone FAB, so the
// bar carries 5 destinations instead of 6 (keeps labels legible and targets
// generous on narrow phones).
const mobileNavItems = navItems.filter((item) => item.to !== "/add");

export default function Sidebar() {
  const location = useLocation();
  const hideFab = location.pathname === "/add";

  return (
    <>
      {/* Desktop Compact Sidebar Rail */}
      <aside className="sidebar-dark hidden md:flex w-16 shrink-0 flex-col items-center border-r border-border bg-background py-4 justify-between h-full z-10">
        <div className="flex flex-col items-center gap-6 w-full">
          {/* Logo */}
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-card text-[13px] font-bold tracking-tight text-foreground"
            title="Content Index"
            aria-label="Content Index"
          >
            CI
          </div>

          {/* Navigation */}
          <nav className="flex flex-col gap-2 w-full px-2" aria-label="Primary">
            {navItems.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                title={label}
                aria-label={label}
                className={({ isActive }) =>
                  cn(
                    "flex h-10 w-10 items-center justify-center rounded-md transition-all duration-150 relative group focus:outline-none focus:ring-1 focus:ring-primary",
                    isActive
                      ? "bg-primary text-primary-foreground font-semibold"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground",
                  )
                }
              >
                <Icon className="h-5 w-5" />
                {/* Tooltip */}
                <div className="absolute left-14 hidden group-hover:block group-focus-within:block bg-card border border-border text-[11px] font-medium text-foreground px-2 py-1 rounded whitespace-nowrap z-50 pointer-events-none">
                  {label}
                </div>
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="flex flex-col items-center gap-4 w-full">
          <div className="h-px w-8 bg-border/40" />
          <div className="text-[10px] text-muted-foreground font-mono text-center tracking-tighter" title="v0.1.0">
            v0.1
          </div>
        </div>
      </aside>

      {/* Mobile Add FAB — thumb-zone primary action, hidden on /add */}
      {!hideFab && (
        <NavLink
          to="/add"
          aria-label="Add content"
          className="md:hidden fixed right-4 bottom-[calc(5rem+env(safe-area-inset-bottom))] z-50 flex h-14 w-14 items-center justify-center rounded-full bg-cobalt text-pure shadow-lg shadow-cobalt/30 transition-transform duration-150 hover:bg-cobalt/90 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
        >
          <PlusCircle className="h-6 w-6" />
        </NavLink>
      )}

      {/* Mobile Bottom Navigation Bar */}
      <nav
        className="sidebar-dark md:hidden fixed bottom-0 left-0 right-0 min-h-16 bg-background border-t border-border flex items-center justify-around z-40 px-2 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]"
        aria-label="Primary"
      >
        {mobileNavItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            aria-label={label}
            className={({ isActive }) =>
              cn(
                "flex flex-col items-center justify-center flex-1 rounded py-1 text-xs transition-all duration-150 focus:outline-none focus:ring-1 focus:ring-primary min-h-12",
                isActive
                  ? "text-foreground font-medium"
                  : "text-muted-foreground hover:text-foreground",
              )
            }
          >
            <Icon className="h-5 w-5" />
            <span className="text-[10px] mt-0.5">{label}</span>
          </NavLink>
        ))}
      </nav>
    </>
  );
}
