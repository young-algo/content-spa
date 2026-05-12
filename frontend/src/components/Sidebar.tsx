import { NavLink } from "react-router-dom";
import {
  Library,
  Search,
  MessageSquare,
  PlusCircle,
  Hash,
  Home,
} from "lucide-react";
import { cn } from "../lib/utils";

const navItems = [
  { to: "/", icon: Home, label: "Browse" },
  { to: "/library", icon: Library, label: "Library" },
  { to: "/search", icon: Search, label: "Search" },
  { to: "/ask", icon: MessageSquare, label: "Ask" },
  { to: "/add", icon: PlusCircle, label: "Add" },
  { to: "/topics", icon: Hash, label: "Topics" },
];

export default function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-card">
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <span className="text-lg">📚</span>
        <span className="font-semibold text-sm tracking-tight">Content Index</span>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border p-3">
        <p className="text-xs text-muted-foreground">Content Index v0.1.0</p>
      </div>
    </aside>
  );
}
