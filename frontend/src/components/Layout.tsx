import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-4 md:px-8 py-6 md:py-8 pb-24 md:pb-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
