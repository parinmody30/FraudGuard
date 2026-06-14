import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Bell, RefreshCw } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { useHealth } from "@/hooks/useModel";

export function AppLayout() {
  const { data: health, isError } = useHealth();
  const online = Boolean(health) && !isError;
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-grid-light">
      <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex items-center justify-end gap-4 border-b border-slate-200 bg-white px-8 py-3">
          <span className="inline-flex items-center gap-2 text-xs font-medium text-slate-500">
            <span
              className={`h-2 w-2 rounded-full ${online ? "bg-emerald-500" : "bg-slate-300"}`}
              title={online ? `API ${health?.version}` : "API offline"}
            />
            {online ? `API online · v${health?.version}` : "API offline"}
          </span>
          <button className="rounded-lg p-2 text-slate-400 hover:bg-slate-100" title="Refresh">
            <RefreshCw className="h-4 w-4" />
          </button>
          <button className="relative rounded-lg p-2 text-slate-400 hover:bg-slate-100" title="Notifications">
            <Bell className="h-4 w-4" />
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-red-500" />
          </button>
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-brand text-xs font-bold text-white">
            FL
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
