import { useEffect, useMemo, useRef, useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Bell, RefreshCw } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { useHealth } from "@/hooks/useModel";
import { useActiveDataset } from "@/hooks/useActiveDataset";
import { useAuth } from "@/context/AuthContext";
import { formatCurrencyINR, formatTimestamp } from "@/lib/format";

export function AppLayout() {
  const { data: health, isError } = useHealth();
  const online = Boolean(health) && !isError;
  const [collapsed, setCollapsed] = useState(false);
  const qc = useQueryClient();
  const fetching = useIsFetching() > 0;
  const { user } = useAuth();
  const initials = (user?.name ?? "FL").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();

  return (
    <div className="flex h-screen overflow-hidden bg-grid-light print:h-auto print:overflow-visible print:bg-white">
      <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)} />
      <div className="flex flex-1 flex-col overflow-hidden print:overflow-visible">
        <header className="flex items-center justify-end gap-4 border-b border-slate-200 bg-white px-8 py-3 print:hidden">
          <span className="inline-flex items-center gap-2 text-xs font-medium text-slate-500">
            <span
              className={`h-2 w-2 rounded-full ${online ? "bg-emerald-500" : "bg-slate-300"}`}
              title={online ? `API ${health?.version}` : "API offline"}
            />
            {online ? `API online · v${health?.version}` : "API offline"}
          </span>
          <button
            className="rounded-lg p-2 text-slate-400 hover:bg-slate-100"
            title="Refresh data"
            onClick={() => qc.invalidateQueries()}
          >
            <RefreshCw className={`h-4 w-4 ${fetching ? "animate-spin" : ""}`} />
          </button>
          <AlertsMenu />
          <div
            className="flex h-8 w-8 items-center justify-center rounded-full bg-brand text-xs font-bold text-white"
            title={user ? `${user.name} · ${user.email}` : undefined}
          >
            {initials}
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-8 print:overflow-visible print:p-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

// Notification bell: lists the highest-risk BLOCK decisions in the active dataset.
function AlertsMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const ds = useActiveDataset();

  const alerts = useMemo(
    () =>
      (ds.data ?? [])
        .filter((r) => r.prediction?.decision === "BLOCK")
        .sort((a, b) => (b.prediction?.fraud_probability ?? 0) - (a.prediction?.fraud_probability ?? 0)),
    [ds.data],
  );

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        className="relative rounded-lg p-2 text-slate-400 hover:bg-slate-100"
        title="High-risk alerts"
        onClick={() => setOpen((o) => !o)}
      >
        <Bell className="h-4 w-4" />
        {alerts.length > 0 && <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-red-500" />}
      </button>
      {open && (
        <div className="absolute right-0 z-40 mt-2 w-80 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
          <div className="border-b border-slate-100 px-4 py-3">
            <div className="text-sm font-semibold text-slate-900">High-risk alerts</div>
            <div className="text-xs text-slate-500">
              {alerts.length} blocked transaction{alerts.length === 1 ? "" : "s"} in the active dataset
            </div>
          </div>
          <ul className="max-h-80 overflow-y-auto">
            {alerts.slice(0, 8).map(({ prediction, ...tx }) => (
              <li key={tx.transaction_id}>
                <button
                  className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-slate-50"
                  onClick={() => {
                    setOpen(false);
                    navigate("/investigations", { state: { transaction: tx } });
                  }}
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
                  <span className="min-w-0 flex-1">
                    <span className="flex justify-between gap-2 text-sm">
                      <span className="font-mono font-medium text-slate-800">{tx.transaction_id}</span>
                      <span className="font-semibold text-red-600">
                        {Math.round((prediction?.fraud_probability ?? 0) * 100)}%
                      </span>
                    </span>
                    <span className="block truncate text-xs text-slate-500">
                      ₹{formatCurrencyINR(tx.amount)} · {tx.merchant_name} · {formatTimestamp(tx.timestamp)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
            {alerts.length === 0 && <li className="px-4 py-6 text-center text-sm text-slate-400">No blocked transactions.</li>}
          </ul>
          {alerts.length > 0 && (
            <button
              className="w-full border-t border-slate-100 px-4 py-2.5 text-sm font-medium text-brand hover:bg-slate-50"
              onClick={() => {
                setOpen(false);
                navigate("/transactions");
              }}
            >
              View all transactions
            </button>
          )}
        </div>
      )}
    </div>
  );
}
