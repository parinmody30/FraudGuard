// Dataset history browser. Lists persisted runs from Postgres, lets the analyst pick the
// "active" dataset that drives the Dashboard + Transactions, and delete old runs. This is the
// "saved in history, no need to regenerate every time" surface.
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Database, Plus, Trash2 } from "lucide-react";
import { useDatasetHistory, useDeleteDatasetRun } from "@/hooks/useDatasetHistory";
import { useDatasetSelection } from "@/context/DatasetContext";
import { LoadingState } from "@/components/common/States";
import { formatInt, formatTimestamp } from "@/lib/format";

export function DatasetHistoryPanel() {
  const navigate = useNavigate();
  const { activeRunId, setActiveRunId } = useDatasetSelection();
  const { data: runs, isLoading, isError } = useDatasetHistory();
  const del = useDeleteDatasetRun();

  // Resolve which row is highlighted as active (selected, else the latest run).
  const effectiveActive = activeRunId && runs?.some((r) => r.id === activeRunId)
    ? activeRunId
    : runs?.[0]?.id ?? null;

  return (
    <div className="card flex h-full flex-col p-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-lg font-bold text-slate-900">
          <Database className="h-5 w-5 text-brand" /> Dataset History
        </h3>
        <button
          onClick={() => navigate("/dataset")}
          className="inline-flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700"
        >
          <Plus className="h-3.5 w-3.5" /> Generate
        </button>
      </div>

      {isLoading && <LoadingState label="Loading saved datasets…" />}

      {(isError || (!isLoading && (!runs || runs.length === 0))) && (
        <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 p-6 text-center">
          <Database className="mb-2 h-8 w-8 text-slate-300" />
          <p className="text-sm font-medium text-slate-500">
            {isError ? "History is unavailable (database offline)." : "No saved datasets yet."}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Generate a dataset and it will be stored here for instant reuse.
          </p>
        </div>
      )}

      {runs && runs.length > 0 && (
        <ul className="-mx-1 space-y-2 overflow-y-auto pr-1" style={{ maxHeight: 340 }}>
          {runs.map((run) => {
            const active = run.id === effectiveActive;
            return (
              <li key={run.id}>
                <div
                  className={`group flex items-center gap-3 rounded-xl border p-3 transition ${
                    active
                      ? "border-brand bg-blue-50/60 ring-1 ring-brand/30"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <button onClick={() => setActiveRunId(run.id)} className="flex flex-1 items-center gap-3 text-left">
                    <span
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                        active ? "bg-brand text-white" : "bg-slate-100 text-slate-400"
                      }`}
                    >
                      {active ? <CheckCircle2 className="h-4 w-4" /> : <Database className="h-4 w-4" />}
                    </span>
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold capitalize text-slate-800">
                        {run.business_type} · {formatInt(run.total_transactions)} txns
                      </div>
                      <div className="truncate text-xs text-slate-400">
                        {formatTimestamp(run.created_at)} · {run.fraud_percentage.toFixed(1)}% fraud ·{" "}
                        {formatInt(run.stored_count)} stored
                      </div>
                    </div>
                  </button>
                  <div className="flex items-center gap-2">
                    {active && (
                      <span className="rounded-full bg-brand/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand">
                        Active
                      </span>
                    )}
                    <button
                      onClick={() => {
                        if (run.id === activeRunId) setActiveRunId(null);
                        del.mutate(run.id);
                      }}
                      title="Delete dataset"
                      className="rounded-md p-1.5 text-slate-300 opacity-0 transition hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
