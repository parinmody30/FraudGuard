// Settings — live system status, model decision thresholds, analyst preferences and account.
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Bot, CheckCircle2, Cpu, Database, ExternalLink, Server, Trash2, User, XCircle } from "lucide-react";
import { useSystemStatus } from "@/hooks/useReports";
import { useModelMetadata } from "@/hooks/useModel";
import { useAuth } from "@/context/AuthContext";
import { useSettings } from "@/lib/settings";
import { API_BASE_URL } from "@/services";
import { formatTimestamp } from "@/lib/format";

export default function Settings() {
  const status = useSystemStatus();
  const meta = useModelMetadata();
  const [settings, update] = useSettings();
  const { user } = useAuth();
  const qc = useQueryClient();
  const [cleared, setCleared] = useState(false);

  const s = status.data;
  const apiUp = Boolean(s) && !status.isError;
  const t = meta.data?.thresholds ?? {};
  const review = t.review ?? 0.4;
  const block = t.block ?? 0.8;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Settings</h1>
        <p className="text-sm text-slate-500">System status, decision policy and analyst preferences</p>
      </div>

      {/* System status */}
      <section className="card p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-900">System Status</h2>
          <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1 text-sm font-medium text-brand hover:underline">
            API docs <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatusTile icon={Server} label="Backend API" ok={apiUp}
            detail={apiUp ? `Online · v${s?.version}` : "Offline — start the backend on port 8000"} />
          <StatusTile icon={Cpu} label="Champion model" ok={apiUp && Boolean(s?.model_available)}
            detail={s?.model_available ? `${meta.data?.model_type ?? "Loaded"}` : "Model artifacts missing"} />
          <StatusTile icon={Database} label="Dataset history (PostgreSQL)" ok={apiUp && Boolean(s?.database)}
            detail={s?.database ? "Connected — runs are saved" : "Not connected — history disabled"} />
          <StatusTile icon={Bot} label="AI narratives (Groq)" ok={apiUp && Boolean(s?.groq_configured)} warnOnly
            detail={s?.groq_configured ? "API key configured" : "No key — using offline narrative engine"} />
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Decision policy */}
        <section className="card p-6">
          <h2 className="text-lg font-bold text-slate-900">Decision Policy</h2>
          <p className="mt-1 text-sm text-slate-500">
            How the model's fraud probability maps to an action. Thresholds are calibrated at training time
            {meta.data?.training_timestamp ? ` (${formatTimestamp(meta.data.training_timestamp)})` : ""}.
          </p>
          <div className="mt-5">
            <div className="flex h-3 overflow-hidden rounded-full">
              <div className="bg-emerald-400" style={{ width: `${review * 100}%` }} />
              <div className="bg-amber-400" style={{ width: `${(block - review) * 100}%` }} />
              <div className="bg-red-500" style={{ width: `${(1 - block) * 100}%` }} />
            </div>
            <div className="relative mt-1 h-4 text-xs text-slate-400">
              <span className="absolute left-0">0%</span>
              <span className="absolute -translate-x-1/2" style={{ left: `${review * 100}%` }}>{Math.round(review * 100)}%</span>
              <span className="absolute -translate-x-1/2" style={{ left: `${block * 100}%` }}>{Math.round(block * 100)}%</span>
              <span className="absolute right-0">100%</span>
            </div>
          </div>
          <dl className="mt-4 space-y-2 text-sm">
            <Policy tone="bg-emerald-400" label="ALLOW" text={`Probability below ${Math.round(review * 100)}% — transaction proceeds.`} />
            <Policy tone="bg-amber-400" label="REVIEW" text={`${Math.round(review * 100)}–${Math.round(block * 100)}% — queued for an analyst.`} />
            <Policy tone="bg-red-500" label="BLOCK" text={`${Math.round(block * 100)}% or higher — stopped and routed to fraud operations.`} />
          </dl>
        </section>

        {/* Preferences + account */}
        <section className="card p-6">
          <h2 className="text-lg font-bold text-slate-900">Analyst Preferences</h2>
          <label className="mt-4 flex cursor-pointer items-start justify-between gap-4">
            <span>
              <span className="block text-sm font-medium text-slate-800">Use AI-written investigation narratives</span>
              <span className="block text-xs text-slate-500">
                Uses Groq when a key is configured on the server; otherwise the deterministic offline engine is used.
              </span>
            </span>
            <input type="checkbox" className="peer sr-only" checked={settings.useAiNarratives}
              onChange={(e) => update({ useAiNarratives: e.target.checked })} />
            <span aria-hidden className={`mt-0.5 flex h-6 w-11 shrink-0 items-center rounded-full p-0.5 transition ${settings.useAiNarratives ? "bg-brand" : "bg-slate-300"}`}>
              <span className={`h-5 w-5 rounded-full bg-white shadow transition ${settings.useAiNarratives ? "translate-x-5" : ""}`} />
            </span>
          </label>

          <label className="mt-5 block">
            <span className="block text-sm font-medium text-slate-800">Analyst name on exported reports</span>
            <input value={settings.analystName} onChange={(e) => update({ analystName: e.target.value })}
              placeholder={user?.name ?? "Your name"}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-brand focus:outline-none" />
          </label>

          <h2 className="mt-6 text-lg font-bold text-slate-900">Account</h2>
          <div className="mt-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-50 text-brand"><User className="h-5 w-5" /></div>
            <div>
              <div className="text-sm font-semibold text-slate-800">{user?.name ?? "—"}</div>
              <div className="text-xs text-slate-500">{user?.email ?? ""}</div>
            </div>
          </div>
          <button
            onClick={() => {
              try {
                Object.keys(localStorage).filter((k) => k.startsWith("fraudlab_notes")).forEach((k) => localStorage.removeItem(k));
              } catch { /* storage unavailable */ }
              qc.clear();
              setCleared(true);
            }}
            className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50">
            <Trash2 className="h-4 w-4" /> Clear cached data & saved notes
          </button>
          {cleared && <p className="mt-2 text-xs text-emerald-600">Cleared. Data will reload from the server.</p>}
        </section>
      </div>
    </div>
  );
}

function StatusTile({ icon: Icon, label, ok, detail, warnOnly }: {
  icon: typeof Server; label: string; ok: boolean; detail: string; warnOnly?: boolean;
}) {
  const bad = warnOnly ? "text-amber-500" : "text-red-500";
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50/60 p-4">
      <div className="flex items-center justify-between">
        <Icon className="h-5 w-5 text-slate-400" />
        {ok ? <CheckCircle2 className="h-5 w-5 text-emerald-500" /> : <XCircle className={`h-5 w-5 ${bad}`} />}
      </div>
      <div className="mt-3 text-sm font-semibold text-slate-800">{label}</div>
      <div className="text-xs text-slate-500">{detail}</div>
    </div>
  );
}

function Policy({ tone, label, text }: { tone: string; label: string; text: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className={`mt-1 h-3 w-3 shrink-0 rounded-full ${tone}`} />
      <div><dt className="inline font-semibold text-slate-800">{label}</dt> <dd className="inline text-slate-600">— {text}</dd></div>
    </div>
  );
}
