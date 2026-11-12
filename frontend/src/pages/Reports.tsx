// Reports — model leaderboard (offline benchmark), champion metrics, live detection performance
// on the active dataset, and the CTGAN hybrid-data experiment outcome.
import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Download, FlaskConical, Printer, ShieldCheck, Target, Trophy } from "lucide-react";
import { useReportsSummary } from "@/hooks/useReports";
import { useModelMetadata } from "@/hooks/useModel";
import { useActiveDataset } from "@/hooks/useActiveDataset";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { toUserMessage } from "@/lib/errors";
import { tierOf } from "@/lib/risk";
import { formatInt, formatTimestamp } from "@/lib/format";
import type { LeaderboardRow, ScoredTransaction } from "@/types";

const MODEL_LABELS: Record<string, string> = {
  xgboost: "XGBoost",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  logistic_regression: "Logistic Regression",
  lstm: "LSTM (sequential)",
  isolation_forest: "Isolation Forest",
};
const modelLabel = (m: string) => MODEL_LABELS[m] ?? m;
const fmt = (n: number | null | undefined, d = 3) => (n == null ? "—" : n.toFixed(d));
const pct = (n: number | null | undefined, d = 1) => (n == null ? "—" : `${(n * 100).toFixed(d)}%`);

export default function Reports() {
  const summary = useReportsSummary();
  const meta = useModelMetadata();
  const ds = useActiveDataset();

  const simulator = useMemo(
    () => (summary.data?.leaderboard ?? []).filter((r) => r.dataset === "simulator").sort((a, b) => a.rank - b.rank),
    [summary.data],
  );
  const champion = meta.data?.champion ?? simulator[0]?.model;
  const m = meta.data?.metrics ?? {};

  const exportCsv = () => {
    const rows = summary.data?.leaderboard ?? [];
    if (!rows.length) return;
    const cols = Object.keys(rows[0]) as (keyof LeaderboardRow)[];
    const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => r[c] ?? "").join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "fraudlab-model-leaderboard.csv" });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Reports</h1>
          <p className="text-sm text-slate-500">Model benchmark, detection performance and data-strategy findings</p>
        </div>
        <div className="flex gap-2 print:hidden">
          <button onClick={exportCsv} disabled={!summary.data?.leaderboard.length}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50">
            <Download className="h-4 w-4" /> Leaderboard CSV
          </button>
          <button onClick={() => window.print()}
            className="inline-flex items-center gap-2 rounded-lg bg-brand px-3 py-2 text-sm font-semibold text-white hover:bg-brand-700">
            <Printer className="h-4 w-4" /> Print / PDF
          </button>
        </div>
      </div>

      {/* Champion KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard icon={Trophy} tag="Champion" title="Production model" value={champion ? modelLabel(champion) : "—"}
          sub={meta.data?.training_timestamp ? `Trained ${formatTimestamp(meta.data.training_timestamp)}` : undefined} highlight />
        <MetricCard icon={Target} tag="Metric" title="PR-AUC" value={fmt(m.pr_auc)} sub="Best metric for rare-event fraud" accent="blue" />
        <MetricCard icon={Activity} tag="Metric" title="ROC-AUC" value={fmt(m.roc_auc)} sub={`Precision ${fmt(m.precision)} · Recall ${fmt(m.recall)}`} accent="emerald" />
        <MetricCard icon={ShieldCheck} tag="Metric" title="Recall on hard fraud" value={pct(m.recall_hard)}
          sub="Subtle, sequence-only scenarios" accent="red" />
      </div>

      {/* Leaderboard */}
      <section className="card p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-bold text-slate-900">Model Leaderboard</h2>
          <span className="text-xs text-slate-400">
            Held-out test set · {summary.data?.benchmark_config ? describeConfig(summary.data.benchmark_config) : "simulator data"}
          </span>
        </div>
        {summary.isLoading && <div className="mt-4"><LoadingState label="Loading benchmark results…" /></div>}
        {summary.isError && <div className="mt-4"><ErrorState message={toUserMessage(summary.error)} onRetry={() => summary.refetch()} /></div>}
        {summary.data && simulator.length === 0 && <div className="mt-4"><EmptyState message="No benchmark results found in reports/leaderboard.csv." /></div>}
        {simulator.length > 0 && (
          <div className="mt-4 grid grid-cols-1 gap-6 xl:grid-cols-5">
            <div className="overflow-x-auto xl:col-span-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 text-left text-xs uppercase tracking-wide text-slate-400">
                    <th className="py-2 pr-3">#</th><th className="py-2 pr-3">Model</th>
                    <th className="py-2 pr-3 text-right">PR-AUC</th><th className="py-2 pr-3 text-right">ROC-AUC</th>
                    <th className="py-2 pr-3 text-right">Precision</th><th className="py-2 pr-3 text-right">Recall</th>
                    <th className="py-2 pr-3 text-right">F1</th><th className="py-2 text-right">Hard recall</th>
                  </tr>
                </thead>
                <tbody>
                  {simulator.map((r) => {
                    const isChamp = r.model === champion;
                    return (
                      <tr key={r.model} className={`border-b border-slate-50 ${isChamp ? "bg-blue-50/60" : ""}`}>
                        <td className="py-2.5 pr-3 font-mono text-slate-400">{r.rank}</td>
                        <td className="py-2.5 pr-3 font-medium text-slate-800">
                          {modelLabel(r.model)}
                          {isChamp && <span className="ml-2 rounded-full bg-brand px-2 py-0.5 text-[10px] font-semibold text-white">CHAMPION</span>}
                        </td>
                        <td className="py-2.5 pr-3 text-right font-mono font-semibold text-slate-900">{fmt(r.pr_auc)}</td>
                        <td className="py-2.5 pr-3 text-right font-mono">{fmt(r.roc_auc)}</td>
                        <td className="py-2.5 pr-3 text-right font-mono">{fmt(r.precision)}</td>
                        <td className="py-2.5 pr-3 text-right font-mono">{fmt(r.recall)}</td>
                        <td className="py-2.5 pr-3 text-right font-mono">{fmt(r.f1)}</td>
                        <td className="py-2.5 text-right font-mono">{fmt(r.recall_hard)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-slate-500">
                Models are ranked by PR-AUC, which reflects performance on the rare fraud class better than accuracy or ROC-AUC.
                Isolation Forest (unsupervised) is included as a no-labels baseline.
              </p>
            </div>
            <div className="h-64 xl:col-span-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={simulator.map((r) => ({ name: modelLabel(r.model), value: r.pr_auc ?? 0, champ: r.model === champion }))}
                  layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eef2f7" />
                  <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(v) => [Number(v).toFixed(3), "PR-AUC"]} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={22}>
                    {simulator.map((r) => <Cell key={r.model} fill={r.model === champion ? "#1d4ed8" : "#cbd5e1"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </section>

      {/* Live detection on active dataset */}
      <DetectionReport rows={ds.data} isLoading={ds.isLoading} label={ds.activeRun?.label} />

      {/* CTGAN experiment */}
      {summary.data?.ctgan_decision && (
        <section className="card p-6">
          <h2 className="flex items-center gap-2 text-lg font-bold text-slate-900">
            <FlaskConical className="h-5 w-5 text-brand" /> Experiment: CTGAN Synthetic Augmentation
          </h2>
          <p className="mt-2 text-sm text-slate-600">
            We tested whether training on a hybrid dataset (simulator data plus CTGAN-generated rows, matched to the same
            {" "}{summary.data.hybrid_composition?.realized_fraud_rate != null ? pct(Number(summary.data.hybrid_composition.realized_fraud_rate)) : ""} fraud rate)
            improves detection compared with simulator data alone.
          </p>
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-red-100 bg-red-50 p-4 md:col-span-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-red-500">Decision</div>
              <div className="mt-1 text-sm font-bold text-red-700">{summary.data.ctgan_decision.recommendation}</div>
            </div>
            <Stat label="PR-AUC change (median across models)" stats={summary.data.ctgan_decision.pr_auc_delta_stats} />
            <Stat label="Hard-fraud recall change (median)" stats={summary.data.ctgan_decision.recall_hard_delta_stats} />
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Takeaway: the behavior simulator produces more learnable, realistic fraud patterns than the GAN, so the
            production model is trained on simulator data only.
          </p>
        </section>
      )}
    </div>
  );
}

function Stat({ label, stats }: { label: string; stats?: { n: number; improved: number; degraded: number; median_delta: number } }) {
  if (!stats) return null;
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${stats.median_delta < 0 ? "text-red-600" : "text-emerald-600"}`}>
        {stats.median_delta > 0 ? "+" : ""}{stats.median_delta.toFixed(3)}
      </div>
      <div className="text-xs text-slate-500">{stats.improved} of {stats.n} models improved · {stats.degraded} degraded</div>
    </div>
  );
}

function describeConfig(cfg: Record<string, unknown>): string {
  const parts = [
    cfg.num_users != null && `${cfg.num_users} users`,
    cfg.transactions_per_user != null && `${cfg.transactions_per_user} txns/user`,
    cfg.fraud_rate != null && `${(Number(cfg.fraud_rate) * 100).toFixed(0)}% fraud`,
    cfg.business_type != null && String(cfg.business_type),
  ].filter(Boolean);
  return parts.join(" · ");
}

function DetectionReport({ rows, isLoading, label }: { rows?: ScoredTransaction[]; isLoading: boolean; label?: string }) {
  const stats = useMemo(() => {
    const scored = (rows ?? []).filter((r) => r.prediction && r.is_fraud != null);
    let tp = 0, fn = 0, fp = 0, tn = 0;
    const tiers = { easy: [0, 0], medium: [0, 0], hard: [0, 0] } as Record<"easy" | "medium" | "hard", [number, number]>;
    const types = new Map<string, [number, number]>();
    for (const r of scored) {
      const flagged = r.prediction!.decision !== "ALLOW";
      if (r.is_fraud) {
        if (flagged) tp++; else fn++;
        const tier = tierOf(r.fraud_type);
        if (tier !== "legit") { tiers[tier][1]++; if (flagged) tiers[tier][0]++; }
        const key = (r.fraud_type ?? "unknown").replace(/_/g, " ");
        const t = types.get(key) ?? [0, 0];
        t[1]++; if (flagged) t[0]++;
        types.set(key, t);
      } else if (flagged) fp++; else tn++;
    }
    return { n: scored.length, tp, fn, fp, tn, tiers, types: [...types.entries()].sort((a, b) => b[1][1] - a[1][1]) };
  }, [rows]);

  return (
    <section className="card p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-lg font-bold text-slate-900">Live Detection Performance</h2>
        <span className="text-xs text-slate-400">{label ? `Active dataset · ${label}` : "Active dataset"}{!isLoading && ` · ${formatInt(stats.n)} scored transactions`}</span>
      </div>
      {isLoading && <div className="mt-4"><LoadingState label="Scoring the active dataset…" /></div>}
      {!isLoading && stats.n === 0 && <div className="mt-4"><EmptyState message="Generate a dataset to see live detection performance." /></div>}
      {!isLoading && stats.n > 0 && (
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Outcome (flagged = REVIEW or BLOCK)</h3>
            <div className="mt-2 grid grid-cols-2 gap-2 text-center text-sm">
              <Cell2 label="Fraud caught" value={stats.tp} tone="emerald" />
              <Cell2 label="Fraud missed" value={stats.fn} tone="red" />
              <Cell2 label="False alarms" value={stats.fp} tone="amber" />
              <Cell2 label="Legit allowed" value={stats.tn} tone="slate" />
            </div>
            <div className="mt-3 flex justify-between text-sm">
              <span className="text-slate-500">Detection rate</span>
              <span className="font-semibold text-slate-900">{pct(stats.tp / Math.max(1, stats.tp + stats.fn))}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">False-alarm rate</span>
              <span className="font-semibold text-slate-900">{pct(stats.fp / Math.max(1, stats.fp + stats.tn), 2)}</span>
            </div>
          </div>
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Detection by difficulty</h3>
            <div className="mt-3 space-y-3">
              {(["easy", "medium", "hard"] as const).map((t) => {
                const [c, total] = stats.tiers[t];
                const rate = total ? c / total : 0;
                const color = { easy: "bg-emerald-500", medium: "bg-amber-500", hard: "bg-red-500" }[t];
                return (
                  <div key={t}>
                    <div className="flex justify-between text-sm">
                      <span className="capitalize text-slate-600">{t}</span>
                      <span className="font-medium text-slate-800">{c}/{total} · {total ? pct(rate, 0) : "—"}</span>
                    </div>
                    <div className="mt-1 h-2 rounded-full bg-slate-100">
                      <div className={`h-2 rounded-full ${color}`} style={{ width: `${rate * 100}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Detection by fraud type</h3>
            <div className="mt-2 max-h-56 overflow-y-auto">
              <table className="w-full text-sm">
                <tbody>
                  {stats.types.map(([name, [c, total]]) => (
                    <tr key={name} className="border-b border-slate-50">
                      <td className="py-1.5 capitalize text-slate-600">{name}</td>
                      <td className="py-1.5 text-right font-mono text-slate-800">{c}/{total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function Cell2({ label, value, tone }: { label: string; value: number; tone: "emerald" | "red" | "amber" | "slate" }) {
  const cls = {
    emerald: "bg-emerald-50 text-emerald-700",
    red: "bg-red-50 text-red-700",
    amber: "bg-amber-50 text-amber-700",
    slate: "bg-slate-50 text-slate-700",
  }[tone];
  return (
    <div className={`rounded-lg p-3 ${cls}`}>
      <div className="text-xl font-bold">{formatInt(value)}</div>
      <div className="text-xs">{label}</div>
    </div>
  );
}
