import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight, Download, Search, SlidersHorizontal } from "lucide-react";
import { DEFAULT_TX_PARAMS, useScoredTransactions } from "@/hooks/useTransactions";
import { DecisionBadge } from "@/components/common/Badge";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { toUserMessage } from "@/lib/errors";
import { formatCurrencyINR, formatScore, formatTimestamp, truncate } from "@/lib/format";
import { scoreColor, tierOf } from "@/lib/risk";
import type { ScoredTransaction } from "@/types";

const PAGE_SIZE = 10;
const FRAUD_TYPES = [
  "account_takeover", "large_spend", "geo_anomaly", "night_time", "new_device",
  "card_testing", "velocity", "merchant", "beneficiary",
  "synthetic_identity", "money_mule", "gradual_escalation", "cross_channel", "multi_stage",
];

export default function Transactions() {
  const { data, isLoading, isError, error, refetch, isFetching } = useScoredTransactions(DEFAULT_TX_PARAMS);
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [fraudFilter, setFraudFilter] = useState("all");
  const [difficulty, setDifficulty] = useState("all");
  const [fraudType, setFraudType] = useState("all");
  const [minAmt, setMinAmt] = useState("");
  const [maxAmt, setMaxAmt] = useState("");
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const rows = data ?? [];
    return rows.filter((t) => {
      const q = search.trim().toLowerCase();
      if (q && !`${t.transaction_id} ${t.user_id} ${t.merchant_name}`.toLowerCase().includes(q)) return false;
      const decision = t.prediction?.decision;
      if (fraudFilter === "fraud" && decision === "ALLOW") return false;
      if (fraudFilter === "legit" && decision !== "ALLOW") return false;
      if (difficulty !== "all" && tierOf(t.fraud_type) !== difficulty) return false;
      if (fraudType !== "all" && t.fraud_type !== fraudType) return false;
      if (minAmt && t.amount < Number(minAmt)) return false;
      if (maxAmt && t.amount > Number(maxAmt)) return false;
      return true;
    });
  }, [data, search, fraudFilter, difficulty, fraudType, minAmt, maxAmt]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const pageRows = filtered.slice(current * PAGE_SIZE, current * PAGE_SIZE + PAGE_SIZE);

  const exportCsv = () => {
    const header = "transaction_id,timestamp,user_id,amount,merchant,channel,device,fraud_score,decision";
    const lines = filtered.map((t) =>
      [t.transaction_id, t.timestamp, t.user_id, t.amount, t.merchant_name, t.channel, t.device_type,
        t.prediction?.fraud_probability ?? "", t.prediction?.decision ?? ""].join(",")
    );
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "fraudlab_transactions.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Transactions Table</h1>
          <p className="text-sm text-slate-500">Browse and search through scored transactions.</p>
        </div>
        <button onClick={exportCsv} disabled={!filtered.length}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50">
          <Download className="h-4 w-4" /> Export
        </button>
      </div>

      {/* Filters */}
      <div className="card p-5">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-6">
          <label className="md:col-span-2">
            <span className="text-xs font-semibold uppercase text-slate-400">Search</span>
            <div className="relative mt-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input value={search} onChange={(e) => { setSearch(e.target.value); setPage(0); }}
                placeholder="Transaction ID, User ID…" className="tx-input pl-9" />
            </div>
          </label>
          <Select label="Fraud" value={fraudFilter} onChange={(v) => { setFraudFilter(v); setPage(0); }}
            options={[["all", "All"], ["fraud", "Flagged"], ["legit", "Allowed"]]} />
          <Select label="Difficulty" value={difficulty} onChange={(v) => { setDifficulty(v); setPage(0); }}
            options={[["all", "All"], ["easy", "Easy"], ["medium", "Medium"], ["hard", "Hard"]]} />
          <Select label="Fraud Type" value={fraudType} onChange={(v) => { setFraudType(v); setPage(0); }}
            options={[["all", "All"], ...FRAUD_TYPES.map((t) => [t, t] as [string, string])]} />
          <div className="grid grid-cols-2 gap-2">
            <label><span className="text-xs font-semibold uppercase text-slate-400">Min Amt</span>
              <input value={minAmt} onChange={(e) => setMinAmt(e.target.value)} placeholder="Min" className="tx-input mt-1" /></label>
            <label><span className="text-xs font-semibold uppercase text-slate-400">Max Amt</span>
              <input value={maxAmt} onChange={(e) => setMaxAmt(e.target.value)} placeholder="Max" className="tx-input mt-1" /></label>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
          <SlidersHorizontal className="h-3.5 w-3.5" />
          {isFetching ? "Scoring transactions…" : `${filtered.length} of ${data?.length ?? 0} transactions match`}
        </div>
      </div>

      {isLoading && <LoadingState label="Generating & scoring transactions…" />}
      {isError && <ErrorState message={toUserMessage(error)} onRetry={() => refetch()} />}
      {data && filtered.length === 0 && !isLoading && <EmptyState message="No transactions match the current filters." />}

      {data && filtered.length > 0 && (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-400">
                <tr>
                  {["Transaction ID", "Timestamp", "User ID", "Amount (₹)", "Merchant", "Channel", "Device", "Fraud Score", "Decision", ""].map((h) => (
                    <th key={h} className="px-4 py-3 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((t) => (
                  <Row key={t.transaction_id} t={t} onInvestigate={() => navigate("/investigations", { state: { transaction: t } })} />
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 text-sm text-slate-500">
            <span>
              Showing {current * PAGE_SIZE + 1}–{Math.min((current + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
            </span>
            <div className="flex items-center gap-1">
              <button disabled={current === 0} onClick={() => setPage(current - 1)}
                className="rounded-md border border-slate-200 p-1.5 disabled:opacity-40"><ChevronLeft className="h-4 w-4" /></button>
              <span className="px-2 font-medium">{current + 1} / {pageCount}</span>
              <button disabled={current >= pageCount - 1} onClick={() => setPage(current + 1)}
                className="rounded-md border border-slate-200 p-1.5 disabled:opacity-40"><ChevronRight className="h-4 w-4" /></button>
            </div>
          </div>
        </div>
      )}

      <style>{`.tx-input{width:100%;border:1px solid #e2e8f0;border-radius:0.6rem;padding:0.5rem 0.7rem;font-size:0.875rem;background:#fff}.tx-input:focus{outline:none;border-color:#1d4ed8}`}</style>
    </div>
  );
}

function Row({ t, onInvestigate }: { t: ScoredTransaction; onInvestigate: () => void }) {
  const p = t.prediction;
  return (
    <tr className="border-t border-slate-100 hover:bg-slate-50/60">
      <td className="px-4 py-3 font-mono text-brand">{t.transaction_id}</td>
      <td className="px-4 py-3 text-slate-500">{formatTimestamp(t.timestamp)}</td>
      <td className="px-4 py-3 font-mono text-slate-600">{t.user_id}</td>
      <td className="px-4 py-3 font-medium">{formatCurrencyINR(t.amount)}</td>
      <td className="px-4 py-3 text-slate-600">{truncate(t.merchant_name, 16)}</td>
      <td className="px-4 py-3 uppercase text-slate-500">{t.channel}</td>
      <td className="px-4 py-3 text-slate-500">{t.device_type}</td>
      <td className={`px-4 py-3 font-mono font-semibold ${p ? scoreColor(p.fraud_probability) : "text-slate-300"}`}>
        {p ? formatScore(p.fraud_probability, 2) : "—"}
      </td>
      <td className="px-4 py-3">{p ? <DecisionBadge decision={p.decision} /> : <span className="text-slate-300">—</span>}</td>
      <td className="px-4 py-3">
        <button onClick={onInvestigate} className="rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-brand hover:bg-blue-50">
          Investigate
        </button>
      </td>
    </tr>
  );
}

function Select({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void; options: [string, string][];
}) {
  return (
    <label>
      <span className="text-xs font-semibold uppercase text-slate-400">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="tx-input mt-1 capitalize">
        {options.map(([v, l]) => <option key={v} value={v}>{l.replace(/_/g, " ")}</option>)}
      </select>
    </label>
  );
}
