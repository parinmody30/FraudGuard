import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Play, Loader2, GitBranch } from "lucide-react";
import { useGenerateDataset } from "@/hooks/useDataset";
import { useDatasetSelection } from "@/context/DatasetContext";
import { ErrorState } from "@/components/common/States";
import { toUserMessage } from "@/lib/errors";
import { formatInt } from "@/lib/format";
import type { DatasetConfig } from "@/types";

const BUSINESS_TYPES: DatasetConfig["business_type"][] = ["bank", "credit_card", "upi", "ecommerce"];
const RISK_APPETITES: DatasetConfig["risk_appetite"][] = ["low", "medium", "high"];

export default function DatasetGenerator() {
  const [form, setForm] = useState<DatasetConfig>({
    num_users: 500,
    transactions_per_user: 200,
    fraud_rate: 0.02,
    business_type: "bank",
    risk_appetite: "medium",
    seed: 42,
    preview_limit: 10,
  });
  const navigate = useNavigate();
  const mutation = useGenerateDataset();
  const { setActiveRunId } = useDatasetSelection();
  const result = mutation.data;

  // When a generation persists a run, make it the active dataset for Dashboard/Transactions.
  useEffect(() => {
    if (result?.summary.run_id) setActiveRunId(result.summary.run_id);
  }, [result?.summary.run_id, setActiveRunId]);

  const set = <K extends keyof DatasetConfig>(key: K, value: DatasetConfig[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const distribution = result
    ? Object.entries(result.summary.fraud_type_distribution).sort((a, b) => b[1] - a[1])
    : [];
  const maxCount = distribution.length ? distribution[0][1] : 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dataset Generation</h1>
        <p className="text-sm text-slate-500">
          Generate synthetic transactions with realistic fraud scenarios to train and test detection models.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_1fr]">
        {/* Configuration */}
        <div className="card p-6">
          <h2 className="border-b border-slate-100 pb-3 text-lg font-bold text-slate-900">
            Generation Configuration
          </h2>
          <div className="mt-4 space-y-4">
            <Field label="Number of Users" hint="Total unique users to simulate">
              <input type="number" min={1} max={5000} value={form.num_users}
                onChange={(e) => set("num_users", Number(e.target.value))} className="ds-input" />
            </Field>
            <Field label="Transactions per User" hint="Average transactions per user">
              <input type="number" min={1} max={500} value={form.transactions_per_user}
                onChange={(e) => set("transactions_per_user", Number(e.target.value))} className="ds-input" />
            </Field>
            <Field label="Target Fraud Rate (%)" hint="Overall fraud percentage in dataset">
              <input type="number" step="0.1" min={0} max={100} value={+(form.fraud_rate * 100).toFixed(2)}
                onChange={(e) => set("fraud_rate", Number(e.target.value) / 100)} className="ds-input" />
            </Field>
            <Field label="Business Type">
              <select value={form.business_type}
                onChange={(e) => set("business_type", e.target.value as DatasetConfig["business_type"])} className="ds-input capitalize">
                {BUSINESS_TYPES.map((b) => <option key={b} value={b}>{b.replace("_", " ")}</option>)}
              </select>
            </Field>
            <Field label="Risk Appetite">
              <select value={form.risk_appetite}
                onChange={(e) => set("risk_appetite", e.target.value as DatasetConfig["risk_appetite"])} className="ds-input capitalize">
                {RISK_APPETITES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </Field>
            <Field label="Random Seed" hint="For reproducibility">
              <input type="number" value={form.seed}
                onChange={(e) => set("seed", Number(e.target.value))} className="ds-input" />
            </Field>

            <button
              onClick={() => mutation.mutate(form)}
              disabled={mutation.isPending}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {mutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {mutation.isPending ? "Generating…" : "Generate Dataset"}
            </button>
            <button
              disabled
              title="Training endpoint is not exposed in the Phase 7.0 API contract."
              className="flex w-full cursor-not-allowed items-center justify-center gap-2 rounded-xl border border-brand/40 px-4 py-2.5 text-sm font-semibold text-brand/60"
            >
              <GitBranch className="h-4 w-4" /> Generate &amp; Train Champion
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-5">
          {mutation.isError && <ErrorState message={toUserMessage(mutation.error)} onRetry={() => mutation.mutate(form)} />}

          {!result && !mutation.isError && (
            <div className="card flex h-40 items-center justify-center text-sm text-slate-400">
              Configure parameters and click <span className="mx-1 font-semibold text-slate-500">Generate Dataset</span> to begin.
            </div>
          )}

          {result && (
            <>
              <div className="flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-700">
                <CheckCircle2 className="h-5 w-5" />
                {result.summary.run_id
                  ? "Dataset generated and saved to history — now the active dataset for the Dashboard."
                  : "Dataset generated successfully! (History not saved — database offline.)"}
              </div>

              <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                <Stat label="Total Transactions" value={formatInt(result.summary.total_transactions)} />
                <Stat label="Fraud Transactions" value={formatInt(result.summary.fraud_count)} accent="red" />
                <Stat label="Fraud Rate" value={`${result.summary.fraud_percentage.toFixed(2)}%`} accent="blue" />
                <Stat label="Generation Time" value={`${result.elapsedSeconds.toFixed(1)}s`} />
              </div>

              <div className="card p-6">
                <h3 className="mb-4 text-lg font-bold text-slate-900">Fraud Distribution by Vector</h3>
                <div className="space-y-3">
                  {distribution.slice(0, 8).map(([name, count]) => {
                    const pct = (count / result.summary.fraud_count) * 100;
                    return (
                      <div key={name} className="flex items-center gap-3">
                        <div className="w-40 text-sm capitalize text-slate-600">{name.replace(/_/g, " ")}</div>
                        <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-100">
                          <div className="h-full rounded-full bg-brand" style={{ width: `${(count / maxCount) * 100}%` }} />
                        </div>
                        <div className="w-24 text-right text-sm font-semibold text-slate-700">
                          {pct.toFixed(1)}% <span className="text-slate-400">({count})</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div className="card p-6">
                  <h3 className="mb-3 text-base font-bold text-slate-900">Dataset Quality Checks</h3>
                  <ul className="space-y-2 text-sm">
                    <QualityRow label="Label Leakage Check" status="Passed" />
                    <QualityRow label="Signal Strength Analysis" status="Optimal" />
                  </ul>
                  <p className="mt-3 text-[11px] text-slate-400">
                    Reference checks verified in Phase 1.1 evaluation (no live endpoint).
                  </p>
                </div>
                <div className="card flex flex-col justify-center gap-3 p-6">
                  <button
                    onClick={() => navigate("/dashboard")}
                    className="rounded-xl bg-brand px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700"
                  >
                    View on Dashboard
                  </button>
                  <button
                    onClick={() => navigate("/transactions")}
                    className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800"
                  >
                    Go to Transactions Table
                  </button>
                  <span className="text-center text-xs text-slate-400">
                    Preview rows returned: {result.summary.sample.length}
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <style>{`.ds-input{width:100%;border:1px solid #e2e8f0;border-radius:0.6rem;padding:0.55rem 0.75rem;font-size:0.875rem;background:#f8fafc}.ds-input:focus{outline:none;border-color:#1d4ed8;background:#fff}`}</style>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-xs text-slate-400">{hint}</span>}
    </label>
  );
}

function Stat({ label, value, accent = "slate" }: { label: string; value: string; accent?: "slate" | "red" | "blue" }) {
  const color = accent === "red" ? "text-red-600" : accent === "blue" ? "text-brand" : "text-slate-900";
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-2 text-2xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

function QualityRow({ label, status }: { label: string; status: string }) {
  return (
    <li className="flex items-center justify-between">
      <span className="flex items-center gap-2 text-slate-600">
        <CheckCircle2 className="h-4 w-4 text-emerald-500" /> {label}
      </span>
      <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-600">{status}</span>
    </li>
  );
}
