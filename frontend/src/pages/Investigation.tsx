import { useMemo, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import {
  ArrowLeft, ArrowDownRight, ArrowUpRight, Ban, Bot, CheckCircle2, Eye, Share2, Download, AlertTriangle,
} from "lucide-react";
import { useAnalyzeTransaction } from "@/hooks/useInvestigation";
import { useActiveDataset } from "@/hooks/useActiveDataset";
import { ConfidenceBadge, DecisionBadge } from "@/components/common/Badge";
import { FraudGauge } from "@/components/investigation/FraudGauge";
import { ShapWaterfall } from "@/components/investigation/ShapWaterfall";
import { RiskFactorRadar } from "@/components/investigation/RiskFactorRadar";
import { SpendComparisonChart } from "@/components/investigation/SpendComparisonChart";
import { ErrorState, LoadingState } from "@/components/common/States";
import { toUserMessage } from "@/lib/errors";
import { formatCurrencyINR, formatInt, formatTimestamp } from "@/lib/format";
import type { TransactionIn } from "@/types";

export default function Investigation() {
  const location = useLocation();
  const passed = (location.state as { transaction?: TransactionIn } | null)?.transaction ?? null;

  // Fallback: if opened directly (e.g. from the sidebar), use the active dataset and open its
  // highest-risk transaction, so the page always lands on a meaningful case.
  const sample = useActiveDataset();
  const transaction = useMemo<TransactionIn | null>(() => {
    if (passed) return passed;
    const rows = sample.data ?? [];
    if (rows.length === 0) return null;
    const score = (r: (typeof rows)[number]) => r.prediction?.fraud_probability ?? (r.is_fraud ? 0.5 : 0);
    const { prediction: _p, ...best } = rows.reduce((a, b) => (score(b) > score(a) ? b : a));
    return best;
  }, [passed, sample.data]);

  const { data, isLoading, isError, error, refetch, isFetching } = useAnalyzeTransaction(transaction, true);

  if (!passed && sample.isLoading) return <LoadingState label="Loading a transaction to investigate…" />;
  if (!transaction) return <ErrorState message="No transaction available to investigate." onRetry={() => sample.refetch()} />;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <button onClick={() => history.back()} className="mb-1 inline-flex items-center gap-1 text-sm font-medium text-brand">
            <ArrowLeft className="h-4 w-4" /> Back to Transactions
          </button>
          <h1 className="flex items-center gap-3 text-2xl font-bold text-slate-900">
            Transaction ID: {transaction.transaction_id}
            {data && <RiskTag prob={data.result.prediction.fraud_probability} />}
          </h1>
          {data && <p className="text-sm text-slate-500">Agent mode: {data.mode.toUpperCase()}</p>}
        </div>
        <div className="flex gap-2">
          <HeaderBtn icon={Download} label="Export Report" />
          <HeaderBtn icon={Share2} label="Share" />
        </div>
      </div>

      {isLoading && <LoadingState label="Analyzing (score · SHAP · agent)…" />}
      {isError && <ErrorState message={toUserMessage(error)} onRetry={() => refetch()} />}

      {data && (
        <div className="space-y-5">
        {/* Analytics band — prediction gauge + two risk graphs */}
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="card flex flex-col p-6 text-center">
            <h2 className="text-lg font-bold text-slate-900">Fraud Prediction</h2>
            <FraudGauge prob={data.result.prediction.fraud_probability} />
            <div className="mt-auto flex items-center justify-center gap-6 pt-2">
              <div className="flex flex-col items-center gap-1.5">
                <span className="text-xs uppercase tracking-wide text-slate-400">Decision</span>
                <DecisionBadge decision={data.result.prediction.decision} />
              </div>
              <div className="flex flex-col items-center gap-1.5">
                <span className="text-xs uppercase tracking-wide text-slate-400">Confidence</span>
                <ConfidenceBadge confidence={data.result.prediction.confidence} />
              </div>
            </div>
          </div>

          <div className="card p-6">
            <h2 className="text-lg font-bold text-slate-900">Risk Factor Fingerprint</h2>
            <p className="mb-1 text-xs text-slate-400">Normalized risk across key dimensions (0–100)</p>
            <RiskFactorRadar transaction={transaction} />
          </div>

          <div className="card p-6">
            <h2 className="text-lg font-bold text-slate-900">Spend vs. User Behavior</h2>
            <p className="mb-1 text-xs text-slate-400">Transaction amount against this user's norms</p>
            <SpendComparisonChart
              amount={transaction.amount}
              avg={transaction.avg_user_spend}
              median={transaction.median_user_spend}
            />
          </div>
        </div>

        {/* Detail · SHAP · AI grid */}
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
          {/* Column 1 — Transaction Details */}
          <section className="card p-6">
            <h2 className="border-b border-slate-100 pb-3 text-lg font-bold text-slate-900">Transaction Details</h2>
            <div className="mt-3 space-y-4 text-sm">
              <DetailGroup title="Overview">
                <Detail k="User ID" v={transaction.user_id} mono />
                <Detail k="Timestamp" v={formatTimestamp(transaction.timestamp)} />
                <Detail k="Amount (₹)" v={formatCurrencyINR(transaction.amount)} strong />
                <Detail k="Channel" v={transaction.channel} />
              </DetailGroup>
              <DetailGroup title="Merchant">
                <Detail k="Merchant" v={transaction.merchant_name} />
                <Detail k="Category" v={transaction.merchant_category} />
              </DetailGroup>
              <DetailGroup title="Device & Location">
                <Detail k="Device" v={transaction.device_type} warn={transaction.device_risk_score >= 0.7} />
                <Detail k="Location" v={`${transaction.city}, ${transaction.state}`} />
              </DetailGroup>
              <DetailGroup title="Spending Behavior">
                <Detail k="User Avg. Spend" v={formatCurrencyINR(transaction.avg_user_spend)} />
                <Detail k="Velocity (1h)" v={`${formatInt(transaction.velocity_1h)} txns`} />
                <Detail k="Velocity (24h)" v={`${formatInt(transaction.velocity_24h)} txns`} />
              </DetailGroup>
            </div>
          </section>

          {/* Column 2 — SHAP */}
          <section className="card p-6">
            <h2 className="mb-4 text-lg font-bold text-slate-900">SHAP Explanation</h2>
            <ShapWaterfall contributors={data.result.explanation.contributors} />
            <p className="mt-3 text-center text-xs text-slate-400">Impact on Fraud Probability (log-odds)</p>
          </section>

          {/* Column 3 — AI Analysis + Actions */}
          <section className="space-y-5">
            <div className="card border-blue-100 bg-blue-50/40 p-6">
              <h2 className="flex items-center gap-2 text-lg font-bold text-slate-900">
                <Bot className="h-5 w-5 text-brand" /> AI Investigation Analysis
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-700">{data.result.investigation.summary}</p>

              <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">Signals Detected</h3>
              <ul className="mt-2 space-y-2 text-sm text-slate-700">
                {data.result.investigation.signals_detected.map((s, i) => {
                  const raises = /increas/i.test(s);
                  const lowers = /reduc|decreas/i.test(s);
                  const Icon = raises ? ArrowUpRight : lowers ? ArrowDownRight : Bot;
                  const tone = raises ? "text-red-500 bg-red-50" : lowers ? "text-emerald-500 bg-emerald-50" : "text-slate-400 bg-slate-100";
                  return (
                    <li key={i} className="flex items-start gap-2.5">
                      <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${tone}`}>
                        <Icon className="h-3.5 w-3.5" />
                      </span>
                      <span>{s}</span>
                    </li>
                  );
                })}
              </ul>

              <div className="mt-4 rounded-lg bg-white p-3 text-sm shadow-sm">
                <div className="flex justify-between"><span className="text-slate-400">Recommendation:</span>
                  <span className="font-semibold text-slate-800">{data.result.investigation.recommendation}</span></div>
                <div className="mt-1 flex justify-between"><span className="text-slate-400">Confidence:</span>
                  <span className="font-semibold text-slate-800">{data.result.prediction.confidence}</span></div>
              </div>
              <p className="mt-3 text-xs italic text-slate-500">{data.result.investigation.confidence_explanation}</p>

              <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">Next Steps</h3>
              <ul className="mt-2 space-y-2 text-sm text-slate-700">
                {data.result.investigation.next_steps.map((s, i) => (
                  <li key={i} className="flex items-start gap-2.5">
                    <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-50 text-xs font-bold text-brand">
                      {i + 1}
                    </span>
                    <span>{s}</span>
                  </li>
                ))}
              </ul>
            </div>

            <ActionsPanel decision={data.result.prediction.decision} />
          </section>
        </div>
        </div>
      )}
      {isFetching && !isLoading && <p className="text-xs text-slate-400">Refreshing…</p>}
    </div>
  );
}

function ActionsPanel({ decision }: { decision: string }) {
  const [action, setAction] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [saved, setSaved] = useState(false);
  return (
    <div className="card p-6">
      <h2 className="text-lg font-bold text-slate-900">Actions</h2>
      <div className="mt-3 space-y-2">
        <ActionBtn label="ALLOW" icon={CheckCircle2} active={action === "ALLOW"} suggested={decision === "ALLOW"}
          tone="emerald" onClick={() => setAction("ALLOW")} />
        <ActionBtn label="REVIEW" icon={Eye} active={action === "REVIEW"} suggested={decision === "REVIEW"}
          tone="amber" onClick={() => setAction("REVIEW")} />
        <ActionBtn label="BLOCK" icon={Ban} active={action === "BLOCK"} suggested={decision === "BLOCK"}
          tone="red" onClick={() => setAction("BLOCK")} />
      </div>
      <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">Add Notes</h3>
      <textarea value={notes} onChange={(e) => { setNotes(e.target.value); setSaved(false); }}
        placeholder="Enter notes about this transaction analysis…"
        className="mt-1 h-24 w-full resize-none rounded-lg border border-slate-200 p-2 text-sm focus:border-brand focus:outline-none" />
      <button onClick={() => setSaved(true)}
        className="mt-2 w-full rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
        Save Notes
      </button>
      {saved && <p className="mt-2 text-xs text-emerald-600">Saved locally (no persistence endpoint in the contract).</p>}
      {action && <p className="mt-1 text-xs text-slate-500">Analyst action “{action}” recorded locally.</p>}
    </div>
  );
}

function ActionBtn({ label, icon: Icon, active, suggested, tone, onClick }: {
  label: string; icon: typeof Ban; active: boolean; suggested: boolean;
  tone: "emerald" | "amber" | "red"; onClick: () => void;
}) {
  const activeRing = { emerald: "ring-2 ring-emerald-400", amber: "ring-2 ring-amber-400", red: "ring-2 ring-red-400" }[tone];
  const ring = active ? activeRing : "ring-1 ring-slate-200";
  const text = { emerald: "text-emerald-600", amber: "text-amber-600", red: "text-red-600" }[tone];
  return (
    <button onClick={onClick} className={`flex w-full items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 text-sm font-semibold ${text} ${ring} hover:bg-slate-50`}>
      <Icon className="h-4 w-4" /> {label}
      {suggested && <span className="ml-1 rounded-full bg-slate-100 px-1.5 text-[10px] font-medium text-slate-500">suggested</span>}
    </button>
  );
}

function DetailGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      <dl className="space-y-1">{children}</dl>
    </div>
  );
}

function Detail({ k, v, mono, strong, warn }: { k: string; v: string; mono?: boolean; strong?: boolean; warn?: boolean }) {
  return (
    <div className={`flex items-center justify-between border-b border-slate-50 py-2 ${warn ? "rounded-md bg-red-50 px-2" : ""}`}>
      <dt className="text-slate-500">{warn && <AlertTriangle className="mr-1 inline h-3.5 w-3.5 text-red-500" />}{k}</dt>
      <dd className={`${mono ? "font-mono" : ""} ${strong ? "font-bold" : "font-medium"} ${warn ? "text-red-600" : "text-slate-800"}`}>{v}</dd>
    </div>
  );
}

function RiskTag({ prob }: { prob: number }) {
  if (prob < 0.4) return null;
  const high = prob >= 0.8;
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold ${high ? "bg-red-50 text-red-600" : "bg-amber-50 text-amber-600"}`}>
      <AlertTriangle className="h-3.5 w-3.5" /> {high ? "HIGH RISK" : "ELEVATED"}
    </span>
  );
}

function HeaderBtn({ icon: Icon, label }: { icon: typeof Share2; label: string }) {
  return (
    <button className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50">
      <Icon className="h-4 w-4" /> {label}
    </button>
  );
}
