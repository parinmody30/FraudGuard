import type { ReactNode } from "react";
import { AlertTriangle, X } from "lucide-react";
import { ConfidenceBadge, DecisionBadge } from "@/components/common/Badge";
import { formatCurrencyINR, formatInt, formatScore, formatTimestamp } from "@/lib/format";
import { scoreColor } from "@/lib/risk";
import type { ScoredTransaction } from "@/types";

export function TransactionDetailModal({
  transaction,
  onClose,
}: {
  transaction: ScoredTransaction;
  onClose: () => void;
}) {
  const p = transaction.prediction;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
          <div>
            <h2 className="font-mono text-lg font-bold text-slate-900">{transaction.transaction_id}</h2>
            <p className="text-xs text-slate-400">{formatTimestamp(transaction.timestamp)}</p>
          </div>
          <button onClick={onClose} className="rounded-lg p-2 text-slate-400 hover:bg-slate-100">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-5 p-6">
          {p && (
            <div className="flex flex-wrap items-center gap-6 rounded-xl bg-slate-50 p-4">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-slate-400">Fraud Score</div>
                <div className={`text-2xl font-bold ${scoreColor(p.fraud_probability)}`}>
                  {formatScore(p.fraud_probability, 2)}
                </div>
              </div>
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">Decision</div>
                <DecisionBadge decision={p.decision} />
              </div>
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">Confidence</div>
                <ConfidenceBadge confidence={p.confidence} />
              </div>
            </div>
          )}

          <Section title="Transaction">
            <Field label="Amount (₹)" value={formatCurrencyINR(transaction.amount)} />
            <Field label="Currency" value={transaction.currency} />
            <Field label="Payment Method" value={transaction.payment_method} />
            <Field label="Channel" value={transaction.channel} />
          </Section>

          <Section title="Merchant">
            <Field label="Merchant" value={transaction.merchant_name} />
            <Field label="Merchant ID" value={transaction.merchant_id} mono />
            <Field label="Category" value={transaction.merchant_category} />
            <Field label="Risk Score" value={formatScore(transaction.merchant_risk_score, 2)} />
          </Section>

          <Section title="User & Device">
            <Field label="User ID" value={transaction.user_id} mono />
            <Field label="Account Age" value={`${formatInt(transaction.account_age_days)} days`} />
            <Field label="Device Type" value={transaction.device_type} />
            <Field label="Device ID" value={transaction.device_id} mono />
            <Field label="Device Risk Score" value={formatScore(transaction.device_risk_score, 2)} />
          </Section>

          <Section title="Location">
            <Field label="City" value={transaction.city} />
            <Field label="State" value={transaction.state} />
            <Field label="Country" value={transaction.country} />
            <Field label="Foreign Transaction" value={transaction.is_foreign ? "Yes" : "No"} />
          </Section>

          <Section title="Spending Behavior">
            <Field label="Avg User Spend" value={formatCurrencyINR(transaction.avg_user_spend)} />
            <Field label="Median User Spend" value={formatCurrencyINR(transaction.median_user_spend)} />
            <Field label="Velocity (1h)" value={`${formatInt(transaction.velocity_1h)} txns`} />
            <Field label="Velocity (24h)" value={`${formatInt(transaction.velocity_24h)} txns`} />
          </Section>

          {transaction.fraud_type && (
            <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              Labeled fraud type:{" "}
              <span className="font-semibold capitalize">{transaction.fraud_type.replace(/_/g, " ")}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">{children}</dl>
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-slate-400">{label}</dt>
      <dd className={`truncate text-sm font-medium text-slate-800 ${mono ? "font-mono" : ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}
