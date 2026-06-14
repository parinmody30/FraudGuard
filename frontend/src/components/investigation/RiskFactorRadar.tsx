// A normalized "risk fingerprint" radar across the main risk dimensions of a transaction.
// Each axis is scaled to 0–100; a wider web = a riskier transaction at a glance.
import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { TransactionIn } from "@/types";

export function RiskFactorRadar({ transaction }: { transaction: TransactionIn }) {
  const amountRatio = transaction.avg_user_spend > 0
    ? Math.min(transaction.amount / (transaction.avg_user_spend * 3), 1)
    : 0;
  const velocity = Math.min(transaction.velocity_1h / 10, 1);

  const data = [
    { factor: "Merchant", value: round(transaction.merchant_risk_score * 100) },
    { factor: "Device", value: round(transaction.device_risk_score * 100) },
    { factor: "Amount", value: round(amountRatio * 100) },
    { factor: "Velocity", value: round(velocity * 100) },
    { factor: "Foreign", value: transaction.is_foreign ? 100 : 0 },
  ];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="#e2e8f0" />
        <PolarAngleAxis dataKey="factor" tick={{ fontSize: 11, fill: "#64748b" }} />
        <Tooltip formatter={(value) => [`${Number(value)} / 100`, "Risk"]} />
        <Radar dataKey="value" stroke="#1d4ed8" fill="#3b82f6" fillOpacity={0.35} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function round(n: number): number {
  return Math.round(n);
}
