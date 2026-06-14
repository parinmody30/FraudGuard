// Compares this transaction's amount against the user's typical spend (average + median).
// The transaction bar is colored by how far above normal it sits — a quick anomaly read.
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatCurrencyINR } from "@/lib/format";

export function SpendComparisonChart({
  amount,
  avg,
  median,
}: {
  amount: number;
  avg: number;
  median: number;
}) {
  const ratio = avg > 0 ? amount / avg : 1;
  const txnColor = ratio >= 3 ? "#ef4444" : ratio >= 1.5 ? "#f59e0b" : "#10b981";

  const data = [
    { name: "This Txn", value: amount, fill: txnColor },
    { name: "Avg Spend", value: avg, fill: "#94a3b8" },
    { name: "Median", value: median, fill: "#cbd5e1" },
  ];

  return (
    <div>
      <ResponsiveContainer width="100%" height={170}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: 8 }}>
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <YAxis hide />
          <Tooltip
            cursor={{ fill: "#f8fafc" }}
            formatter={(value) => [`₹${formatCurrencyINR(Number(value))}`, "Amount"]}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={56}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.fill} />
            ))}
            <LabelList
              dataKey="value"
              position="top"
              formatter={(v) => `₹${formatCurrencyINR(Number(v))}`}
              style={{ fontSize: 10, fill: "#475569", fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-1 text-center text-xs text-slate-500">
        This transaction is{" "}
        <span className="font-semibold" style={{ color: txnColor }}>
          {ratio >= 1 ? `${ratio.toFixed(1)}× the user's average` : `${(ratio * 100).toFixed(0)}% of average`}
        </span>
      </p>
    </div>
  );
}
