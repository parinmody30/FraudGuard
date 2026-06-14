import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { NamedCount } from "@/lib/chartData";

const COLORS = ["#1d4ed8", "#3b82f6", "#60a5fa", "#f97316", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"];

export function FraudTypeDonut({ data }: { data: NamedCount[] }) {
  if (data.length === 0) {
    return <span className="text-sm text-slate-400">No fraud transactions in this sample.</span>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="38%"
          cy="50%"
          innerRadius={48}
          outerRadius={75}
          paddingAngle={2}
        >
          {data.map((entry, i) => (
            <Cell key={entry.name} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend
          layout="vertical"
          align="right"
          verticalAlign="middle"
          wrapperStyle={{ fontSize: 12, textTransform: "capitalize" }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
