import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TimelinePoint } from "@/lib/chartData";

export function TransactionsTimeline({ data }: { data: TimelinePoint[] }) {
  if (data.length === 0) {
    return <span className="text-sm text-slate-400">No transactions in this sample.</span>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="totalFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1d4ed8" stopOpacity={0.25} />
            <stop offset="100%" stopColor="#1d4ed8" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eef2f7" />
        <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#94a3b8" }} minTickGap={20} />
        <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: "#94a3b8" }} />
        <Tooltip />
        <Area type="monotone" dataKey="total" name="Transactions" stroke="#1d4ed8" fill="url(#totalFill)" strokeWidth={2} />
        <Area type="monotone" dataKey="fraud" name="Flagged" stroke="#ef4444" fill="url(#fraudFill)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
