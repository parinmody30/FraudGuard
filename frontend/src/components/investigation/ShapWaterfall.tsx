// Diverging SHAP contribution chart. Red bars (increase) extend right of zero,
// green bars (decrease) extend left.
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Contributor } from "@/types";

export function ShapWaterfall({ contributors }: { contributors: Contributor[] }) {
  if (!contributors.length) {
    return <p className="text-sm text-slate-400">No contributors returned for this prediction.</p>;
  }

  const data = contributors.map((c) => ({
    feature: c.feature,
    impact: c.impact,
    increase: c.direction === "increase",
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={data.length * 38 + 20}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 36, bottom: 4, left: 4 }}>
          <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
          <YAxis
            type="category"
            dataKey="feature"
            width={140}
            tick={{ fontSize: 12, fill: "#475569" }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine x={0} stroke="#cbd5e1" />
          <Tooltip
            cursor={{ fill: "#f8fafc" }}
            formatter={(value) => [Number(value).toFixed(2), "Impact"]}
          />
          <Bar dataKey="impact" radius={4} maxBarSize={18} label={renderLabel}>
            {data.map((d) => (
              <Cell key={d.feature} fill={d.increase ? "#ef4444" : "#10b981"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-1 flex justify-between text-[10px] text-slate-400">
        <span>← lowers risk</span>
        <span>raises risk →</span>
      </div>
    </div>
  );
}

function renderLabel(props: unknown) {
  const { x, y, width, height, value } = props as {
    x?: number | string;
    y?: number | string;
    width?: number | string;
    height?: number | string;
    value?: number | string;
  };
  const num = Number(value ?? 0);
  const px = Number(x ?? 0);
  const py = Number(y ?? 0);
  const pw = Number(width ?? 0);
  const ph = Number(height ?? 0);
  const text = `${num >= 0 ? "+" : ""}${num.toFixed(2)}`;
  const positive = num >= 0;
  const labelX = positive ? px + pw + 6 : px - 6;
  return (
    <text
      x={labelX}
      y={py + ph / 2}
      dy={4}
      textAnchor={positive ? "start" : "end"}
      className="font-mono text-xs font-semibold"
      fill={positive ? "#dc2626" : "#059669"}
    >
      {text}
    </text>
  );
}
