import { RadialBar, RadialBarChart, PolarAngleAxis } from "recharts";

export function FraudGauge({ prob }: { prob: number }) {
  const pct = Math.round(prob * 100);
  const color = prob >= 0.8 ? "#ef4444" : prob >= 0.4 ? "#f59e0b" : "#10b981";
  const data = [{ value: pct, fill: color }];

  return (
    <div className="relative mx-auto h-48 w-48">
      <RadialBarChart
        width={192}
        height={192}
        cx="50%"
        cy="50%"
        innerRadius="72%"
        outerRadius="100%"
        barSize={14}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar background={{ fill: "#f1f5f9" }} dataKey="value" cornerRadius={8} />
      </RadialBarChart>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-extrabold" style={{ color }}>
          {pct}%
        </span>
        <span className="mt-1 text-[10px] uppercase tracking-widest text-slate-400">Fraud Probability</span>
      </div>
    </div>
  );
}
