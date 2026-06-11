import type { LucideIcon } from "lucide-react";

export function MetricCard({
  icon: Icon,
  tag,
  title,
  value,
  sub,
  accent = "slate",
  highlight = false,
}: {
  icon: LucideIcon;
  tag: string;
  title: string;
  value: string;
  sub?: string;
  accent?: "slate" | "red" | "blue" | "emerald";
  highlight?: boolean;
}) {
  const accentBg = {
    slate: "bg-slate-100 text-slate-500",
    red: "bg-red-50 text-red-500",
    blue: "bg-blue-50 text-brand",
    emerald: "bg-emerald-50 text-emerald-500",
  }[accent];

  return (
    <div className={`card p-5 ${highlight ? "ring-2 ring-brand/30" : ""}`}>
      <div className="flex items-start justify-between">
        <div className={`flex h-9 w-9 items-center justify-center rounded-lg ${accentBg}`}>
          <Icon className="h-5 w-5" />
        </div>
        <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
          {tag}
        </span>
      </div>
      <div className="mt-4 text-sm text-slate-500">{title}</div>
      <div className={`mt-1 text-2xl font-bold ${accent === "red" ? "text-red-600" : "text-slate-900"}`}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
    </div>
  );
}
