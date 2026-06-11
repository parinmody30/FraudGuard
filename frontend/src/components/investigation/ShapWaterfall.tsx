// Diverging SHAP contribution chart. Red (increase) extends right, green (decrease) left.
import type { Contributor } from "@/types";

export function ShapWaterfall({ contributors }: { contributors: Contributor[] }) {
  if (!contributors.length) {
    return <p className="text-sm text-slate-400">No contributors returned for this prediction.</p>;
  }
  const maxAbs = Math.max(...contributors.map((c) => Math.abs(c.impact))) || 1;

  return (
    <div className="space-y-2.5">
      {contributors.map((c) => {
        const pct = (Math.abs(c.impact) / maxAbs) * 50; // half-width max
        const increase = c.direction === "increase";
        return (
          <div key={c.feature} className="flex items-center gap-3 text-sm">
            <div className="w-36 truncate text-right text-slate-600" title={c.feature}>{c.feature}</div>
            <div className="relative h-5 flex-1">
              <div className="absolute inset-y-0 left-1/2 w-px bg-slate-200" />
              <div
                className={`absolute inset-y-0 ${increase ? "bg-red-400" : "bg-emerald-400"} rounded`}
                style={
                  increase
                    ? { left: "50%", width: `${pct}%` }
                    : { right: "50%", width: `${pct}%` }
                }
              />
            </div>
            <div className={`w-14 text-right font-mono text-xs font-semibold ${increase ? "text-red-600" : "text-emerald-600"}`}>
              {c.impact >= 0 ? "+" : ""}{c.impact.toFixed(2)}
            </div>
          </div>
        );
      })}
      <div className="flex justify-between px-36 pt-1 text-[10px] text-slate-400">
        <span>← lowers risk</span><span>raises risk →</span>
      </div>
    </div>
  );
}
