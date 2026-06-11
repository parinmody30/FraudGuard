// Honest placeholder: the API contract exposes no time-series / distribution chart
// endpoint yet, so we render a labelled placeholder rather than fabricating data.
export function ChartPlaceholder({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="card p-6">
      <h3 className="text-lg font-bold text-slate-900">{title}</h3>
      <div className="mt-4 flex h-56 items-center justify-center rounded-xl border border-dashed border-slate-200 bg-[repeating-linear-gradient(45deg,#f8fafc,#f8fafc_10px,#eef2f7_10px,#eef2f7_20px)]">
        <span className="text-sm text-slate-400">{hint}</span>
      </div>
      <p className="mt-3 text-[11px] text-slate-400">
        Awaiting a dedicated analytics endpoint (not part of the Phase 7.0 contract).
      </p>
    </div>
  );
}
