// Builds shareable outputs for an investigated transaction: a self-contained HTML report
// (downloaded by "Export Report") and a plain-text case summary (copied by "Share").
import type { AnalyzedTransaction, TransactionIn } from "@/types";
import { formatCurrencyINR, formatTimestamp } from "@/lib/format";

export interface ReportContext {
  transaction: TransactionIn;
  result: AnalyzedTransaction;
  mode: string;
  analystAction: string | null;
  notes: string;
  analyst: string;
}

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

export function buildShareText({ transaction: t, result: r, analystAction, notes }: ReportContext): string {
  const p = r.prediction;
  const lines = [
    `FraudLab investigation — ${t.transaction_id}`,
    `Decision: ${p.decision} (fraud probability ${pct(p.fraud_probability)}, ${p.confidence} confidence)`,
    `Amount: ₹${formatCurrencyINR(t.amount)} · ${t.merchant_name} (${t.merchant_category}) · ${t.channel}`,
    `User: ${t.user_id} · ${t.city}, ${t.state} · ${formatTimestamp(t.timestamp)}`,
    "",
    `Summary: ${r.investigation.summary}`,
    "",
    "Top risk drivers:",
    ...r.explanation.contributors.slice(0, 5).map(
      (c) => `  • ${c.feature}: ${c.impact > 0 ? "+" : ""}${c.impact.toFixed(2)} (${c.direction}s risk)`,
    ),
    "",
    `Recommendation: ${r.investigation.recommendation}`,
  ];
  if (analystAction) lines.push(`Analyst action: ${analystAction}`);
  if (notes.trim()) lines.push(`Analyst notes: ${notes.trim()}`);
  return lines.join("\n");
}

export function buildHtmlReport(ctx: ReportContext): string {
  const { transaction: t, result: r, mode, analystAction, notes, analyst } = ctx;
  const p = r.prediction;
  const color = p.decision === "BLOCK" ? "#dc2626" : p.decision === "REVIEW" ? "#d97706" : "#059669";
  const maxImpact = Math.max(0.01, ...r.explanation.contributors.map((c) => Math.abs(c.impact)));
  const detailRows: [string, string][] = [
    ["Transaction ID", t.transaction_id],
    ["User ID", t.user_id],
    ["Timestamp", formatTimestamp(t.timestamp)],
    ["Amount", `₹${formatCurrencyINR(t.amount)}`],
    ["Merchant", `${t.merchant_name} (${t.merchant_category})`],
    ["Payment / channel", `${t.payment_method} · ${t.channel}`],
    ["Device", `${t.device_type} (risk ${t.device_risk_score.toFixed(2)})`],
    ["Location", `${t.city}, ${t.state}, ${t.country}${t.is_foreign ? " (foreign)" : ""}`],
    ["Account age", `${t.account_age_days} days`],
    ["User avg / median spend", `₹${formatCurrencyINR(t.avg_user_spend)} / ₹${formatCurrencyINR(t.median_user_spend)}`],
    ["Velocity 1h / 24h", `${t.velocity_1h} / ${t.velocity_24h} transactions`],
    ["Merchant risk", t.merchant_risk_score.toFixed(2)],
  ];

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Investigation ${esc(t.transaction_id)}</title>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;background:#f8fafc;margin:0;padding:32px}
  .wrap{max-width:900px;margin:0 auto;background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:32px}
  h1{font-size:22px;margin:0} h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin:28px 0 10px}
  .muted{color:#64748b;font-size:13px} .row{display:flex;gap:16px;flex-wrap:wrap;margin-top:20px}
  .kpi{flex:1;min-width:150px;border:1px solid #e2e8f0;border-radius:12px;padding:14px}
  .kpi b{display:block;font-size:22px;margin-top:4px} table{width:100%;border-collapse:collapse;font-size:14px}
  td{padding:7px 0;border-bottom:1px solid #f1f5f9} td:first-child{color:#64748b;width:42%}
  .bar{height:10px;border-radius:5px} ul,ol{padding-left:20px;margin:0} li{margin:6px 0;font-size:14px}
  .box{background:#eff6ff;border:1px solid #dbeafe;border-radius:12px;padding:14px;font-size:14px;line-height:1.5}
  footer{margin-top:28px;font-size:12px;color:#94a3b8}
  @media print{body{background:#fff;padding:0}.wrap{border:none}}
</style></head><body><div class="wrap">
  <div class="muted">FraudLab AI · Transaction Investigation Report</div>
  <h1>Transaction ${esc(t.transaction_id)}</h1>
  <div class="row">
    <div class="kpi"><span class="muted">Decision</span><b style="color:${color}">${esc(p.decision)}</b></div>
    <div class="kpi"><span class="muted">Fraud probability</span><b>${pct(p.fraud_probability)}</b></div>
    <div class="kpi"><span class="muted">Confidence</span><b>${esc(p.confidence)}</b></div>
    <div class="kpi"><span class="muted">Analyst action</span><b>${esc(analystAction ?? "Pending")}</b></div>
  </div>

  <h2>AI investigation summary</h2>
  <div class="box">${esc(r.investigation.summary)}<br><br><b>Recommendation:</b> ${esc(r.investigation.recommendation)}<br>
  <span class="muted">${esc(r.investigation.confidence_explanation)}</span></div>

  <h2>Top risk drivers (SHAP)</h2>
  <table>${r.explanation.contributors
    .map(
      (c) => `<tr><td>${esc(c.feature)}</td><td><div style="display:flex;align-items:center;gap:8px">
      <div class="bar" style="width:${(Math.abs(c.impact) / maxImpact) * 60}%;background:${c.impact > 0 ? "#ef4444" : "#10b981"}"></div>
      <span>${c.impact > 0 ? "+" : ""}${c.impact.toFixed(2)}</span></div></td></tr>`,
    )
    .join("")}</table>
  <p class="muted">Red bars raise fraud risk, green bars lower it (impact in log-odds).</p>

  <h2>Signals detected</h2>
  <ul>${r.investigation.signals_detected.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>

  <h2>Recommended next steps</h2>
  <ol>${r.investigation.next_steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>

  <h2>Transaction details</h2>
  <table>${detailRows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>

  ${notes.trim() ? `<h2>Analyst notes</h2><div class="box" style="background:#f8fafc;border-color:#e2e8f0">${esc(notes).replace(/\n/g, "<br>")}</div>` : ""}

  <footer>Generated ${esc(new Date().toLocaleString())}${analyst ? ` by ${esc(analyst)}` : ""} · Narrative engine: ${mode === "groq" ? "Groq LLM" : "Offline analyst engine (rule-based)"} · Model: XGBoost champion</footer>
</div></body></html>`;
}

export function downloadFile(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for non-secure contexts (e.g. opened over a LAN IP instead of localhost).
    try {
      const ta = Object.assign(document.createElement("textarea"), { value: text });
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}
