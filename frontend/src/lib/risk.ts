// UI mappings for decision/confidence/tier. Mirrors the locked Phase 4 policy
// (display-only — thresholds themselves live in the backend).
import type { Confidence, Decision } from "@/types";

export const FRAUD_TIERS: Record<string, "easy" | "medium" | "hard"> = {
  account_takeover: "easy",
  large_spend: "easy",
  geo_anomaly: "easy",
  night_time: "easy",
  new_device: "easy",
  card_testing: "medium",
  velocity: "medium",
  merchant: "medium",
  beneficiary: "medium",
  synthetic_identity: "hard",
  money_mule: "hard",
  gradual_escalation: "hard",
  cross_channel: "hard",
  multi_stage: "hard",
};

export function tierOf(fraudType: string | null | undefined): "legit" | "easy" | "medium" | "hard" {
  if (!fraudType) return "legit";
  return FRAUD_TIERS[fraudType] ?? "legit";
}

export function decisionClasses(decision: Decision): string {
  switch (decision) {
    case "BLOCK":
      return "bg-red-50 text-red-600 ring-1 ring-red-200";
    case "REVIEW":
      return "bg-amber-50 text-amber-600 ring-1 ring-amber-200";
    default:
      return "bg-emerald-50 text-emerald-600 ring-1 ring-emerald-200";
  }
}

export function confidenceClasses(confidence: Confidence): string {
  switch (confidence) {
    case "HIGH":
      return "bg-emerald-50 text-emerald-600 ring-1 ring-emerald-200";
    case "MEDIUM":
      return "bg-amber-50 text-amber-600 ring-1 ring-amber-200";
    default:
      return "bg-slate-100 text-slate-600 ring-1 ring-slate-200";
  }
}

export function scoreColor(prob: number): string {
  if (prob >= 0.8) return "text-red-600";
  if (prob >= 0.4) return "text-amber-600";
  return "text-slate-500";
}

export function riskLabel(prob: number): { label: string; classes: string } {
  if (prob >= 0.8) return { label: "HIGH RISK", classes: "bg-red-50 text-red-600 ring-1 ring-red-200" };
  if (prob >= 0.4) return { label: "MEDIUM RISK", classes: "bg-amber-50 text-amber-600 ring-1 ring-amber-200" };
  return { label: "LOW RISK", classes: "bg-emerald-50 text-emerald-600 ring-1 ring-emerald-200" };
}
