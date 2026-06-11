import type { Confidence, Decision } from "@/types";
import { confidenceClasses, decisionClasses } from "@/lib/risk";

export function DecisionBadge({ decision }: { decision: Decision }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${decisionClasses(decision)}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {decision}
    </span>
  );
}

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${confidenceClasses(confidence)}`}>
      {confidence}
    </span>
  );
}

export function RiskBadge({ label, classes }: { label: string; classes: string }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${classes}`}>
      {label}
    </span>
  );
}
