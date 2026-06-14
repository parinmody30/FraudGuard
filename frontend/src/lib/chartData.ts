// Derives dashboard chart data from scored transaction rows (client-side aggregation —
// the Phase 7.0 contract has no dedicated analytics endpoint).
import { tierOf } from "@/lib/risk";
import type { ScoredTransaction } from "@/types";

export interface NamedCount {
  name: string;
  value: number;
}

export interface TimelinePoint {
  label: string;
  total: number;
  fraud: number;
}

const TIER_LABELS: Record<"easy" | "medium" | "hard", string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

export function fraudByType(rows: ScoredTransaction[], maxSlices = 6): NamedCount[] {
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (!row.fraud_type) continue;
    const key = row.fraud_type.replace(/_/g, " ");
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const sorted = Array.from(counts.entries())
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  if (sorted.length <= maxSlices) return sorted;

  const top = sorted.slice(0, maxSlices - 1);
  const otherTotal = sorted.slice(maxSlices - 1).reduce((sum, d) => sum + d.value, 0);
  return [...top, { name: "other", value: otherTotal }];
}

export function fraudByDifficulty(rows: ScoredTransaction[]): NamedCount[] {
  const counts: Record<"easy" | "medium" | "hard", number> = { easy: 0, medium: 0, hard: 0 };
  for (const row of rows) {
    if (!row.fraud_type) continue;
    const tier = tierOf(row.fraud_type);
    if (tier === "legit") continue;
    counts[tier] += 1;
  }
  return (["easy", "medium", "hard"] as const).map((tier) => ({
    name: TIER_LABELS[tier],
    value: counts[tier],
  }));
}

const DAY_MS = 24 * 60 * 60 * 1000;

export function transactionsOverTime(rows: ScoredTransaction[], maxBuckets = 12): TimelinePoint[] {
  const dated = rows
    .map((row) => ({
      date: new Date(row.timestamp),
      fraud: row.prediction ? row.prediction.decision !== "ALLOW" : Boolean(row.is_fraud),
    }))
    .filter((row) => !Number.isNaN(row.date.getTime()))
    .sort((a, b) => a.date.getTime() - b.date.getTime());

  if (dated.length === 0) return [];

  const minDay = Math.floor(dated[0].date.getTime() / DAY_MS);
  const maxDay = Math.floor(dated[dated.length - 1].date.getTime() / DAY_MS);
  const totalDays = maxDay - minDay + 1;
  const daysPerBucket = Math.max(1, Math.ceil(totalDays / maxBuckets));

  const buckets = new Map<number, { total: number; fraud: number; start: number }>();
  for (const { date, fraud } of dated) {
    const day = Math.floor(date.getTime() / DAY_MS);
    const idx = Math.floor((day - minDay) / daysPerBucket);
    const start = (minDay + idx * daysPerBucket) * DAY_MS;
    const entry = buckets.get(idx) ?? { total: 0, fraud: 0, start };
    entry.total += 1;
    if (fraud) entry.fraud += 1;
    buckets.set(idx, entry);
  }

  return Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => {
      const start = new Date(v.start);
      const end = new Date(v.start + (daysPerBucket - 1) * DAY_MS);
      const fmt = (d: Date) => d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      return {
        label: daysPerBucket === 1 ? fmt(start) : `${fmt(start)}–${fmt(end)}`,
        total: v.total,
        fraud: v.fraud,
      };
    });
}
