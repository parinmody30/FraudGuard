import type { ReactNode } from "react";
import {
  Activity,
  Database,
  FileText,
  Gauge,
  Percent,
  ShieldX,
  Trophy,
} from "lucide-react";
import { useModelMetadata } from "@/hooks/useModel";
import { useActiveDataset } from "@/hooks/useActiveDataset";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { FraudTypeDonut } from "@/components/dashboard/FraudTypeDonut";
import { FraudDifficultyBar } from "@/components/dashboard/FraudDifficultyBar";
import { TransactionsTimeline } from "@/components/dashboard/TransactionsTimeline";
import { DatasetHistoryPanel } from "@/components/dashboard/DatasetHistoryPanel";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { formatInt, formatPct, formatScore, formatTimestamp } from "@/lib/format";
import { fraudByDifficulty, fraudByType, transactionsOverTime } from "@/lib/chartData";
import { toUserMessage } from "@/lib/errors";
import type { DatasetRunListItem, ScoredTransaction } from "@/types";

export default function Dashboard() {
  const { data, isLoading, isError, error, refetch } = useModelMetadata();
  const ds = useActiveDataset();

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-sm text-slate-500">Overview of your fraud detection environment</p>
        </div>
        <div className="text-right text-xs text-slate-400">
          {ds.activeRun && (
            <div className="flex items-center justify-end gap-1.5 font-medium text-brand">
              <Database className="h-3.5 w-3.5" /> Active dataset · {formatTimestamp(ds.activeRun.created_at)}
            </div>
          )}
          {data?.training_timestamp && (
            <div>Champion trained {formatTimestamp(data.training_timestamp)}</div>
          )}
        </div>
      </div>

      {isLoading && <LoadingState label="Loading champion model metadata…" />}
      {isError && <ErrorState message={toUserMessage(error)} onRetry={() => refetch()} />}

      {data && (
        <>
          <DashboardMetrics data={data} activeRun={ds.activeRun} />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <ChartCard title="Fraud by Type" ds={ds}>
              {(rows) => <FraudTypeDonut data={fraudByType(rows)} />}
            </ChartCard>
            <ChartCard title="Fraud by Difficulty" ds={ds}>
              {(rows) => <FraudDifficultyBar data={fraudByDifficulty(rows)} />}
            </ChartCard>
            <ChartCard title="Transactions Over Time" ds={ds}>
              {(rows) => <TransactionsTimeline data={transactionsOverTime(rows)} />}
            </ChartCard>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="card p-6 lg:col-span-2">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-bold text-slate-900">Champion Model</h3>
                <span className="text-sm text-slate-400">from /api/model/metadata</span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-400">
                    <th className="pb-2">Model</th>
                    <th className="pb-2">PR-AUC</th>
                    <th className="pb-2">Recall</th>
                    <th className="pb-2">Recall (hard)</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-slate-100">
                    <td className="py-3 font-medium text-slate-800">
                      <span className="mr-2 inline-block h-2 w-2 rounded-full bg-brand align-middle" />
                      {data.model_type ?? "XGBoost"} (Champion)
                    </td>
                    <td className="py-3 font-mono text-emerald-600">
                      {formatScore(data.metrics?.pr_auc)}
                    </td>
                    <td className="py-3 font-mono">{formatScore(data.metrics?.recall)}</td>
                    <td className="py-3 font-mono">{formatScore(data.metrics?.recall_hard)}</td>
                    <td className="py-3">
                      <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-brand">
                        DEPLOYED
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-4 text-[11px] text-slate-400">
                Full training leaderboard requires a dedicated endpoint (not in the Phase 7.0 contract).
              </p>
            </div>

            <DatasetHistoryPanel />
          </div>
        </>
      )}
    </div>
  );
}

function ChartCard({
  title,
  ds,
  children,
}: {
  title: string;
  ds: ReturnType<typeof useActiveDataset>;
  children: (rows: ScoredTransaction[]) => ReactNode;
}) {
  return (
    <div className="card p-6">
      <h3 className="text-lg font-bold text-slate-900">{title}</h3>
      <div className="mt-4 flex h-56 items-center justify-center">
        {ds.isLoading && <LoadingState label="Loading transactions…" />}
        {ds.isError && <ErrorState message={toUserMessage(ds.error)} onRetry={() => ds.refetch()} />}
        {!ds.isLoading && !ds.isError && ds.data && ds.data.length === 0 && (
          <EmptyState message="No transactions in this dataset." />
        )}
        {ds.data && ds.data.length > 0 && children(ds.data)}
      </div>
    </div>
  );
}

function DashboardMetrics({
  data,
  activeRun,
}: {
  data: NonNullable<ReturnType<typeof useModelMetadata>["data"]>;
  activeRun: DatasetRunListItem | null;
}) {
  // Prefer the active (saved) dataset's stats; fall back to champion-training metadata.
  const datasetSize = activeRun?.total_transactions ?? data.dataset_size ?? null;
  const fraudRate = activeRun ? activeRun.fraud_percentage / 100 : data.fraud_rate ?? null;
  const fraudCount =
    activeRun?.fraud_count ??
    (data.dataset_size != null && data.fraud_rate != null
      ? Math.round(data.dataset_size * data.fraud_rate)
      : null);
  const sub = activeRun ? "Saved dataset" : "Training dataset";

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-6">
      <MetricCard
        icon={FileText}
        tag="Total"
        title="Total Transactions"
        value={formatInt(datasetSize)}
        sub={sub}
        accent="blue"
      />
      <MetricCard
        icon={ShieldX}
        tag="Flagged"
        title="Fraud Transactions"
        value={formatInt(fraudCount)}
        sub={fraudRate !== null ? formatPct(fraudRate) + " of dataset" : undefined}
        accent="red"
      />
      <MetricCard
        icon={Percent}
        tag="Rate"
        title="Fraud Rate"
        value={formatPct(fraudRate)}
        sub="Target: < 2.5%"
      />
      <MetricCard
        icon={Trophy}
        tag="Model"
        title="Champion Model"
        value={data.model_type ?? "XGBoost"}
        sub="Active in Prod"
        accent="blue"
        highlight
      />
      <MetricCard
        icon={Activity}
        tag="Metric"
        title="PR-AUC"
        value={formatScore(data.metrics?.pr_auc)}
        accent="emerald"
      />
      <MetricCard
        icon={Gauge}
        tag="Metric"
        title="Recall"
        value={formatScore(data.metrics?.recall)}
        accent="emerald"
      />
    </div>
  );
}
