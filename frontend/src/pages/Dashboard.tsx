import {
  Activity,
  FileText,
  Gauge,
  Percent,
  ShieldX,
  Trophy,
} from "lucide-react";
import { useModelMetadata } from "@/hooks/useModel";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ChartPlaceholder } from "@/components/dashboard/ChartPlaceholder";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { formatInt, formatPct, formatScore, formatTimestamp } from "@/lib/format";
import { toUserMessage } from "@/lib/errors";

export default function Dashboard() {
  const { data, isLoading, isError, error, refetch } = useModelMetadata();

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-sm text-slate-500">Overview of your fraud detection environment</p>
        </div>
        {data?.training_timestamp && (
          <span className="text-xs text-slate-400">
            Champion trained {formatTimestamp(data.training_timestamp)}
          </span>
        )}
      </div>

      {isLoading && <LoadingState label="Loading champion model metadata…" />}
      {isError && <ErrorState message={toUserMessage(error)} onRetry={() => refetch()} />}

      {data && (
        <>
          <DashboardMetrics data={data} />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <ChartPlaceholder title="Fraud by Type" hint="Donut Chart: Type Distribution" />
            <ChartPlaceholder title="Fraud by Difficulty" hint="Bar Chart: Difficulty Distribution" />
            <ChartPlaceholder title="Transactions Over Time" hint="Line Chart: Timeline Trend" />
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

            <div className="card p-6">
              <h3 className="mb-4 text-lg font-bold text-slate-900">Recent Investigations</h3>
              <EmptyState message="No persisted investigations endpoint in the contract. Open the Investigations workspace to analyze a transaction live." />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function DashboardMetrics({ data }: { data: NonNullable<ReturnType<typeof useModelMetadata>["data"]> }) {
  const datasetSize = data.dataset_size ?? null;
  const fraudRate = data.fraud_rate ?? null;
  const fraudCount =
    datasetSize !== null && fraudRate !== null ? Math.round(datasetSize * fraudRate) : null;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-6">
      <MetricCard
        icon={FileText}
        tag="Total"
        title="Total Transactions"
        value={formatInt(datasetSize)}
        sub="Training dataset"
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
