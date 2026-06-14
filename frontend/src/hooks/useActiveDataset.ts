// Resolves the "active" dataset and its scored transactions for the Dashboard + Transactions
// page. Priority: the user-selected run, else the latest persisted run, else a one-off live
// generation (when history is empty / DB offline). This is what removes the regenerate-on-every-
// load behavior — saved datasets are replayed from Postgres instead.
import { useMemo } from "react";
import { useDatasetHistory, useDatasetRunTransactions } from "@/hooks/useDatasetHistory";
import { DEFAULT_TX_PARAMS, useScoredTransactions } from "@/hooks/useTransactions";
import { useDatasetSelection } from "@/context/DatasetContext";
import type { DatasetRunListItem, ScoredTransaction } from "@/types";

export type DatasetSource = "history" | "live" | "none";

export interface ActiveDataset {
  data: ScoredTransaction[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  refetch: () => void;
  source: DatasetSource;
  activeRun: DatasetRunListItem | null;
  runs: DatasetRunListItem[];
}

export function useActiveDataset(): ActiveDataset {
  const { activeRunId } = useDatasetSelection();
  const history = useDatasetHistory();
  const runs = history.data ?? [];

  const activeRun = useMemo<DatasetRunListItem | null>(() => {
    if (activeRunId) {
      const match = runs.find((r) => r.id === activeRunId);
      if (match) return match;
    }
    return runs[0] ?? null;
  }, [activeRunId, runs]);

  const effectiveId = activeRun?.id ?? null;

  // Fall back to a live generation only once history has loaded and is genuinely empty.
  const fallbackEnabled = !history.isLoading && !history.isError && runs.length === 0;

  const stored = useDatasetRunTransactions(effectiveId);
  const live = useScoredTransactions(DEFAULT_TX_PARAMS, fallbackEnabled);

  if (effectiveId) {
    return {
      data: stored.data,
      isLoading: history.isLoading || stored.isLoading,
      isError: stored.isError,
      error: stored.error,
      refetch: () => {
        history.refetch();
        stored.refetch();
      },
      source: "history",
      activeRun,
      runs,
    };
  }

  if (fallbackEnabled) {
    return {
      data: live.data,
      isLoading: live.isLoading,
      isError: live.isError,
      error: live.error,
      refetch: () => live.refetch(),
      source: "live",
      activeRun: null,
      runs,
    };
  }

  // History still loading (or errored) and no run resolved yet.
  return {
    data: undefined,
    isLoading: history.isLoading,
    isError: history.isError,
    error: history.error,
    refetch: () => history.refetch(),
    source: "none",
    activeRun: null,
    runs,
  };
}
