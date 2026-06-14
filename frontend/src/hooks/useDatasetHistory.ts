// Dataset history hooks — list persisted runs, fetch one run's stored rows (scored), and delete.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteDatasetRun, getDatasetHistory, getDatasetRun, scoreTransactions } from "@/services";
import type { DatasetRunListItem, ScoredTransaction } from "@/types";

/** List of persisted dataset runs (newest first). Empty when DB is offline. */
export function useDatasetHistory() {
  return useQuery<DatasetRunListItem[]>({
    queryKey: ["dataset-history"],
    queryFn: getDatasetHistory,
    staleTime: 30_000,
  });
}

/**
 * Stored transactions for a run, scored for decisions/probabilities.
 * `runId` may be null — the query stays disabled until an id is resolved.
 */
export function useDatasetRunTransactions(runId: string | null | undefined) {
  return useQuery<ScoredTransaction[]>({
    queryKey: ["dataset-run-transactions", runId],
    enabled: Boolean(runId),
    queryFn: async (): Promise<ScoredTransaction[]> => {
      const run = await getDatasetRun(runId as string);
      const rows = run.transactions ?? [];
      if (rows.length === 0) return [];
      const scored = await scoreTransactions(rows);
      const byId = new Map(scored.predictions.map((p) => [p.transaction_id, p]));
      return rows.map((row) => ({ ...row, prediction: byId.get(row.transaction_id) }));
    },
  });
}

export function useDeleteDatasetRun() {
  const qc = useQueryClient();
  return useMutation<void, unknown, string>({
    mutationFn: (id) => deleteDatasetRun(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-history"] });
    },
  });
}
