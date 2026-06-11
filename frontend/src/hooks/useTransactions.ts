// Transactions Explorer hook.
// Contract has no "list transactions" endpoint, so rows are sourced via the only
// contract path: dataset/generate (preview sample) -> score those rows. Both calls go
// through the service layer; nothing is fabricated.
import { useQuery } from "@tanstack/react-query";
import { generateDataset, scoreTransactions } from "@/services";
import type { ScoredTransaction } from "@/types";
import type { DatasetConfig, TransactionIn } from "@/types";

export interface TransactionsParams {
  num_users: number;
  transactions_per_user: number;
  fraud_rate: number;
  business_type: DatasetConfig["business_type"];
  seed: number;
  preview_limit: number;
}

export const DEFAULT_TX_PARAMS: TransactionsParams = {
  num_users: 200,
  transactions_per_user: 50,
  fraud_rate: 0.06,
  business_type: "bank",
  seed: 42,
  preview_limit: 200,
};

// Lightweight sample fetch (no scoring) — used by the Investigation page when opened
// directly without a transaction passed via router state.
export function useSampleTransactions(limit = 25, enabled = true) {
  return useQuery({
    queryKey: ["sample-transactions", limit],
    enabled,
    queryFn: async (): Promise<TransactionIn[]> => {
      const dataset = await generateDataset({
        num_users: 60,
        transactions_per_user: 40,
        fraud_rate: 0.08,
        seed: 42,
        preview_limit: limit,
      });
      return (dataset.sample ?? []) as unknown as TransactionIn[];
    },
  });
}

export function useScoredTransactions(params: TransactionsParams) {
  return useQuery({
    queryKey: ["transactions", params],
    queryFn: async (): Promise<ScoredTransaction[]> => {
      const dataset = await generateDataset(params);
      const rows = (dataset.sample ?? []) as unknown as TransactionIn[];
      if (rows.length === 0) return [];
      const scored = await scoreTransactions(rows);
      const byId = new Map(scored.predictions.map((p) => [p.transaction_id, p]));
      return rows.map((row) => ({ ...row, prediction: byId.get(row.transaction_id) }));
    },
  });
}
