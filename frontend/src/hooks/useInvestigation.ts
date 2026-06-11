// Investigation Workspace hook — single call to analyzeTransactions (score+explain+investigate).
import { useQuery } from "@tanstack/react-query";
import { analyzeTransactions } from "@/services";
import type { AnalyzedTransaction, TransactionIn } from "@/types";

export function useAnalyzeTransaction(
  transaction: TransactionIn | null,
  useGroq = true
) {
  return useQuery({
    queryKey: ["analyze", transaction?.transaction_id, useGroq],
    enabled: Boolean(transaction),
    queryFn: async (): Promise<{ result: AnalyzedTransaction; mode: string }> => {
      const response = await analyzeTransactions([transaction as TransactionIn], useGroq);
      return { result: response.results[0], mode: response.mode };
    },
  });
}
