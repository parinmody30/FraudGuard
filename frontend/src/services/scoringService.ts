// Scoring service (Phase 7.0). Consumer page: Transactions / Scoring.
import { apiClient } from "./apiClient";
import type { Prediction, TransactionIn } from "./types";

export interface ScoreResponse {
  predictions: Prediction[];
}

/** POST /api/score — fraud probability + ALLOW/REVIEW/BLOCK + confidence per transaction. */
export function scoreTransactions(transactions: TransactionIn[]): Promise<ScoreResponse> {
  return apiClient.post<ScoreResponse>("/api/score", { transactions });
}
