// Explainability service (Phase 7.0). Consumer page: Investigation Workspace (SHAP waterfall).
import { apiClient } from "./apiClient";
import type { Explanation, TransactionIn } from "./types";

export interface ExplainResponse {
  explanations: Explanation[];
}

/** POST /api/explain — prediction + SHAP contributors (waterfall-ready) per transaction. */
export function explainTransactions(transactions: TransactionIn[]): Promise<ExplainResponse> {
  return apiClient.post<ExplainResponse>("/api/explain", { transactions });
}
