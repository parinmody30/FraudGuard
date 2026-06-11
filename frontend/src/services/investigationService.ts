// Investigation service (Phase 7.0). Consumer page: Investigation Workspace.
import { apiClient } from "./apiClient";
import type {
  AgentMode,
  AnalyzedTransaction,
  Explanation,
  Investigation,
  TransactionIn,
} from "./types";

export interface InvestigateResponse {
  investigations: Investigation[];
  mode: AgentMode;
}

export interface AnalyzeResponse {
  results: AnalyzedTransaction[];
  mode: AgentMode;
}

/** POST /api/investigate — grounded narratives from pre-computed explanations. */
export function investigateExplanations(
  explanations: Explanation[],
  useGroq = true
): Promise<InvestigateResponse> {
  return apiClient.post<InvestigateResponse>("/api/investigate", {
    explanations,
    use_groq: useGroq,
  });
}

/** POST /api/transactions/analyze — end-to-end score + explain + investigate. */
export function analyzeTransactions(
  transactions: TransactionIn[],
  useGroq = true
): Promise<AnalyzeResponse> {
  return apiClient.post<AnalyzeResponse>("/api/transactions/analyze", {
    transactions,
    use_groq: useGroq,
  });
}
