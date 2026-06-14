// Single source of contract types: re-export the service-layer types (Phase 7.0).
// Do NOT redefine backend contracts here.
export type {
  Decision,
  Confidence,
  Direction,
  AgentMode,
  TransactionIn,
  Prediction,
  Contributor,
  Explanation,
  Investigation,
  AnalyzedTransaction,
  DatasetConfig,
  DatasetSummary,
  DatasetRunListItem,
  DatasetRunDetail,
  ModelMetadata,
  Health,
  ApiError,
} from "@/services/types";

// UI-only derived view models (not backend contracts).
import type { Prediction, TransactionIn } from "@/services/types";

export interface ScoredTransaction extends TransactionIn {
  prediction?: Prediction;
}
