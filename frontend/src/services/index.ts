// FraudLab frontend service layer (Phase 7.0) — single import surface.
export * from "./types";
export { apiClient, API_BASE_URL, ApiRequestError } from "./apiClient";
export { scoreTransactions } from "./scoringService";
export { explainTransactions } from "./explainabilityService";
export { investigateExplanations, analyzeTransactions } from "./investigationService";
export {
  generateDataset,
  getDatasetHistory,
  getDatasetRun,
  deleteDatasetRun,
  DEFAULT_DATASET_CONFIG,
} from "./datasetService";
export { getModelMetadata, getHealth } from "./modelService";
