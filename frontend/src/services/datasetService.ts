// Dataset service (Phase 7.0). Consumer page: Dataset Studio / Simulator.
import { apiClient } from "./apiClient";
import type { DatasetConfig, DatasetSummary } from "./types";

export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  num_users: 200,
  transactions_per_user: 50,
  fraud_rate: 0.06,
  business_type: "bank",
  risk_appetite: "medium",
  seed: 42,
  preview_limit: 20,
};

/** POST /api/dataset/generate — generate a synthetic dataset; returns summary + preview. */
export function generateDataset(
  config: Partial<DatasetConfig> = {}
): Promise<DatasetSummary> {
  return apiClient.post<DatasetSummary>("/api/dataset/generate", {
    ...DEFAULT_DATASET_CONFIG,
    ...config,
  });
}
