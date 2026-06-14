// Dataset service (Phase 7.0). Consumer page: Dataset Studio / Simulator.
import { apiClient } from "./apiClient";
import type { DatasetConfig, DatasetRunDetail, DatasetRunListItem, DatasetSummary } from "./types";

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

/** GET /api/dataset/history — list persisted dataset runs (newest first). */
export function getDatasetHistory(): Promise<DatasetRunListItem[]> {
  return apiClient.get<DatasetRunListItem[]>("/api/dataset/history");
}

/** GET /api/dataset/history/{id} — one run with its stored transaction rows. */
export function getDatasetRun(id: string): Promise<DatasetRunDetail> {
  return apiClient.get<DatasetRunDetail>(`/api/dataset/history/${id}`);
}

/** DELETE /api/dataset/history/{id} — remove a persisted run. */
export function deleteDatasetRun(id: string): Promise<void> {
  return apiClient.del(`/api/dataset/history/${id}`);
}
