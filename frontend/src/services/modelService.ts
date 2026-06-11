// Model + health service (Phase 7.0). Consumer page: Dashboard.
import { apiClient } from "./apiClient";
import type { Health, ModelMetadata } from "./types";

/** GET /api/model/metadata — champion metrics/thresholds for the dashboard. */
export function getModelMetadata(): Promise<ModelMetadata> {
  return apiClient.get<ModelMetadata>("/api/model/metadata");
}

/** GET /api/health — backend liveness probe. */
export function getHealth(): Promise<Health> {
  return apiClient.get<Health>("/api/health");
}
