// Reports + system status service. Consumer pages: Reports, Settings.
import { apiClient } from "./apiClient";
import type { ReportsSummary, SystemStatus } from "./types";

/** GET /api/reports/summary — offline benchmark leaderboard + CTGAN experiment outcome. */
export function getReportsSummary(): Promise<ReportsSummary> {
  return apiClient.get<ReportsSummary>("/api/reports/summary");
}

/** GET /api/system/status — availability of database, Groq key and model artifacts. */
export function getSystemStatus(): Promise<SystemStatus> {
  return apiClient.get<SystemStatus>("/api/system/status");
}
