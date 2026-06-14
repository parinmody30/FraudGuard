// Base HTTP client for the FraudLab API (Phase 7.0).
// A thin typed fetch wrapper. No business logic; services build on this.

import type { ApiError } from "./types";

export const API_BASE_URL =
  (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

export class ApiRequestError extends Error {
  status: number;
  code?: string | null;
  constructor(status: number, detail: string, code?: string | null) {
    super(detail);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });

  if (!response.ok) {
    let detail = response.statusText;
    let code: string | null = null;
    try {
      const body = (await response.json()) as ApiError;
      detail = body.detail ?? detail;
      code = body.code ?? null;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiRequestError(response.status, detail, code);
  }
  // 204 No Content (e.g. DELETE) has no JSON body to parse.
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const apiClient = {
  get: <T>(path: string) => request<T>(path, { method: "GET" }),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
