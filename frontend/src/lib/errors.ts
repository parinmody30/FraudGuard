// Convert any thrown error (incl. ApiRequestError) into a user-safe message.
// Never surface stack traces to the UI.
import { ApiRequestError } from "@/services/apiClient";

export function toUserMessage(error: unknown): string {
  if (error instanceof ApiRequestError) {
    if (error.status === 503) {
      return "A required model artifact is not available on the server yet. Please train the champion model and retry.";
    }
    if (error.status === 422) {
      return `Invalid request: ${error.message}`;
    }
    return error.message || `Request failed (${error.status}).`;
  }
  if (error instanceof Error) {
    if (error.message.includes("Failed to fetch")) {
      return "Cannot reach the FraudLab API. Confirm the backend is running on the configured base URL.";
    }
    return error.message;
  }
  return "An unexpected error occurred.";
}
