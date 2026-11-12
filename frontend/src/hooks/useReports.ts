// Reports + system status hooks (services only).
import { useQuery } from "@tanstack/react-query";
import { getReportsSummary, getSystemStatus } from "@/services";

export function useReportsSummary() {
  return useQuery({ queryKey: ["reports", "summary"], queryFn: getReportsSummary, staleTime: 5 * 60_000 });
}

export function useSystemStatus() {
  return useQuery({ queryKey: ["system", "status"], queryFn: getSystemStatus, refetchInterval: 30_000 });
}
