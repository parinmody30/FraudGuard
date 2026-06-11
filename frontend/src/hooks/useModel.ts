// Dashboard data hooks — model metadata + health (services only).
import { useQuery } from "@tanstack/react-query";
import { getHealth, getModelMetadata } from "@/services";

export function useModelMetadata() {
  return useQuery({ queryKey: ["model", "metadata"], queryFn: getModelMetadata });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });
}
