// Dataset Generator hook — wraps datasetService.generateDataset as a mutation,
// and tracks client-side generation time (the contract response has no timing field).
// On success it refreshes the saved-dataset history so the new run appears immediately.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { generateDataset } from "@/services";
import type { DatasetConfig, DatasetSummary } from "@/types";

export interface DatasetResult {
  summary: DatasetSummary;
  elapsedSeconds: number;
}

export function useGenerateDataset() {
  const qc = useQueryClient();
  return useMutation<DatasetResult, unknown, Partial<DatasetConfig>>({
    mutationFn: async (config) => {
      const start = performance.now();
      const summary = await generateDataset(config);
      return { summary, elapsedSeconds: (performance.now() - start) / 1000 };
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-history"] });
    },
  });
}
