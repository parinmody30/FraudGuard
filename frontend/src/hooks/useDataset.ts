// Dataset Generator hook — wraps datasetService.generateDataset as a mutation,
// and tracks client-side generation time (the contract response has no timing field).
import { useMutation } from "@tanstack/react-query";
import { generateDataset } from "@/services";
import type { DatasetConfig, DatasetSummary } from "@/types";

export interface DatasetResult {
  summary: DatasetSummary;
  elapsedSeconds: number;
}

export function useGenerateDataset() {
  return useMutation<DatasetResult, unknown, Partial<DatasetConfig>>({
    mutationFn: async (config) => {
      const start = performance.now();
      const summary = await generateDataset(config);
      return { summary, elapsedSeconds: (performance.now() - start) / 1000 };
    },
  });
}
