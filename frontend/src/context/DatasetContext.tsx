// Tracks which persisted dataset run is "active" — the one that drives the Dashboard
// charts/metrics and the Transactions table. Persisted in localStorage so the selection
// survives reloads. `null` means "use the latest run in history" (resolved by consumers).
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

interface DatasetContextValue {
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;
}

const ACTIVE_KEY = "fraudlab.activeDatasetId";

const DatasetContext = createContext<DatasetContextValue | undefined>(undefined);

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [activeRunId, setActiveRunIdState] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(ACTIVE_KEY);
    if (stored) setActiveRunIdState(stored);
  }, []);

  const setActiveRunId = useCallback((id: string | null) => {
    setActiveRunIdState(id);
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
  }, []);

  return (
    <DatasetContext.Provider value={{ activeRunId, setActiveRunId }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDatasetSelection(): DatasetContextValue {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error("useDatasetSelection must be used within a DatasetProvider");
  return ctx;
}
