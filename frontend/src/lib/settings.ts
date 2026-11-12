// User preferences persisted in localStorage (per browser). Read by the Investigation page.
import { useCallback, useEffect, useState } from "react";

export interface AppSettings {
  /** Ask the backend for Groq-written narratives (falls back to offline text without a key). */
  useAiNarratives: boolean;
  /** Analyst name printed on exported investigation reports. */
  analystName: string;
}

const KEY = "fraudlab_settings";
const EVENT = "fraudlab-settings-changed";
export const DEFAULT_SETTINGS: AppSettings = { useAiNarratives: true, analystName: "" };

export function readSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<AppSettings>) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function writeSettings(next: AppSettings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable — settings stay in memory for this session */
  }
  window.dispatchEvent(new Event(EVENT));
}

export function useSettings(): [AppSettings, (patch: Partial<AppSettings>) => void] {
  const [settings, setSettings] = useState<AppSettings>(readSettings);
  useEffect(() => {
    const sync = () => setSettings(readSettings());
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  const update = useCallback((patch: Partial<AppSettings>) => {
    const next = { ...readSettings(), ...patch };
    writeSettings(next);
    setSettings(next);
  }, []);
  return [settings, update];
}
