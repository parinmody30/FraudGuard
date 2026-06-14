// FraudLab API contract types (Phase 7.0).
// Mirror backend/api/schemas.py exactly. Single source of truth for the frontend.

export type Decision = "ALLOW" | "REVIEW" | "BLOCK";
export type Confidence = "LOW" | "MEDIUM" | "HIGH";
export type Direction = "increase" | "decrease";
export type AgentMode = "groq" | "fallback";

export interface TransactionIn {
  transaction_id: string;
  user_id: string;
  timestamp: string;
  amount: number;
  currency: string;
  merchant_id: string;
  merchant_name: string;
  merchant_category: string;
  payment_method: string;
  channel: string;
  device_id: string;
  device_type: string;
  city: string;
  state: string;
  country: string;
  is_foreign: boolean;
  account_age_days: number;
  avg_user_spend: number;
  median_user_spend: number;
  velocity_1h: number;
  velocity_24h: number;
  merchant_risk_score: number;
  device_risk_score: number;
  is_fraud?: boolean | null;
  fraud_type?: string | null;
  [key: string]: unknown;
}

export interface Prediction {
  transaction_id: string;
  fraud_probability: number;
  decision: Decision;
  confidence: Confidence;
}

export interface Contributor {
  feature: string;
  impact: number;
  direction: Direction;
}

export interface Explanation {
  transaction_id: string;
  fraud_probability: number;
  decision: Decision;
  confidence: Confidence;
  contributors: Contributor[];
}

export interface Investigation {
  transaction_id: string;
  summary: string;
  signals_detected: string[];
  recommendation: string;
  confidence_explanation: string;
  next_steps: string[];
}

export interface AnalyzedTransaction {
  transaction_id: string;
  prediction: Prediction;
  explanation: Explanation;
  investigation: Investigation;
}

export interface DatasetConfig {
  num_users: number;
  transactions_per_user: number;
  fraud_rate: number;
  business_type: "bank" | "credit_card" | "upi" | "ecommerce";
  risk_appetite: "low" | "medium" | "high";
  seed: number;
  preview_limit: number;
}

export interface DatasetSummary {
  total_transactions: number;
  fraud_count: number;
  fraud_percentage: number;
  fraud_type_distribution: Record<string, number>;
  business_type: string;
  num_users: number;
  transactions_per_user: number;
  random_seed: number;
  sample: Record<string, unknown>[];
  run_id?: string | null;
}

/** A persisted dataset generation (summary fields only) — GET /api/dataset/history. */
export interface DatasetRunListItem {
  id: string;
  created_at: string;
  label: string;
  business_type: string;
  risk_appetite: string;
  num_users: number;
  transactions_per_user: number;
  fraud_rate: number;
  seed: number;
  total_transactions: number;
  fraud_count: number;
  fraud_percentage: number;
  stored_count: number;
}

/** A persisted run including its capped sample of rows — GET /api/dataset/history/{id}. */
export interface DatasetRunDetail extends DatasetRunListItem {
  fraud_type_distribution: Record<string, number>;
  transactions: TransactionIn[];
}

export interface ModelMetadata {
  model_type?: string;
  champion?: string;
  training_timestamp?: string;
  dataset_size?: number;
  fraud_rate?: number;
  metrics?: Record<string, number | null>;
  thresholds?: Record<string, number>;
  confidence_bands?: Record<string, number>;
  [key: string]: unknown;
}

export interface Health {
  status: "ok";
  service: string;
  version: string;
}

export interface ApiError {
  detail: string;
  code?: string | null;
}
