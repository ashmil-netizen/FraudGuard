export interface ApiEnvelope<T> {
  data: T;
}

export interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    details?: Array<{ location: string; message: string; type: string }>;
  };
}

export interface Health {
  status: "ready" | "degraded";
  components: {
    dataset: boolean;
    model: boolean;
    graph: boolean;
    reports: boolean;
    artifacts_loaded: boolean;
  };
  errors: Record<string, string>;
}

export interface EvaluationMetrics {
  records: number;
  fraud_support: number;
  non_fraud_support: number;
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  average_precision: number | null;
  roc_auc: number | null;
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  flagged_records: number;
  flagged_percentage: number;
  score_description: string;
}

export interface DashboardMetrics {
  scope: string;
  transactions_available: number;
  accounts_represented: number;
  model_predicted_candidates: number;
  evaluation_metrics: EvaluationMetrics | null;
  ground_truth_positive_count: number;
  ground_truth_is_not_prediction_count: boolean;
  transaction_type_distribution: Array<{ type: string; count: number }>;
  amount_distribution: Array<{ range: string; count: number }>;
}

export interface DatasetColumn {
  name: string;
  data_types: string[];
  missing_values: number;
  missing_percentage: number;
  unique_values: Array<[string, number]> | null;
  identifier: boolean;
}

export interface DatasetSummary {
  dataset_identifier: string;
  dataset_kind: string;
  rows: number;
  columns_count: number;
  columns: DatasetColumn[];
  transaction_types: string[];
  ground_truth_labels: {
    column: string;
    counts: Array<[string, number]> | null;
    meaning: string;
  };
  limitations: string[];
  api_transaction_scope: string;
}

export interface Transaction {
  transaction_id: string;
  source_row_index: number;
  step: number;
  sender: string;
  recipient: string;
  type: string;
  amount: number;
  prediction: {
    predicted_class: 0 | 1;
    score: number;
    threshold: number;
    score_interpretation: string;
  };
  connected_accounts?: string[];
}

export interface TransactionPage {
  items: Transaction[];
  page: number;
  page_size: number;
  total: number;
  has_next: boolean;
}

export interface GraphNode {
  id: string;
  node_type?: string;
  attributes?: Record<string, unknown>;
  depth?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  transaction_id: string;
  step?: number;
  amount?: number;
  type?: string;
  prediction?: Transaction["prediction"];
  attributes?: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  counts: { nodes: number; edges: number };
  truncated?: boolean;
}

export interface TraversalResult extends GraphData {
  algorithm: "bfs" | "dfs";
  start_node: string;
  traversal_order: string[];
  depths: Record<string, number>;
  statistics: Record<string, number | boolean | string>;
  interpretation: string;
  ground_truth_labels_included: false;
}

export interface ModelSummary {
  model_type: string;
  dataset_identifier: string;
  feature_schema: string[];
  training_configuration: Record<string, unknown>;
  split_strategy: {
    name: string;
    step_ranges: Record<string, number[]>;
    training_rows_available: number;
    training_rows_used: number;
    validation_rows: number;
    test_rows: number;
  };
  evaluation_metrics: EvaluationMetrics | null;
  classification_threshold: number;
  threshold_selection: string;
  score_interpretation: string;
  limitations: string[];
}

export interface PredictionInput {
  step: number;
  amount: number;
  oldbalanceOrg: number | null;
  newbalanceOrig: number | null;
  oldbalanceDest: number | null;
  newbalanceDest: number | null;
  type: string;
}

export interface HypotheticalPrediction {
  prediction_type: "hypothetical";
  predicted_class: 0 | 1;
  score: number;
  threshold: number;
  score_interpretation: string;
  ground_truth_used: false;
  graph_relationship_created: false;
  prediction_point: string;
}

export interface GraphFilters {
  limit: number;
  type?: string;
  predicted?: boolean;
  minimum_amount?: number;
  maximum_amount?: number;
}

export interface TransactionFilters {
  page: number;
  page_size: number;
  type?: string;
  predicted?: boolean;
  transaction_id?: string;
  minimum_amount?: number;
  maximum_amount?: number;
}
