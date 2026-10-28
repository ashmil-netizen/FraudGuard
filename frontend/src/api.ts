import type {
  ApiEnvelope,
  ApiErrorPayload,
  DashboardMetrics,
  DatasetSummary,
  GraphData,
  GraphFilters,
  Health,
  HypotheticalPrediction,
  ModelSummary,
  PredictionInput,
  Transaction,
  TransactionFilters,
  TransactionPage,
  TraversalResult,
} from "./types";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(
    message: string,
    status: number,
    code = "api_error",
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch {
    throw new ApiError(
      `Could not connect to FraudGuard API at ${API_BASE_URL}. Start the backend and try again.`,
      0,
      "network_error",
    );
  }

  const payload = (await response.json().catch(() => ({}))) as
    | ApiEnvelope<T>
    | ApiErrorPayload;
  if (!response.ok) {
    const failure = payload as ApiErrorPayload;
    throw new ApiError(
      failure.error?.message ?? `Request failed with status ${response.status}.`,
      response.status,
      failure.error?.code,
    );
  }
  return (payload as ApiEnvelope<T>).data;
}

function queryString(params: object): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (typeof value !== "string" && typeof value !== "number" && typeof value !== "boolean" && value !== undefined) continue;
    if (value !== undefined) query.set(key, String(value));
  }
  const encoded = query.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<Health>("/api/health", { signal }),
  dashboard: (signal?: AbortSignal) =>
    request<DashboardMetrics>("/api/dashboard/metrics", { signal }),
  dataset: (signal?: AbortSignal) =>
    request<DatasetSummary>("/api/dataset/summary", { signal }),
  model: (signal?: AbortSignal) =>
    request<ModelSummary>("/api/model/summary", { signal }),
  transactions: (filters: TransactionFilters, signal?: AbortSignal) =>
    request<TransactionPage>(
      `/api/transactions${queryString(filters)}`,
      { signal },
    ),
  transaction: (transactionId: string, signal?: AbortSignal) =>
    request<Transaction>(
      `/api/transactions/${encodeURIComponent(transactionId)}`,
      { signal },
    ),
  graph: (filters: GraphFilters, signal?: AbortSignal) =>
    request<GraphData>(`/api/graph${queryString(filters)}`, { signal }),
  graphNode: (
    nodeId: string,
    direction: "out" | "in" | "both" = "both",
    maximumTransactions = 500,
    signal?: AbortSignal,
  ) =>
    request<GraphData & { account_id: string }>(
      `/api/graph/node/${encodeURIComponent(nodeId)}${queryString({
        direction,
        maximum_transactions: maximumTransactions,
      })}`,
      { signal },
    ),
  traverse: (
    payload: {
      start_node: string;
      algorithm: "bfs" | "dfs";
      maximum_depth: number;
      direction: "out" | "in" | "both";
      maximum_nodes: number;
      maximum_edges: number;
    },
    signal?: AbortSignal,
  ) =>
    request<TraversalResult>("/api/graph/traverse", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
  predict: (payload: PredictionInput, signal?: AbortSignal) =>
    request<HypotheticalPrediction>("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
};
