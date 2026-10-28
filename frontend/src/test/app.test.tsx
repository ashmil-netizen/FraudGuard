import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { api } from "../api";
import App from "../App";
import AnalyzerPage from "../pages/AnalyzerPage";
import GraphPage from "../pages/GraphPage";
import InvestigationPage from "../pages/InvestigationPage";
import QueuePage from "../pages/QueuePage";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
  ApiError: class ApiError extends Error {
    status = 503;
    code = "api_error";
  },
  api: {
    health: vi.fn().mockResolvedValue({
      status: "ready",
      components: { dataset: true, model: true, graph: true, reports: true, artifacts_loaded: true },
      errors: {},
    }),
    dashboard: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    transactions: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    transaction: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    graph: vi.fn().mockResolvedValue({ nodes: [], edges: [], counts: { nodes: 0, edges: 0 } }),
    graphNode: vi.fn().mockResolvedValue({ account_id: "A", nodes: [], edges: [], counts: { nodes: 0, edges: 0 } }),
    traverse: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    dataset: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    model: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    predict: vi.fn().mockRejectedValue(new Error("Backend unavailable")),
  },
  };
});

vi.mock("react-force-graph-2d", () => ({
  default: ({
    graphData,
    onNodeClick,
    onLinkClick,
  }: {
    graphData: { nodes: Array<{ id: string }>; links: Array<{ id: string }> };
    onNodeClick?: (node: { id: string }) => void;
    onLinkClick?: (link: { id: string }) => void;
  }) => (
    <div data-testid="force-graph">
      {graphData.nodes.map((node) => <button key={node.id} onClick={() => onNodeClick?.(node)}>Select node {node.id}</button>)}
      {graphData.links.map((link) => <button key={link.id} onClick={() => onLinkClick?.(link)}>Select edge {link.id}</button>)}
    </div>
  ),
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.health).mockResolvedValue({
    status: "ready",
    components: { dataset: true, model: true, graph: true, reports: true, artifacts_loaded: true },
    errors: {},
  });
  vi.mocked(api.dashboard).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.transactions).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.transaction).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.graph).mockResolvedValue({ nodes: [], edges: [], counts: { nodes: 0, edges: 0 } });
  vi.mocked(api.graphNode).mockResolvedValue({ account_id: "A", nodes: [], edges: [], counts: { nodes: 0, edges: 0 } });
  vi.mocked(api.traverse).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.dataset).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.model).mockRejectedValue(new Error("Backend unavailable"));
  vi.mocked(api.predict).mockRejectedValue(new Error("Backend unavailable"));
});

describe("application navigation", () => {
  it("opens each primary route from the workspace navigation", async () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Transaction overview" })).toBeInTheDocument(), { timeout: 5000 });
    fireEvent.click(screen.getByRole("link", { name: /Transaction analyzer/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Transaction analyzer" })).toBeInTheDocument(), { timeout: 5000 });
    fireEvent.click(screen.getByRole("link", { name: /Graph explorer/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Graph explorer" })).toBeInTheDocument(), { timeout: 5000 });
    fireEvent.click(screen.getByRole("link", { name: /BFS \/ DFS lab/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "BFS / DFS lab" })).toBeInTheDocument(), { timeout: 5000 });
    fireEvent.click(screen.getByRole("link", { name: /Candidate queue/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Candidate queue" })).toBeInTheDocument(), { timeout: 5000 });
    fireEvent.click(screen.getByRole("link", { name: /Dataset & model/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dataset and model" })).toBeInTheDocument(), { timeout: 5000 });
  });

  it("shows a real API loading error rather than fabricated dashboard metrics", async () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(await screen.findAllByText("Live data unavailable")).not.toHaveLength(0);
    expect(screen.queryByText("99.7%")).not.toBeInTheDocument();
  });

  it("applies graph filters and opens the selected account panel", async () => {
    const edge = {
      id: "tx-1",
      transaction_id: "tx-1",
      source: "A",
      target: "B",
      type: "TRANSFER",
      amount: 25,
      prediction: { predicted_class: 1 as const, score: 0.8, threshold: 0.5, score_interpretation: "uncalibrated" },
    };
    vi.mocked(api.graph).mockResolvedValue({
      nodes: [{ id: "A" }, { id: "B" }],
      edges: [edge],
      counts: { nodes: 2, edges: 1 },
    });
    vi.mocked(api.graphNode).mockResolvedValue({
      account_id: "A",
      nodes: [{ id: "A" }, { id: "B" }],
      edges: [edge],
      counts: { nodes: 2, edges: 1 },
    });
    render(<MemoryRouter initialEntries={["/graph"]}><GraphPage /></MemoryRouter>);
    await screen.findByRole("button", { name: "Select node A" });
    const typeFields = screen.getAllByLabelText("Transaction type");
    fireEvent.change(typeFields[typeFields.length - 1]!, { target: { value: "TRANSFER" } });
    fireEvent.click(screen.getByLabelText("Candidate edges only"));
    fireEvent.click(screen.getByRole("button", { name: /Apply filters/i }));
    await waitFor(() => expect(api.graph).toHaveBeenLastCalledWith(
      { limit: 300, type: "TRANSFER", predicted: true },
      expect.any(AbortSignal),
    ));
    fireEvent.click(screen.getByRole("button", { name: "Select node A" }));
    expect(await screen.findByText("A", { selector: ".selection-id" })).toBeInTheDocument();
    expect(await screen.findByText("Predicted candidates")).toBeInTheDocument();
  });

  it("searches a real transaction and submits a clearly hypothetical score", async () => {
    vi.mocked(api.transaction).mockResolvedValue({
      transaction_id: "paysim-1",
      source_row_index: 1,
      step: 400,
      sender: "sender-1",
      recipient: "recipient-1",
      type: "TRANSFER",
      amount: 25,
      prediction: { predicted_class: 0, score: 0.1, threshold: 0.5, score_interpretation: "uncalibrated score" },
    });
    vi.mocked(api.predict).mockResolvedValue({
      prediction_type: "hypothetical",
      predicted_class: 1,
      score: 0.8,
      threshold: 0.5,
      score_interpretation: "uncalibrated score",
      ground_truth_used: false,
      graph_relationship_created: false,
      prediction_point: "post-transaction review",
    });
    render(<MemoryRouter><AnalyzerPage /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Transaction identifier"), { target: { value: "paysim-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText("sender-1")).toBeInTheDocument();
    expect(api.transaction).toHaveBeenCalledWith("paysim-1", expect.any(AbortSignal));

    fireEvent.change(screen.getByLabelText("Transaction amount"), { target: { value: "200" } });
    const typeFields = screen.getAllByLabelText("Transaction type");
    fireEvent.change(typeFields[typeFields.length - 1]!, { target: { value: "TRANSFER" } });
    fireEvent.click(screen.getByRole("button", { name: /Score hypothetical transaction/i }));
    expect(await screen.findByText("Candidate for investigation")).toBeInTheDocument();
    expect(api.predict).toHaveBeenCalledWith(expect.objectContaining({
      amount: 200,
      type: "TRANSFER",
      step: 400,
    }));
    expect(screen.getByText(/not a calibrated fraud probability/i)).toBeInTheDocument();
  });

  it("validates BFS depth and calls backend traversal for a valid start account", async () => {
    const traversal = {
      algorithm: "bfs" as const,
      start_node: "A",
      nodes: [{ id: "A", depth: 0 }],
      edges: [],
      counts: { nodes: 1, edges: 0 },
      traversal_order: ["A"],
      depths: { A: 0 },
      statistics: { visited_accounts: 1, transactions_returned: 0, maximum_depth_reached: 0, truncated: false },
      interpretation: "Connectivity only.",
      ground_truth_labels_included: false as const,
    };
    vi.mocked(api.traverse).mockResolvedValue(traversal);
    render(<MemoryRouter><InvestigationPage /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Starting account"), { target: { value: "A" } });
    const depthInput = screen.getByLabelText("Maximum depth") as HTMLInputElement;
    fireEvent.change(depthInput, { target: { value: "7" } });
    expect(depthInput.validity.valid).toBe(false);
    expect(api.traverse).not.toHaveBeenCalled();
    fireEvent.change(depthInput, { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: /Run investigation/i }));
    await waitFor(() => expect(api.traverse).toHaveBeenCalledWith({
        start_node: "A",
        algorithm: "bfs",
        maximum_depth: 2,
        direction: "both",
        maximum_nodes: 500,
        maximum_edges: 2_000,
      }));
    expect(await screen.findByText("Traversal order")).toBeInTheDocument();
  });

  it("uses backend pagination for the candidate queue", async () => {
    vi.mocked(api.transactions)
      .mockResolvedValueOnce({
        items: [{
          transaction_id: "candidate-1",
          source_row_index: 1,
          step: 400,
          sender: "sender-1",
          recipient: "recipient-1",
          type: "TRANSFER",
          amount: 25,
          prediction: { predicted_class: 1, score: 0.8, threshold: 0.5, score_interpretation: "uncalibrated" },
        }],
        page: 1,
        page_size: 25,
        total: 26,
        has_next: true,
      })
      .mockResolvedValueOnce({
        items: [{
          transaction_id: "candidate-2",
          source_row_index: 2,
          step: 401,
          sender: "sender-2",
          recipient: "recipient-2",
          type: "CASH_OUT",
          amount: 75,
          prediction: { predicted_class: 1, score: 0.9, threshold: 0.5, score_interpretation: "uncalibrated" },
        }],
        page: 2,
        page_size: 25,
        total: 26,
        has_next: false,
      });
    render(<MemoryRouter><QueuePage /></MemoryRouter>);
    await screen.findByText("Page 1 · 26 results");
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(api.transactions).toHaveBeenLastCalledWith(
      { page: 2, page_size: 25, predicted: true },
      expect.any(AbortSignal),
    ));
    expect(await screen.findByText("Page 2 · 26 results")).toBeInTheDocument();
  });
});
