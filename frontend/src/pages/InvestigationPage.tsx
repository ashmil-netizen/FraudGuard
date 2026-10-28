import { useState } from "react";
import { GitBranch, GitFork, Search } from "lucide-react";
import { api, ApiError } from "../api";
import { GraphCanvas } from "../components/GraphCanvas";
import {
  EmptyState,
  ErrorState,
  formatAmount,
  LoadingState,
  PageTitle,
  Panel,
} from "../components/Ui";
import type { GraphEdge, GraphNode, TraversalResult } from "../types";

export default function InvestigationPage() {
  const [result, setResult] = useState<TraversalResult | null>(null);
  const [startNode, setStartNode] = useState("");
  const [algorithm, setAlgorithm] = useState<"bfs" | "dfs">("bfs");
  const [depth, setDepth] = useState(2);
  const [direction, setDirection] = useState<"out" | "in" | "both">("both");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<GraphNode | GraphEdge | null>(null);

  async function runTraversal(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!startNode.trim() || depth < 0 || depth > 6) {
      setError("Enter an account identifier and choose a depth from 0 to 6.");
      return;
    }
    setError(null);
    setResult(null);
    setSelected(null);
    setLoading(true);
    try {
      setResult(await api.traverse({
        start_node: startNode.trim(),
        algorithm,
        maximum_depth: depth,
        direction,
        maximum_nodes: 500,
        maximum_edges: 2_000,
      }));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "The graph traversal request failed.");
    } finally {
      setLoading(false);
    }
  }

  const highlightedEdges = result?.edges.map((edge) => edge.transaction_id) ?? [];
  return (
    <>
      <PageTitle eyebrow="ACCOUNT INVESTIGATION" title="BFS / DFS lab" description="Use the backend traversal services to explore bounded account connectivity." />
      <div className="scope-banner"><GitBranch size={16} /><span>BFS and DFS reveal graph reachability only. They are not fraud classifiers, and account nodes are never labeled fraudulent from connected transactions.</span></div>
      <div className="algorithm-cards">
        <div className={`algorithm-card ${algorithm === "bfs" ? "algorithm-selected" : ""}`}><GitFork size={19} /><div><strong>Breadth-first search</strong><p>Explores connected accounts level by level from the start account.</p></div></div>
        <div className={`algorithm-card ${algorithm === "dfs" ? "algorithm-selected" : ""}`}><GitBranch size={19} /><div><strong>Depth-first search</strong><p>Follows one connection path deeper before backtracking.</p></div></div>
      </div>
      <Panel title="Configure traversal" subtitle="Limits and validation are enforced by the API">
        <form className="traversal-form" onSubmit={runTraversal}>
          <label>Starting account<div className="field-with-icon"><Search size={16} /><input value={startNode} onChange={(event) => setStartNode(event.target.value)} required maxLength={128} placeholder="Enter PaySim account ID" /></div></label>
          <label>Algorithm<select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as "bfs" | "dfs")}><option value="bfs">BFS — level by level</option><option value="dfs">DFS — path before backtracking</option></select></label>
          <label>Maximum depth<input type="number" min="0" max="6" value={depth} onChange={(event) => setDepth(Number(event.target.value))} /></label>
          <label>Direction<select value={direction} onChange={(event) => setDirection(event.target.value as "out" | "in" | "both")}><option value="both">Incoming and outgoing</option><option value="out">Outgoing</option><option value="in">Incoming</option></select></label>
          <button className="button button-primary traversal-submit" disabled={loading || !startNode.trim()}>{loading ? "Traversing…" : "Run investigation"}<GitFork size={15} /></button>
        </form>
        {error && <div className="form-message"><ErrorState message={error} />{error.includes("not found") && <p className="subtle-note">Try an account from the graph explorer or a transaction's sender/recipient details.</p>}</div>}
        {loading && <LoadingState label="Running bounded backend traversal" />}
      </Panel>
      {result && <>
        <div className="metrics-grid traversal-metrics">
          <div className="mini-metric"><span>Discovered accounts</span><strong>{result.statistics.visited_accounts}</strong></div>
          <div className="mini-metric"><span>Transactions returned</span><strong>{result.statistics.transactions_returned}</strong></div>
          <div className="mini-metric"><span>Maximum depth reached</span><strong>{result.statistics.maximum_depth_reached}</strong></div>
          <div className="mini-metric"><span>Traversal truncated</span><strong>{result.statistics.truncated ? "Yes" : "No"}</strong></div>
        </div>
        <div className="graph-layout">
          <Panel title={`${result.algorithm.toUpperCase()} result network`} subtitle={`${result.start_node} · ${result.traversal_order.length} accounts in backend traversal order`}>
            <GraphCanvas
              data={result}
              highlightedNodes={result.traversal_order}
              highlightedEdges={highlightedEdges}
              onNodeSelect={(node) => setSelected(node)}
              onEdgeSelect={(edge) => setSelected(edge)}
              height={430}
            />
            <p className="subtle-note">{result.interpretation}</p>
          </Panel>
          <Panel title="Traversal order" subtitle="Select a result to inspect it">
            {result.traversal_order.length ? <ol className="traversal-list">{result.traversal_order.map((nodeId, index) => <li key={nodeId}><button onClick={() => setSelected(result.nodes.find((node) => node.id === nodeId) ?? { id: nodeId })}><span className="order-index">{index + 1}</span><span className="mono">{nodeId}</span><span className="depth-tag">Depth {result.depths[nodeId]}</span></button></li>)}</ol> : <EmptyState>No accounts were discovered.</EmptyState>}
            <div className="selection-details">
              {selected && "transaction_id" in selected ? <><strong className="mono">{selected.transaction_id}</strong><p>{selected.source} → {selected.target}</p><p>{selected.type ?? "Transaction"} · {selected.amount == null ? "Amount unavailable" : `${formatAmount(selected.amount)} units`}</p></> : selected && "id" in selected ? <><strong className="mono">{selected.id}</strong><p>Account node · depth {selected.depth ?? result.depths[selected.id]}</p></> : <p className="muted">No result selected.</p>}
            </div>
          </Panel>
        </div>
      </>}
    </>
  );
}
