import { useMemo, useState } from "react";
import { Filter, RefreshCw, Search, Waypoints } from "lucide-react";
import { api } from "../api";
import { GraphCanvas } from "../components/GraphCanvas";
import {
  EmptyState,
  ErrorState,
  formatAmount,
  LoadingState,
  PageTitle,
  Panel,
  ScoreBadge,
} from "../components/Ui";
import { useResource } from "../hooks";
import type { GraphEdge, GraphFilters, GraphNode } from "../types";

const initialFilters: GraphFilters = { limit: 300 };

export default function GraphPage() {
  const [filters, setFilters] = useState<GraphFilters>(initialFilters);
  const [formType, setFormType] = useState("");
  const [predictedOnly, setPredictedOnly] = useState(false);
  const [accountQuery, setAccountQuery] = useState("");
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const graph = useResource((signal) => api.graph(filters, signal), [filters]);
  const account = useResource(
    (signal) => selectedAccount ? api.graphNode(selectedAccount, "both", 500, signal) : Promise.resolve(null),
    [selectedAccount],
  );
  const filteredData = useMemo(() => graph.data ?? { nodes: [], edges: [], counts: { nodes: 0, edges: 0 } }, [graph.data]);

  function applyFilters(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSelectedEdge(null);
    setSelectedAccount(null);
    setFilters({
      limit: 300,
      ...(formType ? { type: formType } : {}),
      ...(predictedOnly ? { predicted: true } : {}),
    });
  }

  function submitAccountSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (accountQuery.trim()) {
      setSelectedEdge(null);
      setSelectedAccount(accountQuery.trim());
    }
  }

  return (
    <>
      <PageTitle eyebrow="NETWORK INTELLIGENCE" title="Graph explorer" description="Explore directed sender-to-recipient transfers and their individual model-scored transaction edges." action={<button className="button button-secondary" onClick={graph.refresh}><RefreshCw size={15} /> Refresh graph</button>} />
      <div className="scope-banner"><Waypoints size={16} /><span>This is a bounded view of the scored PaySim test split, not the complete transaction network. Red edges are predicted candidates; accounts are not classified.</span></div>
      <div className="graph-layout">
        <Panel title="Transaction network" subtitle={graph.data ? `${graph.data.counts.nodes.toLocaleString()} accounts · ${graph.data.counts.edges.toLocaleString()} transaction edges` : "Loading bounded graph"}>
          <form className="graph-filterbar" onSubmit={applyFilters}>
            <label><span>Transaction type</span><select value={formType} onChange={(event) => setFormType(event.target.value)}><option value="">All types</option><option>PAYMENT</option><option>TRANSFER</option><option>CASH_OUT</option><option>CASH_IN</option><option>DEBIT</option></select></label>
            <label className="checkbox-label"><input type="checkbox" checked={predictedOnly} onChange={(event) => setPredictedOnly(event.target.checked)} /> Candidate edges only</label>
            <button type="submit" className="button button-primary"><Filter size={15} /> Apply filters</button>
          </form>
          {graph.loading ? <LoadingState label="Loading bounded account network" /> : graph.error ? <ErrorState message={graph.error} onRetry={graph.refresh} /> : graph.data?.edges.length ? (
            <>
              {graph.data.truncated && <div className="info-strip">This view is limited to the requested graph size. Narrow it with type or prediction filters.</div>}
              <GraphCanvas data={filteredData} onNodeSelect={(node) => { setSelectedAccount(node.id); setSelectedEdge(null); }} onEdgeSelect={(edge) => { setSelectedEdge(edge); setSelectedAccount(null); }} />
              <div className="graph-legend"><span><i className="legend-node" />Account</span><span><i className="legend-line" />Transaction</span><span><i className="legend-line legend-alert" />Predicted candidate</span><span className="muted">Arrow direction: sender → recipient</span></div>
            </>
          ) : <EmptyState>No graph edges match the selected filters.</EmptyState>}
        </Panel>
        <aside className="graph-side-stack">
          <Panel title="Find an account" subtitle="Inspect a bounded one-hop neighborhood">
            <form className="inline-search compact-search" onSubmit={submitAccountSearch}>
              <label className="sr-only" htmlFor="account-search">Account identifier</label><Search size={16} aria-hidden="true" />
              <input id="account-search" value={accountQuery} onChange={(event) => setAccountQuery(event.target.value)} placeholder="Account ID" />
              <button className="icon-button" aria-label="Search account" type="submit"><Search size={16} /></button>
            </form>
          </Panel>
          {selectedAccount && <Panel title="Selected account" subtitle={selectedAccount}>
            {account.loading ? <LoadingState label="Loading account neighborhood" /> : account.error ? <ErrorState message={account.error} onRetry={account.refresh} /> : account.data && <AccountDetails data={account.data} />}
          </Panel>}
          {selectedEdge && <Panel title="Transaction edge" subtitle={selectedEdge.transaction_id}>
            <EdgeDetails edge={selectedEdge} />
          </Panel>}
          {!selectedAccount && !selectedEdge && <Panel title="Selection details" subtitle="Select an account or transaction edge">
            <EmptyState>Click a node to inspect its connected transactions, or click an edge for its real transaction attributes and model output.</EmptyState>
          </Panel>}
          <Panel title="Rendering limits" subtitle="Performance and coverage">
            <p className="subtle-note">Graph endpoint returns at most 5,000 edges. It selects a bounded subset from the indexed chronological test split. Repeated transfers remain separate keyed edges.</p>
            <p className="subtle-note">A highlighted transaction is a model prediction candidate only; graph proximity does not determine account risk.</p>
          </Panel>
        </aside>
      </div>
    </>
  );
}

function AccountDetails({ data }: { data: { account_id: string; nodes: GraphNode[]; edges: GraphEdge[] } }) {
  return (
    <div className="selection-details">
      <div className="selection-id mono">{data.account_id}</div>
      <div className="detail-grid compact-details">
        <div><span>Transactions in response</span><strong>{data.edges.length}</strong></div>
        <div><span>Connected accounts</span><strong>{Math.max(data.nodes.length - 1, 0)}</strong></div>
        <div><span>Predicted candidates</span><strong>{data.edges.filter((edge) => edge.prediction?.predicted_class === 1).length}</strong></div>
      </div>
      <ul className="connected-list">{data.edges.slice(0, 8).map((edge) => <li key={edge.transaction_id}><span className="mono">{edge.transaction_id}</span><small>{edge.source === data.account_id ? `to ${edge.target}` : `from ${edge.source}`}</small></li>)}</ul>
    </div>
  );
}

function EdgeDetails({ edge }: { edge: GraphEdge }) {
  return (
    <div className="selection-details">
      <div className="detail-grid compact-details">
        <div><span>Direction</span><strong className="mono">{edge.source} → {edge.target}</strong></div>
        <div><span>Transaction type</span><strong>{edge.type ?? "Unavailable"}</strong></div>
        <div><span>Amount units</span><strong>{edge.amount == null ? "Unavailable" : formatAmount(edge.amount)}</strong></div>
        <div><span>Simulation step</span><strong>{edge.step ?? "Unavailable"}</strong></div>
      </div>
      {edge.prediction && <ScoreBadge score={edge.prediction.score} flagged={Boolean(edge.prediction.predicted_class)} />}
    </div>
  );
}
