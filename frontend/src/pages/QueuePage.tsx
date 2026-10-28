import { useState } from "react";
import { Download, Filter, ShieldAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api";
import {
  EmptyState,
  ErrorState,
  formatAmount,
  formatNumber,
  LoadingState,
  PageTitle,
  Panel,
  ScoreBadge,
} from "../components/Ui";
import { useResource } from "../hooks";
import type { TransactionFilters } from "../types";

export default function QueuePage() {
  const [filters, setFilters] = useState<TransactionFilters>({ page: 1, page_size: 25, predicted: true });
  const [search, setSearch] = useState("");
  const [type, setType] = useState("");
  const [minAmount, setMinAmount] = useState("");
  const [maxAmount, setMaxAmount] = useState("");
  const key = JSON.stringify(filters);
  const result = useResource((signal) => api.transactions(filters, signal), [key]);

  function applyFilters(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFilters({
      page: 1,
      page_size: 25,
      predicted: true,
      ...(search.trim() ? { transaction_id: search.trim() } : {}),
      ...(type ? { type } : {}),
      ...(minAmount ? { minimum_amount: Number(minAmount) } : {}),
      ...(maxAmount ? { maximum_amount: Number(maxAmount) } : {}),
    });
  }

  return (
    <>
      <PageTitle eyebrow="MODEL REVIEW WORKFLOW" title="Candidate queue" description="Search and paginate transactions flagged by the saved model threshold." action={<button className="button button-disabled" disabled title="CSV export is not implemented by the API"><Download size={15} /> Export unavailable</button>} />
      <div className="scope-banner"><ShieldAlert size={16} /><span>Queue membership is based on predicted class only. The backend has no investigation-status persistence or CSV export endpoint; those controls are intentionally unavailable.</span></div>
      <Panel title="Filter predicted candidates" subtitle="Filters query actual scored records in the test split">
        <form className="queue-filters" onSubmit={applyFilters}>
          <label>Transaction ID<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Exact identifier" /></label>
          <label>Type<select value={type} onChange={(event) => setType(event.target.value)}><option value="">All types</option><option>PAYMENT</option><option>TRANSFER</option><option>CASH_OUT</option><option>CASH_IN</option><option>DEBIT</option></select></label>
          <label>Minimum amount<input value={minAmount} onChange={(event) => setMinAmount(event.target.value)} type="number" min="0" step="any" /></label>
          <label>Maximum amount<input value={maxAmount} onChange={(event) => setMaxAmount(event.target.value)} type="number" min="0" step="any" /></label>
          <button className="button button-primary" type="submit"><Filter size={15} /> Apply filters</button>
        </form>
      </Panel>
      <Panel title="Predicted candidates" subtitle={result.data ? `${formatNumber(result.data.total)} records match filters` : "Scored test-split records"}>
        {result.loading ? <LoadingState label="Loading candidate queue" /> : result.error ? <ErrorState message={result.error} onRetry={result.refresh} /> : result.data?.items.length ? <>
          <div className="table-scroll"><table><thead><tr><th>Transaction ID</th><th>Amount</th><th>Type</th><th>Class</th><th>Score</th><th>Sender</th><th>Recipient</th><th>Step</th><th>Review</th></tr></thead><tbody>
            {result.data.items.map((row) => <tr key={row.transaction_id}><td className="mono">{row.transaction_id}</td><td>{formatAmount(row.amount)} units</td><td>{row.type}</td><td><span className="class-tag">1 · predicted</span></td><td><ScoreBadge score={row.prediction.score} flagged /></td><td className="mono">{row.sender}</td><td className="mono">{row.recipient}</td><td>{formatNumber(row.step)}</td><td><Link className="table-link" to={`/analyzer?id=${encodeURIComponent(row.transaction_id)}`}>Inspect</Link></td></tr>)}
          </tbody></table></div>
          <div className="pagination">
            <span>Page {result.data.page} · {formatNumber(result.data.total)} results</span>
            <div><button className="button button-secondary" disabled={filters.page <= 1} onClick={() => setFilters((current) => ({ ...current, page: Math.max(1, current.page - 1) }))}>Previous</button><button className="button button-secondary" disabled={!result.data.has_next} onClick={() => setFilters((current) => ({ ...current, page: current.page + 1 }))}>Next</button></div>
          </div>
        </> : <EmptyState>No predicted candidates match these filters.</EmptyState>}
      </Panel>
      <p className="footnote">No operational status updates are available. Review state is not represented by the PaySim ground-truth label.</p>
    </>
  );
}
