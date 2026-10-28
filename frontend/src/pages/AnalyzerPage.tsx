import { useState } from "react";
import { AlertCircle, ArrowRight, Search, Sparkles } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api";
import {
  ErrorState,
  formatAmount,
  formatNumber,
  LoadingState,
  PageTitle,
  Panel,
  ScoreBadge,
} from "../components/Ui";
import { useResource } from "../hooks";
import type { HypotheticalPrediction, PredictionInput, Transaction } from "../types";

const BALANCE_FIELDS = [
  ["oldbalanceOrg", "Sender balance before"],
  ["newbalanceOrig", "Sender balance after"],
  ["oldbalanceDest", "Recipient balance before"],
  ["newbalanceDest", "Recipient balance after"],
] as const;

export default function AnalyzerPage() {
  const [params] = useSearchParams();
  const [query, setQuery] = useState(params.get("id") ?? "");
  const [transactionId, setTransactionId] = useState(params.get("id") ?? "");
  const lookup = useResource<Transaction | null>(
    (signal) => transactionId ? api.transaction(transactionId, signal) : Promise.resolve(null),
    [transactionId],
  );
  const [prediction, setPrediction] = useState<HypotheticalPrediction | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  async function submitPrediction(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPredictError(null);
    setPrediction(null);
    const form = new FormData(event.currentTarget);
    const numeric = (name: string) => {
      const value = String(form.get(name) ?? "").trim();
      return value ? Number(value) : null;
    };
    const input: PredictionInput = {
      step: Number(form.get("step")),
      amount: Number(form.get("amount")),
      oldbalanceOrg: numeric("oldbalanceOrg"),
      newbalanceOrig: numeric("newbalanceOrig"),
      oldbalanceDest: numeric("oldbalanceDest"),
      newbalanceDest: numeric("newbalanceDest"),
      type: String(form.get("type") ?? ""),
    };
    setPredicting(true);
    try {
      setPrediction(await api.predict(input));
    } catch (error) {
      setPredictError(error instanceof ApiError ? error.message : "The hypothetical prediction failed.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <>
      <PageTitle eyebrow="RECORD REVIEW" title="Transaction analyzer" description="Inspect an indexed scored transaction or submit a hypothetical transaction to the saved classifier." />
      <div className="analyzer-grid">
        <Panel title="Find an existing transaction" subtitle="Exact ID lookup from the scored PaySim test split">
          <form className="inline-search" onSubmit={(event) => { event.preventDefault(); setTransactionId(query.trim()); }}>
            <label className="sr-only" htmlFor="transaction-search">Transaction identifier</label>
            <Search size={17} aria-hidden="true" />
            <input id="transaction-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. paysim-000005444003" />
            <button className="button button-primary" type="submit" disabled={!query.trim()}>Search</button>
          </form>
          {lookup.loading && transactionId && <LoadingState label="Looking up transaction" />}
          {lookup.error && <ErrorState message={lookup.error} onRetry={lookup.refresh} />}
          {lookup.data && <TransactionCard transaction={lookup.data} />}
          {!transactionId && <p className="muted hint-line">Search a real record by transaction identifier. IDs are lookup keys, not model features.</p>}
        </Panel>
        <Panel title="Hypothetical prediction" subtitle="No dataset record or graph relationship is created">
          <div className="scope-banner compact"><Sparkles size={15} /><span>Enter the model's post-transaction features. This is a hypothetical review score.</span></div>
          <form className="prediction-form" onSubmit={submitPrediction}>
            <div className="form-grid">
              <label>Simulation step<input name="step" type="number" min="1" step="1" required defaultValue="400" /></label>
              <label>Transaction amount<input name="amount" type="number" min="0" step="any" required /></label>
              <label>Transaction type<select name="type" required defaultValue=""><option value="" disabled>Select type</option><option>PAYMENT</option><option>TRANSFER</option><option>CASH_OUT</option><option>CASH_IN</option><option>DEBIT</option></select></label>
              {BALANCE_FIELDS.map(([name, label]) => <label key={name}>{label}<input name={name} type="number" min="0" step="any" placeholder="Optional" /></label>)}
            </div>
            <p className="subtle-note">Blank or invalid balance values use the saved pipeline's training-fitted imputation. Balance fields are available only after transaction posting.</p>
            <button className="button button-primary" type="submit" disabled={predicting}>{predicting ? "Scoring…" : "Score hypothetical transaction"}<ArrowRight size={15} /></button>
          </form>
          {predictError && <div className="inline-error"><AlertCircle size={16} />{predictError}</div>}
          {prediction && <div className={`prediction-result ${prediction.predicted_class ? "prediction-flagged" : ""}`}>
            <div className="result-heading"><div><p className="eyebrow">HYPOTHETICAL MODEL OUTPUT</p><h3>{prediction.predicted_class ? "Candidate for investigation" : "Not flagged at the configured threshold"}</h3></div><ScoreBadge score={prediction.score} flagged={Boolean(prediction.predicted_class)} /></div>
            <p>Uncalibrated positive-class score <strong>{prediction.score.toFixed(4)}</strong>; threshold <strong>{prediction.threshold.toFixed(4)}</strong>.</p>
            <p className="subtle-note">This score is not a calibrated fraud probability or a confirmed fraud finding. No graph link was fabricated.</p>
          </div>}
        </Panel>
      </div>
    </>
  );
}

function TransactionCard({ transaction }: { transaction: Transaction }) {
  return (
    <div className="transaction-detail">
      <div className="result-heading"><div><p className="eyebrow">INDEXED RECORD</p><h3 className="mono">{transaction.transaction_id}</h3></div><ScoreBadge score={transaction.prediction.score} flagged={Boolean(transaction.prediction.predicted_class)} /></div>
      <div className="detail-grid">
        <div><span>Amount units</span><strong>{formatAmount(transaction.amount)}</strong></div>
        <div><span>Type</span><strong>{transaction.type}</strong></div>
        <div><span>Simulation step</span><strong>{formatNumber(transaction.step)}</strong></div>
        <div><span>Source row</span><strong>{formatNumber(transaction.source_row_index)}</strong></div>
        <div><span>Sender account</span><strong className="mono">{transaction.sender}</strong></div>
        <div><span>Recipient account</span><strong className="mono">{transaction.recipient}</strong></div>
      </div>
      <p className="subtle-note">{transaction.prediction.score_interpretation}; not a calibrated probability. Simulation ground-truth labels are not included in operational record details.</p>
    </div>
  );
}
