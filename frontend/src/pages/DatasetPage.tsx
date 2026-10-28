import { useResource } from "../hooks";
import { api } from "../api";
import {
  EmptyState,
  ErrorState,
  formatNumber,
  LoadingState,
  MetricCard,
  PageTitle,
  Panel,
  Percent,
} from "../components/Ui";
import { Activity, Database, Gauge, Layers3 } from "lucide-react";

export default function DatasetPage() {
  const dataset = useResource((signal) => api.dataset(signal), []);
  const model = useResource((signal) => api.model(signal), []);
  const dashboard = useResource((signal) => api.dashboard(signal), []);

  return (
    <>
      <PageTitle eyebrow="METHODOLOGY & PROVENANCE" title="Dataset and model" description="Verified PaySim schema, saved classifier configuration, evaluation methodology, and scope limitations." />
      <div className="metrics-grid">
        {dataset.data && <>
          <MetricCard label="Dataset rows" value={formatNumber(dataset.data.rows)} detail="Full inspected source dimensions" icon={<Database size={18} />} />
          <MetricCard label="Source columns" value={formatNumber(dataset.data.columns_count)} detail="PaySim simulation CSV" icon={<Layers3 size={18} />} tone="cyan" />
          <MetricCard label="Scored API records" value={formatNumber(dashboard.data?.transactions_available)} detail="Chronological held-out test split only" icon={<Activity size={18} />} tone="green" />
          <MetricCard label="Classification threshold" value={model.data?.classification_threshold.toFixed(4) ?? "Unavailable"} detail="Selected on validation split" icon={<Gauge size={18} />} tone="violet" />
        </>}
      </div>
      <div className="dashboard-grid">
        <Panel title="Simulation labels vs model predictions" subtitle="Distinct concepts with separate scopes">
          {dataset.loading || dashboard.loading ? <LoadingState /> : dataset.error ? <ErrorState message={dataset.error} onRetry={dataset.refresh} /> : dashboard.error ? <ErrorState message={dashboard.error} onRetry={dashboard.refresh} /> : dataset.data && dashboard.data && <>
            <div className="evaluation-grid">
              <div><span>Full dataset positive simulation labels</span><strong>{formatNumber(Number(dataset.data.ground_truth_labels.counts?.find(([label]) => label === "1")?.[1] ?? 0))}</strong></div>
              <div><span>Model-predicted candidates in scored split</span><strong>{formatNumber(dashboard.data.model_predicted_candidates)}</strong></div>
              <p className="subtle-note full-span">The first is an offline PaySim simulation-label count for the full source file. The second is the classifier's prediction count only in the held-out test split. Neither is confirmed real-world fraud.</p>
            </div>
          </>}
        </Panel>
        <Panel title="PaySim dataset" subtitle={dataset.data?.dataset_identifier ?? "Dataset summary"}>
          {dataset.loading ? <LoadingState /> : dataset.error ? <ErrorState message={dataset.error} onRetry={dataset.refresh} /> : dataset.data && <>
            <p className="subtle-note">{dataset.data.dataset_kind}. Source dimensions are full-dataset facts; transaction and graph endpoints serve scored test data only.</p>
            <div className="table-scroll"><table><thead><tr><th>Field</th><th>Observed type</th><th>Missing</th><th>Role</th></tr></thead><tbody>{dataset.data.columns.map((column) => <tr key={column.name}><td className="mono">{column.name}</td><td>{column.data_types.join(", ")}</td><td>{formatNumber(column.missing_values)} ({column.missing_percentage.toFixed(2)}%)</td><td>{column.identifier ? "Identifier" : column.name === "isFraud" ? "Simulation label" : column.name === "isFlaggedFraud" ? "Existing rule flag" : "Transaction field"}</td></tr>)}</tbody></table></div>
            <div className="info-strip"><strong>Transaction types:</strong> {dataset.data.transaction_types.join(", ")}</div>
          </>}
        </Panel>
        <Panel title="Saved classifier" subtitle={model.data?.model_type ?? "Model summary"}>
          {model.loading ? <LoadingState /> : model.error ? <ErrorState message={model.error} onRetry={model.refresh} /> : model.data && <>
            <dl className="definition-list"><dt>Model type</dt><dd>{model.data.model_type}</dd><dt>Feature schema</dt><dd className="schema-chips">{model.data.feature_schema.map((feature) => <span key={feature}>{feature}</span>)}</dd><dt>Classification threshold</dt><dd>{model.data.classification_threshold.toFixed(6)}</dd><dt>Threshold source</dt><dd>{model.data.threshold_selection}</dd><dt>Training split</dt><dd>{formatNumber(model.data.split_strategy.training_rows_used)} sampled rows of {formatNumber(model.data.split_strategy.training_rows_available)} available</dd><dt>Validation rows</dt><dd>{formatNumber(model.data.split_strategy.validation_rows)}</dd><dt>Scored test rows</dt><dd>{formatNumber(model.data.split_strategy.test_rows)}</dd></dl>
            <div className="subtle-note">{model.data.score_interpretation}</div>
          </>}
        </Panel>
        <Panel title="Offline evaluation" subtitle="Metrics compare model predictions with simulated PaySim labels on the later test period">
          {dashboard.loading ? <LoadingState /> : dashboard.error ? <ErrorState message={dashboard.error} onRetry={dashboard.refresh} /> : dashboard.data?.evaluation_metrics ? <div className="evaluation-grid">
            <div><span>Precision</span><strong><Percent value={dashboard.data.evaluation_metrics.precision} /></strong></div>
            <div><span>Recall</span><strong><Percent value={dashboard.data.evaluation_metrics.recall} /></strong></div>
            <div><span>F1</span><strong>{dashboard.data.evaluation_metrics.f1.toFixed(4)}</strong></div>
            <div><span>Average precision</span><strong>{dashboard.data.evaluation_metrics.average_precision?.toFixed(4) ?? "Unavailable"}</strong></div>
            <div><span>ROC-AUC</span><strong>{dashboard.data.evaluation_metrics.roc_auc?.toFixed(4) ?? "Unavailable"}</strong></div>
            <div><span>Predicted candidate count</span><strong>{formatNumber(dashboard.data.model_predicted_candidates)}</strong></div>
            <p className="subtle-note full-span">Predicted candidates are not the same as the positive simulation-label count. These metrics do not establish effectiveness on real financial activity.</p>
          </div> : <EmptyState>Evaluation metrics are unavailable.</EmptyState>}
        </Panel>
        <Panel title="Method and limitations" subtitle="Interpret results within the simulation's scope">
          {model.loading ? <LoadingState /> : model.error ? <ErrorState message={model.error} onRetry={model.refresh} /> : model.data && <ul className="limitation-list">{model.data.limitations.map((item) => <li key={item}>{item}</li>)}<li>Model inputs include post-transaction balances; use is limited to after-posting review, not pre-transaction authorization.</li><li>Account IDs and transaction IDs are lookup/graph keys and are not classifier features.</li><li>Transaction review statuses, notes, and CSV export are not supported by this API.</li></ul>}
        </Panel>
      </div>
    </>
  );
}
