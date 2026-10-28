import { useMemo } from "react";
import { Activity, AlertOctagon, ArrowUpRight, RefreshCw, ShieldCheck, Users } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ChartFrame } from "../components/ChartTooltip";
import {
  EmptyState,
  ErrorState,
  formatAmount,
  formatNumber,
  LoadingState,
  MetricCard,
  PageTitle,
  Panel,
  Percent,
  ScoreBadge,
} from "../components/Ui";
import { useResource } from "../hooks";

const COLORS = ["#8b5cf6", "#22d3ee", "#34d399", "#fbbf24", "#f87171"];

export default function OverviewPage() {
  const dashboard = useResource((signal) => api.dashboard(signal), []);
  const candidates = useResource(
    (signal) => api.transactions({ page: 1, page_size: 6, predicted: true }, signal),
    [],
  );
  const typeData = useMemo(() => dashboard.data?.transaction_type_distribution ?? [], [dashboard.data]);
  const amountData = useMemo(() => dashboard.data?.amount_distribution ?? [], [dashboard.data]);
  const metrics = dashboard.data?.evaluation_metrics;
  const predictions = dashboard.data
    ? [
        { name: "Flagged candidates", value: dashboard.data.model_predicted_candidates },
        { name: "Not flagged", value: Math.max(0, dashboard.data.transactions_available - dashboard.data.model_predicted_candidates) },
      ]
    : [];

  return (
    <>
      <PageTitle
        eyebrow="FINANCIAL INTELLIGENCE · PAYSIM"
        title="Transaction overview"
        description="A clear view of model-scored PaySim test transactions and investigation activity."
        action={<button className="button button-secondary" onClick={() => { dashboard.refresh(); candidates.refresh(); }}><RefreshCw size={15} /> Refresh data</button>}
      />
      {dashboard.loading ? <LoadingState /> : dashboard.error ? <ErrorState message={dashboard.error} onRetry={dashboard.refresh} /> : dashboard.data && (
        <>
          <div className="scope-banner"><Activity size={16} /><span><strong>Scope:</strong> {dashboard.data.scope}. Predicted candidates and offline simulation-label evaluation are separate measures.</span></div>
          <div className="metrics-grid">
            <MetricCard label="Scored transactions" value={formatNumber(dashboard.data.transactions_available)} detail="Held-out chronological test period" icon={<Activity size={18} />} />
            <MetricCard label="Accounts represented" value={formatNumber(dashboard.data.accounts_represented)} detail="Sender or recipient in test-split records" icon={<Users size={18} />} tone="cyan" />
            <MetricCard label="Predicted candidates" value={formatNumber(dashboard.data.model_predicted_candidates)} detail="Model prediction at saved threshold" icon={<AlertOctagon size={18} />} tone="red" />
            <MetricCard label="Test precision" value={metrics ? <Percent value={metrics.precision} /> : "Unavailable"} detail="Against PaySim simulation labels" icon={<ShieldCheck size={18} />} tone="green" />
          </div>
          <div className="dashboard-grid">
            <Panel title="Prediction distribution" subtitle="Model outputs on scored test transactions">
              <ChartFrame>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={predictions} dataKey="value" nameKey="name" innerRadius={66} outerRadius={94} paddingAngle={4} stroke="none">
                      {predictions.map((_entry, index) => <Cell key={index} fill={index === 0 ? "#f87171" : "#334155"} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: "#111827", border: "1px solid #293449", borderRadius: 10 }} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="chart-center"><strong>{formatNumber(dashboard.data.model_predicted_candidates)}</strong><span>candidates</span></div>
              </ChartFrame>
              <div className="chart-legend"><span><i className="dot dot-red" />Predicted candidates</span><span><i className="dot dot-slate" />Not flagged</span></div>
            </Panel>
            <Panel title="Transaction types" subtitle="Counts from the scored test-split records">
              <ChartFrame>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={typeData} margin={{ top: 8, right: 12, left: -18, bottom: 4 }}>
                    <CartesianGrid stroke="#253044" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="type" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#111827", border: "1px solid #293449", borderRadius: 10 }} />
                    <Bar dataKey="count" radius={[5, 5, 0, 0]}>{typeData.map((_entry, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </Panel>
            <Panel title="Transaction amount bands" subtitle="PaySim amount units; the source does not establish a currency">
              <ChartFrame>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={amountData} margin={{ top: 8, right: 12, left: -18, bottom: 4 }}>
                    <CartesianGrid stroke="#253044" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="range" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#111827", border: "1px solid #293449", borderRadius: 10 }} />
                    <Bar dataKey="count" fill="#22d3ee" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </Panel>
            <Panel title="Offline evaluation snapshot" subtitle="Measured on later chronological test data with simulation labels">
              {metrics ? (
                <div className="evaluation-grid">
                  <div><span>Recall</span><strong><Percent value={metrics.recall} /></strong></div>
                  <div><span>F1 score</span><strong>{metrics.f1.toFixed(3)}</strong></div>
                  <div><span>Average precision</span><strong>{metrics.average_precision?.toFixed(3) ?? "Unavailable"}</strong></div>
                  <div><span>ROC-AUC</span><strong>{metrics.roc_auc?.toFixed(3) ?? "Unavailable"}</strong></div>
                  <p className="subtle-note full-span">PaySim labels are simulated, not independently adjudicated real-world fraud.</p>
                </div>
              ) : <EmptyState>Evaluation metrics are not available from the API.</EmptyState>}
            </Panel>
          </div>
          <Panel title="Latest predicted candidates" subtitle="Most recent source-row records in the scored test split" action={<Link className="text-link" to="/queue">Open queue <ArrowUpRight size={14} /></Link>}>
            {candidates.loading ? <LoadingState label="Loading candidate records" /> : candidates.error ? <ErrorState message={candidates.error} onRetry={candidates.refresh} /> : candidates.data?.items.length ? (
              <div className="table-scroll"><table><thead><tr><th>Transaction</th><th>Type</th><th>Amount</th><th>Time step</th><th>Prediction</th><th /></tr></thead><tbody>
                {candidates.data.items.map((row) => <tr key={row.transaction_id}><td className="mono">{row.transaction_id}</td><td>{row.type}</td><td>{formatAmount(row.amount)} units</td><td>{formatNumber(row.step)}</td><td><ScoreBadge score={row.prediction.score} flagged /></td><td><Link className="table-link" to={`/analyzer?id=${encodeURIComponent(row.transaction_id)}`}>Inspect</Link></td></tr>)}
              </tbody></table></div>
            ) : <EmptyState>No candidate records are available in this test split.</EmptyState>}
          </Panel>
          <p className="footnote">Candidate records are model outputs for review, not confirmed fraud. Counts shown here reflect actual scored records and may differ from ground-truth simulation-label counts.</p>
        </>
      )}
    </>
  );
}
