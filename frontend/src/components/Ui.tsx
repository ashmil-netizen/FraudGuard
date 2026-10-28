import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, LoaderCircle, RefreshCw } from "lucide-react";
import type { PropsWithChildren, ReactNode } from "react";

export function PageTitle({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="muted page-description">{description}</p>
      </div>
      {action}
    </div>
  );
}

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: PropsWithChildren<{
  title: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
}>) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="muted">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function MetricCard({
  label,
  value,
  detail,
  icon,
  tone = "violet",
}: {
  label: string;
  value: ReactNode;
  detail: string;
  icon: ReactNode;
  tone?: "violet" | "cyan" | "red" | "green";
}) {
  const reducedMotion = useReducedMotion();
  return (
    <motion.article
      className={`metric-card tone-${tone}`}
      initial={reducedMotion ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={reducedMotion ? undefined : { y: -3 }}
      transition={{ duration: 0.25 }}
    >
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </motion.article>
  );
}

export function LoadingState({ label = "Loading live data" }: { label?: string }) {
  return (
    <div className="state-box" role="status">
      <LoaderCircle className="spin" size={20} aria-hidden="true" />
      <span>{label}</span>
      <div className="skeleton-line" />
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="state-box error-box" role="alert">
      <AlertTriangle size={20} aria-hidden="true" />
      <div>
        <strong>Live data unavailable</strong>
        <p>{message}</p>
      </div>
      {onRetry && (
        <button className="icon-button" onClick={onRetry} aria-label="Retry request">
          <RefreshCw size={16} />
        </button>
      )}
    </div>
  );
}

export function EmptyState({ children }: PropsWithChildren) {
  return <div className="empty-state">{children}</div>;
}

export function ScoreBadge({ score, flagged }: { score: number; flagged: boolean }) {
  return (
    <span className={`score-badge ${flagged ? "score-high" : "score-low"}`}>
      {flagged ? "Candidate" : "Not flagged"} · {score.toFixed(3)}
    </span>
  );
}

export function formatNumber(value: number | undefined | null): string {
  return value == null ? "Unavailable" : new Intl.NumberFormat("en-US").format(value);
}

export function formatAmount(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

export function Percent({ value }: { value: number | null | undefined }) {
  return <>{value == null ? "Unavailable" : `${(value * 100).toFixed(1)}%`}</>;
}
