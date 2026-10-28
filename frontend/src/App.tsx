import { lazy, Suspense, useState } from "react";
import {
  Activity,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Database,
  GitFork,
  Layers3,
  Menu,
  Search,
  ShieldAlert,
  ShieldCheck,
  Waypoints,
  X,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import { ErrorState, LoadingState } from "./components/Ui";
import { useResource } from "./hooks";

const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const AnalyzerPage = lazy(() => import("./pages/AnalyzerPage"));
const GraphPage = lazy(() => import("./pages/GraphPage"));
const InvestigationPage = lazy(() => import("./pages/InvestigationPage"));
const QueuePage = lazy(() => import("./pages/QueuePage"));
const DatasetPage = lazy(() => import("./pages/DatasetPage"));

const navigation = [
  { to: "/", label: "Overview", icon: ChartNoAxesCombined, end: true },
  { to: "/analyzer", label: "Transaction analyzer", icon: Search },
  { to: "/graph", label: "Graph explorer", icon: Waypoints },
  { to: "/investigation", label: "BFS / DFS lab", icon: GitFork },
  { to: "/queue", label: "Candidate queue", icon: ShieldAlert },
  { to: "/dataset", label: "Dataset & model", icon: Database },
];

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const reducedMotion = useReducedMotion();
  const health = useResource((signal) => api.health(signal), []);
  const isReady = health.data?.status === "ready";

  const sidebar = (
    <aside className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}>
      <div className="brand-row">
        <div className="brand-mark"><ShieldCheck size={21} /></div>
        {!collapsed && <div className="brand-copy"><strong>FraudGuard</strong><span>AI INTELLIGENCE</span></div>}
        <button
          className="icon-button collapse-button"
          onClick={() => setCollapsed((value) => !value)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
      </div>
      {!collapsed && <p className="nav-caption">WORKSPACE</p>}
      <nav className="main-nav" aria-label="Main navigation">
        {navigation.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            title={collapsed ? label : undefined}
            className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}
            onClick={() => setMobileOpen(false)}
          >
            <Icon size={18} aria-hidden="true" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-note">
          <div className="note-icon"><Layers3 size={16} /></div>
          {!collapsed && <div><strong>PaySim test split</strong><small>Simulated transaction data</small></div>}
        </div>
        {!collapsed && <div className="sidebar-version">FraudGuard AI <span>v1.0</span></div>}
      </div>
    </aside>
  );

  return (
    <div className={`app-shell ${collapsed ? "shell-collapsed" : ""}`}>
      {sidebar}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            className="mobile-sidebar-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setMobileOpen(false)}
          >
            <div className="mobile-sidebar" onClick={(event) => event.stopPropagation()}>
              <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
                <X size={20} />
              </button>
              {sidebar}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <main className="main-shell">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <Menu size={20} />
          </button>
          <div className="breadcrumb"><span>FraudGuard AI</span><span className="crumb-slash">/</span><strong>{navigation.find((item) => item.to === location.pathname)?.label ?? "Overview"}</strong></div>
          <div className="topbar-right">
            <span className={`service-status ${isReady ? "status-ready" : "status-warn"}`}>
              <i />{health.loading ? "Checking API" : isReady ? "API online" : "API unavailable"}
            </span>
            <span className="topbar-dataset"><Activity size={15} /> PaySim</span>
            <CircleHelp className="help-icon" size={18} aria-label="Academic analysis dashboard" />
          </div>
        </header>
        <div className="content-area">
          <Suspense fallback={<LoadingState label="Opening workspace" />}>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={location.pathname}
                initial={reducedMotion ? false : { opacity: 0, y: 7 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reducedMotion ? undefined : { opacity: 0, y: -4 }}
                transition={{ duration: 0.18 }}
              >
                {health.error && <div className="global-api-alert"><ErrorState message={health.error} onRetry={health.refresh} /></div>}
                <Routes location={location}>
                  <Route path="/" element={<OverviewPage />} />
                  <Route path="/analyzer" element={<AnalyzerPage />} />
                  <Route path="/graph" element={<GraphPage />} />
                  <Route path="/investigation" element={<InvestigationPage />} />
                  <Route path="/queue" element={<QueuePage />} />
                  <Route path="/dataset" element={<DatasetPage />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </motion.div>
            </AnimatePresence>
          </Suspense>
        </div>
      </main>
    </div>
  );
}
