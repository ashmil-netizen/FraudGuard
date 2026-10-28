import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import type { GraphData, GraphEdge, GraphNode } from "../types";

interface CanvasNode extends GraphNode {
  id: string;
  highlighted?: boolean;
}

interface CanvasLink extends Omit<GraphEdge, "source" | "target"> {
  source: string | number | CanvasNode;
  target: string | number | CanvasNode;
  highlighted?: boolean;
}

interface CanvasData {
  nodes: CanvasNode[];
  links: CanvasLink[];
}

function graphLink(edge: GraphEdge, highlighted: boolean): CanvasLink {
  const attributes = edge.attributes ?? {};
  const score = attributes.predicted_fraud_score;
  const predicted = attributes.predicted_fraud;
  const threshold = attributes.classification_threshold;
  const prediction =
    edge.prediction ??
    (typeof score === "number" &&
    typeof predicted === "boolean" &&
    typeof threshold === "number"
      ? {
          predicted_class: predicted ? 1 : 0,
          score,
          threshold,
          score_interpretation: "uncalibrated positive-class model score",
        }
      : undefined);

  return {
    ...edge,
    type: edge.type ?? (typeof attributes.type === "string" ? attributes.type : undefined),
    amount: edge.amount ?? (typeof attributes.amount === "number" ? attributes.amount : undefined),
    step: edge.step ?? (typeof attributes.step === "number" ? attributes.step : undefined),
    prediction,
    source: edge.source,
    target: edge.target,
    highlighted,
  };
}

export function GraphCanvas({
  data,
  onNodeSelect,
  onEdgeSelect,
  highlightedNodes = [],
  highlightedEdges = [],
  height = 490,
}: {
  data: GraphData;
  onNodeSelect?: (node: GraphNode) => void;
  onEdgeSelect?: (edge: GraphEdge) => void;
  highlightedNodes?: string[];
  highlightedEdges?: string[];
  height?: number;
}) {
  const graphRef = useRef<ForceGraphMethods<CanvasNode, CanvasLink> | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const nodeHighlightSet = useMemo(() => new Set(highlightedNodes), [highlightedNodes]);
  const edgeHighlightSet = useMemo(() => new Set(highlightedEdges), [highlightedEdges]);
  const graphData = useMemo<CanvasData>(() => ({
    nodes: data.nodes.map((node) => ({ ...node, id: node.id, highlighted: nodeHighlightSet.has(node.id) })),
    links: data.edges.map((edge) =>
      graphLink(edge, edgeHighlightSet.has(edge.transaction_id)),
    ),
  }), [data, edgeHighlightSet, nodeHighlightSet]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="graph-canvas-wrap" ref={containerRef} style={{ height }}>
      <ForceGraph2D<CanvasNode, CanvasLink>
        ref={graphRef}
        graphData={graphData}
        width={width || 600}
        height={height}
        backgroundColor="#080d18"
        nodeId="id"
        nodeRelSize={3}
        nodeVal={(node) => node.highlighted ? 4 : 2}
        nodeColor={(node) => node.highlighted ? "#ffffff" : "#22d3ee"}
        nodeLabel={(node) => `Account ${node.id}${node.depth !== undefined ? ` · depth ${node.depth}` : ""}`}
        linkLabel={(link) => `${link.transaction_id} · ${link.type ?? "transaction"} · ${link.amount == null ? "" : link.amount.toLocaleString()}`}
        linkColor={(link) => link.highlighted ? "#ffffff" : link.prediction?.predicted_class ? "#f87171" : "#475569"}
        linkWidth={(link) => link.highlighted ? 2.4 : link.prediction?.predicted_class ? 1.5 : 0.8}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={(link) => link.highlighted ? "#ffffff" : link.prediction?.predicted_class ? "#f87171" : "#64748b"}
        cooldownTicks={80}
        onNodeClick={(node) => onNodeSelect?.({ id: node.id })}
        onLinkClick={(link) => onEdgeSelect?.({
          ...link,
          source: typeof link.source === "object" ? String(link.source.id) : String(link.source),
          target: typeof link.target === "object" ? String(link.target.id) : String(link.target),
        })}
        enableNodeDrag
        enableZoomInteraction
        enablePanInteraction
        d3AlphaDecay={0.035}
      />
      <div className="graph-controls">
        <button className="icon-button" aria-label="Fit graph to screen" onClick={() => graphRef.current?.zoomToFit(450, 36)} title="Fit to screen">Fit</button>
        <button className="icon-button" aria-label="Reset graph view" onClick={() => { graphRef.current?.centerAt(0, 0, 450); graphRef.current?.zoom(1, 450); }} title="Reset view">Reset</button>
      </div>
    </div>
  );
}
