import type { PropsWithChildren } from "react";

export function ChartFrame({ children }: PropsWithChildren) {
  return <div className="chart-frame">{children}</div>;
}
