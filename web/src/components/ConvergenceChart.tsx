import { useMemo } from "react";

export interface ChartPoint {
  x: number;
  y: number;
}

interface Props {
  points: ChartPoint[];
  xLabel: string;
  yLabel: string;
  width?: number;
  height?: number;
}

const PAD = { top: 12, right: 16, bottom: 32, left: 72 };

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(3);
  return Number(v.toPrecision(6)).toString();
}

/**
 * 纯 SVG 折线图（opt 步-能量收敛曲线 / scan 坐标-能量点线，PLAN §2.6）。
 * 不引图表库。
 */
export default function ConvergenceChart({
  points,
  xLabel,
  yLabel,
  width = 520,
  height = 260,
}: Props) {
  const geom = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (yMin === yMax) {
      yMin -= 1e-6;
      yMax += 1e-6;
    }
    const iw = width - PAD.left - PAD.right;
    const ih = height - PAD.top - PAD.bottom;
    const sx = (x: number) =>
      PAD.left + (xMax === xMin ? iw / 2 : ((x - xMin) / (xMax - xMin)) * iw);
    const sy = (y: number) => PAD.top + ih - ((y - yMin) / (yMax - yMin)) * ih;
    const path = points
      .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`)
      .join(" ");
    const yTicks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);
    return { path, sx, sy, xMin, xMax, yMin, yMax, yTicks, ih, iw };
  }, [points, width, height]);

  if (!geom) return null;

  const xTicks = Array.from({ length: 6 }, (_, i) =>
    geom.xMin + ((geom.xMax - geom.xMin) * i) / 5,
  );

  return (
    <svg
      className="conv-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${xLabel} - ${yLabel} 折线图`}
    >
      {/* 网格 + Y 轴刻度 */}
      {geom.yTicks.map((t) => (
        <g key={t}>
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={geom.sy(t)}
            y2={geom.sy(t)}
            className="grid"
          />
          <text x={PAD.left - 6} y={geom.sy(t) + 4} textAnchor="end" className="tick">
            {fmt(t)}
          </text>
        </g>
      ))}
      {/* X 轴刻度 */}
      {xTicks.map((t) => (
        <text
          key={t}
          x={geom.sx(t)}
          y={height - PAD.bottom + 16}
          textAnchor="middle"
          className="tick"
        >
          {fmt(t)}
        </text>
      ))}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={height - PAD.bottom} className="axis" />
      <line x1={PAD.left} y1={height - PAD.bottom} x2={width - PAD.right} y2={height - PAD.bottom} className="axis" />
      <path d={geom.path} className="line" />
      {points.map((p, i) => (
        <circle key={i} cx={geom.sx(p.x)} cy={geom.sy(p.y)} r={3} className="pt">
          <title>{`(${fmt(p.x)}, ${fmt(p.y)})`}</title>
        </circle>
      ))}
      <text x={PAD.left + geom.iw / 2} y={height - 4} textAnchor="middle" className="axis-label">
        {xLabel}
      </text>
      <text
        x={14}
        y={PAD.top + geom.ih / 2}
        textAnchor="middle"
        className="axis-label"
        transform={`rotate(-90 14 ${PAD.top + geom.ih / 2})`}
      >
        {yLabel}
      </text>
    </svg>
  );
}
