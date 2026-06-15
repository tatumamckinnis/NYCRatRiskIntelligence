"use client";

import type { WeekForecast } from "@/lib/types";

interface ForecastChartProps {
  forecasts: WeekForecast[];
  currentScore: number;
}

export function ForecastChart({ forecasts, currentScore }: ForecastChartProps) {
  if (!forecasts.length) {
    return (
      <div className="flex items-center justify-center h-40 text-sm text-muted-foreground">
        No forecast data available
      </div>
    );
  }

  const allScores = forecasts.flatMap((f) => [f.risk_score, f.ci_low, f.ci_high]);
  const minScore = Math.min(currentScore, ...allScores, 0);
  const maxScore = Math.max(currentScore, ...allScores, 0.2);
  const range = maxScore - minScore || 0.1;

  const width = 400;
  const height = 120;
  const padLeft = 32;
  const padRight = 8;
  const padTop = 8;
  const padBottom = 24;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const n = forecasts.length;

  const xOf = (i: number) => padLeft + (i / (n - 1)) * chartW;
  const yOf = (v: number) =>
    padTop + chartH - ((v - minScore) / range) * chartH;

  const linePoints = forecasts
    .map((f, i) => `${xOf(i)},${yOf(f.risk_score)}`)
    .join(" ");

  const areaPoints = [
    ...forecasts.map((f, i) => `${xOf(i)},${yOf(f.ci_high)}`),
    ...forecasts
      .slice()
      .reverse()
      .map((f, i) => `${xOf(n - 1 - i)},${yOf(f.ci_low)}`),
  ].join(" ");

  // Y-axis ticks
  const ticks = [0, 0.25, 0.5, 0.75, 1].filter(
    (t) => t >= minScore - 0.05 && t <= maxScore + 0.05
  );

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        role="img"
        aria-label="12-week rat risk forecast chart"
      >
        {/* Grid lines */}
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={padLeft}
              y1={yOf(t)}
              x2={width - padRight}
              y2={yOf(t)}
              stroke="currentColor"
              strokeOpacity={0.1}
              strokeWidth={1}
            />
            <text
              x={padLeft - 4}
              y={yOf(t) + 4}
              textAnchor="end"
              fontSize={9}
              fill="currentColor"
              fillOpacity={0.5}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {/* CI band */}
        <polygon
          points={areaPoints}
          fill="#f97316"
          fillOpacity={0.15}
        />

        {/* Forecast line */}
        <polyline
          points={linePoints}
          fill="none"
          stroke="#f97316"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Current week marker */}
        <line
          x1={padLeft}
          y1={yOf(currentScore)}
          x2={padLeft + 12}
          y2={yOf(currentScore)}
          stroke="#dc2626"
          strokeWidth={2}
          strokeDasharray="3,2"
        />

        {/* X-axis labels (first, middle, last) */}
        {[0, Math.floor(n / 2), n - 1].map((i) => (
          <text
            key={i}
            x={xOf(i)}
            y={height - 4}
            textAnchor="middle"
            fontSize={8}
            fill="currentColor"
            fillOpacity={0.5}
          >
            {forecasts[i].week.slice(5)} {/* MM-DD */}
          </text>
        ))}
      </svg>
      <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-orange-400" />
          Forecast
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 bg-orange-400/20 rounded-sm" />
          95% CI
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-red-500" style={{ borderTop: "2px dashed" }} />
          Current
        </span>
      </div>
    </div>
  );
}
