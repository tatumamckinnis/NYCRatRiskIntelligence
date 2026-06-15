import type { RiskFactor } from "@/lib/types";

interface FactorBarProps {
  factors: RiskFactor[];
}

export function FactorBar({ factors }: FactorBarProps) {
  if (!factors.length) {
    return (
      <div className="text-sm text-muted-foreground">No factor data</div>
    );
  }

  const maxContrib = Math.max(...factors.map((f) => Math.abs(f.contribution)));

  return (
    <ul className="space-y-2" aria-label="Top risk factors">
      {factors.map((f) => {
        const pct = Math.round((Math.abs(f.contribution) / maxContrib) * 100);
        const isUp = f.direction === "up";
        return (
          <li key={f.feature} className="flex flex-col gap-0.5">
            <div className="flex justify-between text-xs">
              <span className="text-foreground font-medium truncate max-w-[70%]">
                {f.readable}
              </span>
              <span
                className={isUp ? "text-red-500" : "text-green-600"}
                aria-label={`${isUp ? "increases" : "decreases"} risk`}
              >
                {isUp ? "▲" : "▼"} {Math.abs(f.contribution).toFixed(3)}
              </span>
            </div>
            <div className="h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  isUp ? "bg-red-400" : "bg-green-500"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
