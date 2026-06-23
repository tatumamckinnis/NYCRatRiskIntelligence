"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Nav } from "@/components/nav";
import { Badge } from "@/components/ui/badge";
import { getMapRisk } from "@/lib/api";

// Lazy-load map (uses browser APIs not available in SSR)
const RiskMap = dynamic(
  () => import("@/components/map/RiskMap").then((m) => m.RiskMap),
  { ssr: false, loading: () => <div className="flex-1 bg-muted/30 rounded-lg animate-pulse" /> }
);

// Latest week with materialized predictions in the DB
const LATEST_DATA_WEEK = "2026-05-11";

// Generate ISO Mondays from a given start date up to LATEST_DATA_WEEK
function weeksUpTo(latest: string): string[] {
  const end = new Date(latest);
  const weeks: string[] = [];
  const d = new Date("2023-05-22"); // earliest panel week
  while (d <= end) {
    weeks.push(d.toISOString().slice(0, 10));
    d.setDate(d.getDate() + 7);
  }
  return weeks;
}

export default function HomePage() {
  const weeks = useMemo(() => weeksUpTo(LATEST_DATA_WEEK), []);
  const [selectedWeek, setSelectedWeek] = useState(LATEST_DATA_WEEK);

  const { data: mapItems = [], isLoading, isError } = useQuery({
    queryKey: ["map-risk", selectedWeek],
    queryFn: () => getMapRisk(selectedWeek),
    retry: 2,
  });

  const decileDistrib = useMemo(() => {
    const counts = Array(10).fill(0) as number[];
    mapItems.forEach((item) => counts[item.risk_decile - 1]++);
    return counts;
  }, [mapItems]);

  const highRisk = mapItems.filter((i) => i.risk_decile >= 8).length;

  const topNeighborhoods = useMemo(
    () =>
      [...mapItems]
        .sort((a, b) => b.risk_score - a.risk_score)
        .slice(0, 5),
    [mapItems]
  );

  return (
    <div className="flex flex-col h-screen">
      <Nav />

      <main className="flex flex-1 min-h-0 gap-0">
        {/* Sidebar */}
        <aside className="w-72 border-r bg-muted/30 flex flex-col p-4 gap-4 overflow-y-auto shrink-0">
          <div>
            <h1 className="text-lg font-semibold leading-tight">
              NYC Rat Risk Map
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Neighborhood-level rodent risk predictions updated weekly.
            </p>
          </div>

          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Week
            </div>
            <div className="text-sm font-mono">{selectedWeek}</div>
          </div>

          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-4 bg-muted rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <>
              <div className="space-y-2">
                <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Summary
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="border rounded p-2 bg-background">
                    <div className="text-muted-foreground">NTAs mapped</div>
                    <div className="text-lg font-semibold">{mapItems.length}</div>
                  </div>
                  <div className="border rounded p-2 bg-background">
                    <div className="text-muted-foreground">High risk (D8+)</div>
                    <div className="text-lg font-semibold text-red-500">
                      {highRisk}
                    </div>
                  </div>
                </div>
              </div>

              {topNeighborhoods.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                    Highest Risk This Week
                  </div>
                  <ol className="space-y-1">
                    {topNeighborhoods.map((item, i) => (
                      <li
                        key={item.nta_id}
                        className="flex items-center justify-between text-xs cursor-pointer hover:text-foreground text-muted-foreground transition-colors"
                        onClick={() => window.location.href = `/nta/${item.nta_id}`}
                      >
                        <span className="flex items-center gap-1.5">
                          <span className="w-4 text-right tabular-nums opacity-50">{i + 1}.</span>
                          <span className="truncate max-w-[140px]">{item.nta_name ?? item.nta_id}</span>
                        </span>
                        <span
                          className="font-mono font-semibold shrink-0"
                          style={{
                            color: item.risk_score > 0.7 ? "#dc2626" : item.risk_score > 0.4 ? "#f97316" : "#fbbf24",
                          }}
                        >
                          D{item.risk_decile}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              <div>
                <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                  Decile Distribution
                </div>
                <div className="flex gap-0.5 items-end h-12">
                  {decileDistrib.map((count, i) => {
                    const max = Math.max(...decileDistrib, 1);
                    const pct = (count / max) * 100;
                    return (
                      <div
                        key={i}
                        title={`Decile ${i + 1}: ${count} NTAs`}
                        className="flex-1 rounded-t-sm"
                        style={{
                          height: `${Math.max(pct, 4)}%`,
                          background: `oklch(${0.75 - i * 0.05} ${0.1 + i * 0.015} 30)`,
                        }}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-0.5">
                  <span>D1</span>
                  <span>D10</span>
                </div>
              </div>
            </>
          )}

          <div className="mt-auto pt-4 border-t space-y-2">
            <Link
              href="/chat"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              ⚖️ Ask the regulation assistant
            </Link>
            <Link
              href="/about"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              📖 About this project
            </Link>
          </div>
        </aside>

        {/* Map */}
        <div className="flex-1 relative p-4">
          {isLoading && (
            <div className="absolute inset-4 flex items-center justify-center bg-background/50 z-10 rounded-lg">
              <Badge variant="outline" className="animate-pulse">
                Loading risk data…
              </Badge>
            </div>
          )}
          {isError && (
            <div className="absolute inset-4 flex items-center justify-center bg-background/80 z-10 rounded-lg">
              <div className="text-center text-sm text-muted-foreground space-y-1">
                <div className="font-medium text-foreground">Could not load risk data</div>
                <div>The API may be waking up — refresh in 30 seconds</div>
              </div>
            </div>
          )}
          {!isLoading && !isError && mapItems.length === 0 && (
            <div className="absolute inset-4 flex items-center justify-center bg-background/80 z-10 rounded-lg">
              <div className="text-sm text-muted-foreground">No data available for this week</div>
            </div>
          )}
          <RiskMap
            items={mapItems}
            week={selectedWeek}
            onWeekChange={setSelectedWeek}
            weeks={weeks}
          />
        </div>
      </main>
    </div>
  );
}
