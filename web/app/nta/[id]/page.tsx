"use client";

import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { Nav } from "@/components/nav";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { FactorBar } from "@/components/charts/FactorBar";
import { getNtaRisk, getInspections } from "@/lib/api";

const DECILE_LABELS: Record<number, string> = {
  1: "Very Low",
  2: "Very Low",
  3: "Low",
  4: "Low",
  5: "Moderate",
  6: "Moderate",
  7: "Elevated",
  8: "High",
  9: "High",
  10: "Very High",
};

const DECILE_COLORS: Record<number, string> = {
  1: "bg-green-100 text-green-800",
  2: "bg-green-100 text-green-800",
  3: "bg-lime-100 text-lime-800",
  4: "bg-lime-100 text-lime-800",
  5: "bg-yellow-100 text-yellow-800",
  6: "bg-yellow-100 text-yellow-800",
  7: "bg-orange-100 text-orange-800",
  8: "bg-red-100 text-red-800",
  9: "bg-red-100 text-red-800",
  10: "bg-red-200 text-red-900",
};

export default function NtaDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  const {
    data: risk,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["nta-risk", id],
    queryFn: () => getNtaRisk(id),
  });

  const { data: inspections = [] } = useQuery({
    queryKey: ["inspections", id],
    queryFn: () => getInspections(id),
    enabled: !!risk,
  });

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen">
        <Nav />
        <main className="flex-1 container mx-auto px-4 py-8 max-w-4xl">
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-32 bg-muted rounded-lg animate-pulse" />
            ))}
          </div>
        </main>
      </div>
    );
  }

  if (error || !risk) {
    return (
      <div className="flex flex-col min-h-screen">
        <Nav />
        <main className="flex-1 container mx-auto px-4 py-8 max-w-4xl">
          <div className="flex flex-col items-center justify-center py-24 gap-4 text-muted-foreground">
            <AlertCircle className="w-12 h-12" />
            <p>NTA <code className="bg-muted px-1 rounded">{id}</code> not found or data unavailable.</p>
            <Link href="/" className="text-sm underline">
              ← Back to map
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const decileLabel = DECILE_LABELS[risk.risk_decile] ?? "Unknown";
  const decileColor = DECILE_COLORS[risk.risk_decile] ?? "bg-muted";

  return (
    <div className="flex flex-col min-h-screen">
      <Nav />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-4xl space-y-6">
        {/* Back + heading */}
        <div>
          <Link
            href="/"
            className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to map
          </Link>
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-2xl font-bold">NTA {id}</h1>
              <p className="text-muted-foreground text-sm mt-1">
                Week of {risk.current_week} · model v{risk.model_version}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-3xl font-bold tabular-nums">
                  {(risk.risk_score * 100).toFixed(1)}
                  <span className="text-lg text-muted-foreground">%</span>
                </div>
                <div className="text-xs text-muted-foreground">risk score</div>
              </div>
              <Badge className={`${decileColor} px-3 py-1 text-sm font-semibold`}>
                D{risk.risk_decile} · {decileLabel}
              </Badge>
            </div>
          </div>
        </div>

        <Separator />

        {/* Cards row */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Forecast */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">12-Week Forecast</CardTitle>
            </CardHeader>
            <CardContent>
              <ForecastChart
                forecasts={risk.forecast_12w}
                currentScore={risk.risk_score}
              />
            </CardContent>
          </Card>

          {/* Top factors */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Top Risk Factors</CardTitle>
            </CardHeader>
            <CardContent>
              <FactorBar factors={risk.top_factors} />
            </CardContent>
          </Card>
        </div>

        {/* Recent inspections */}
        {inspections.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Recent Inspections</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="divide-y text-sm">
                {inspections.slice(0, 10).map((insp) => (
                  <div
                    key={insp.inspection_id}
                    className="flex items-center justify-between py-2 gap-4"
                  >
                    <span className="text-muted-foreground text-xs tabular-nums">
                      {insp.date}
                    </span>
                    <span className="flex-1 truncate">{insp.result}</span>
                    <Badge
                      variant={
                        insp.result.toLowerCase().includes("pass")
                          ? "outline"
                          : "destructive"
                      }
                      className="text-xs shrink-0"
                    >
                      {insp.result.toLowerCase().includes("pass")
                        ? "Pass"
                        : "Fail"}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
