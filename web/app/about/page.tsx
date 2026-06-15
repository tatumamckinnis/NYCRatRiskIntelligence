import Link from "next/link";
import { Nav } from "@/components/nav";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export const metadata = {
  title: "About · NYC Rat Risk Intelligence",
};

export default function AboutPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <Nav />
      <main className="flex-1 container mx-auto px-4 py-12 max-w-3xl space-y-8">
        <div>
          <h1 className="text-3xl font-bold">About This Project</h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Machine-learning–powered rodent risk predictions for NYC
            neighborhoods, with a cited legal Q&A assistant.
          </p>
        </div>

        <Separator />

        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Methodology</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Risk scores are computed weekly at the{" "}
            <strong>Neighborhood Tabulation Area (NTA)</strong> level using a
            multi-modal ensemble model trained on:
          </p>
          <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-5">
            <li>NYC DOHMH rodent inspection outcomes (2015–present)</li>
            <li>311 complaint counts for rodent-related categories</li>
            <li>ACS 5-year median household income estimates</li>
            <li>Satellite-derived vegetation and building density indices (Sentinel-2)</li>
            <li>Temporal lag features (4-week, 8-week, 12-week complaint lags)</li>
          </ul>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The ensemble combines a CatBoost tabular model, a Temporal Fusion
            Transformer (TFT) for 12-week forecasting, and a LightGBM baseline.
            All models use expanding-window time-series cross-validation with a
            28-day gap to prevent leakage. The supervised label is{" "}
            <code className="bg-muted px-1 rounded text-xs">
              active_rat_signs_ind
            </code>{" "}
            as defined by DOHMH inspection results.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Regulation Assistant</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The chat interface uses a Retrieval-Augmented Generation (RAG)
            pipeline. Questions are answered by retrieving relevant passages from
            five legal sources, then generating a cited response using a free
            Groq-hosted LLM. Answers are grounded only in retrieved content —
            the model does not speculate beyond the documents.
          </p>

          <div className="grid sm:grid-cols-2 gap-3">
            {[
              {
                title: "NYC Health Code Article 151",
                body: "Primary rodent control requirements for property owners.",
                auth: "DOHMH",
              },
              {
                title: "Housing Maintenance Code §§27-2017–2018",
                body: "Landlord obligations for pest extermination in residential buildings.",
                auth: "HPD",
              },
              {
                title: "24 RCNY §81.23",
                body: "Integrated Pest Management requirements for food establishments.",
                auth: "DOHMH",
              },
              {
                title: "ECB Penalty Schedule",
                body: "Environmental Control Board fines for rodent violations.",
                auth: "ECB/OATH",
              },
              {
                title: "DOHMH Rodent Academy",
                body: "Training materials on rodent biology and control practices.",
                auth: "DOHMH",
              },
            ].map((s) => (
              <Card key={s.title} className="text-sm">
                <CardHeader className="pb-1 pt-3 px-4">
                  <CardTitle className="text-sm leading-tight">{s.title}</CardTitle>
                </CardHeader>
                <CardContent className="px-4 pb-3">
                  <p className="text-muted-foreground text-xs">{s.body}</p>
                  <p className="text-xs mt-1 text-orange-600 font-medium">{s.auth}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Limitations</h2>
          <ul className="text-sm text-muted-foreground space-y-2 list-disc pl-5">
            <li>
              Risk scores reflect <em>reporting patterns</em>, not absolute rat
              populations. High-income areas may be under-reported.
            </li>
            <li>
              The 12-week forecast has increasing uncertainty beyond 4 weeks;
              treat distant predictions as directional only.
            </li>
            <li>
              Legal citations are extracted from PDF documents and may contain
              OCR errors. Always verify with official NYC sources before acting.
            </li>
            <li>
              The backend runs on Render&apos;s free tier and may experience a
              ~30-second cold-start delay after periods of inactivity.
            </li>
          </ul>
        </section>

        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Data Sources</h2>
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>NYC Open Data — Rodent Inspections, 311 Service Requests</li>
            <li>U.S. Census Bureau — American Community Survey (ACS 5-year)</li>
            <li>ESA / Copernicus — Sentinel-2 satellite imagery</li>
            <li>NYC Department of City Planning — NTA boundaries (2010)</li>
          </ul>
        </section>

        <Separator />

        <div className="flex gap-4 text-sm">
          <Link
            href="/"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            ← Map
          </Link>
          <Link
            href="/chat"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            ⚖️ Regulation Chat
          </Link>
        </div>
      </main>
    </div>
  );
}
