import { z } from "zod";

// ------------------------------------------------------------------
// Risk API response schemas (mirror Pydantic models from rat-api)
// ------------------------------------------------------------------

export const RiskFactorSchema = z.object({
  feature: z.string(),
  contribution: z.number(),
  direction: z.enum(["up", "down"]),
  readable: z.string(),
});

export const WeekForecastSchema = z.object({
  week: z.string(), // ISO date string
  risk_score: z.number(),
  ci_low: z.number(),
  ci_high: z.number(),
});

export const NtaRiskResponseSchema = z.object({
  nta_id: z.string(),
  current_week: z.string(),
  risk_score: z.number(),
  risk_decile: z.number().int().min(1).max(10),
  top_factors: z.array(RiskFactorSchema),
  model_version: z.string(),
  forecast_12w: z.array(WeekForecastSchema),
});

export const MapRiskItemSchema = z.object({
  nta_id: z.string(),
  risk_score: z.number(),
  risk_decile: z.number().int(),
  nta_name: z.string().nullable().optional(),
  centroid_lat: z.number().nullable().optional(),
  centroid_lon: z.number().nullable().optional(),
});

export const InspectionItemSchema = z.object({
  inspection_id: z.string(),
  date: z.string(),
  result: z.string(),
  bbl: z.string().nullable(),
  lat: z.number().nullable(),
  lon: z.number().nullable(),
});

export type RiskFactor = z.infer<typeof RiskFactorSchema>;
export type WeekForecast = z.infer<typeof WeekForecastSchema>;
export type NtaRiskResponse = z.infer<typeof NtaRiskResponseSchema>;
export type MapRiskItem = z.infer<typeof MapRiskItemSchema>;
export type InspectionItem = z.infer<typeof InspectionItemSchema>;

// ------------------------------------------------------------------
// Chat types
// ------------------------------------------------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export interface Citation {
  citation: string;
  authority: string;
  document: string;
  content: string;
}
