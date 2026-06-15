/**
 * Typed API client for rat-api backend.
 * Base URL is configured via NEXT_PUBLIC_API_BASE_URL env var.
 */

import {
  NtaRiskResponseSchema,
  MapRiskItemSchema,
  InspectionItemSchema,
  type NtaRiskResponse,
  type MapRiskItem,
  type InspectionItem,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function fetchJson<T>(
  path: string,
  schema: { parse: (v: unknown) => T }
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${path}`);
  }
  const json = await res.json();
  return schema.parse(json);
}

// GET /risk/nta/{nta_id}
export async function getNtaRisk(ntaId: string): Promise<NtaRiskResponse> {
  return fetchJson(`/risk/nta/${ntaId}`, NtaRiskResponseSchema);
}

// GET /risk/map?week={week}
export async function getMapRisk(week?: string): Promise<MapRiskItem[]> {
  const params = week ? `?week=${week}` : "";
  const raw = await fetch(`${BASE_URL}/risk/map${params}`);
  if (!raw.ok) throw new Error(`API error ${raw.status}: /risk/map`);
  const json = await raw.json();
  return (json as unknown[]).map((item) => MapRiskItemSchema.parse(item));
}

// GET /inspections/nta/{nta_id}?since={date}
export async function getInspections(
  ntaId: string,
  since?: string
): Promise<InspectionItem[]> {
  const params = since ? `?since=${since}` : "";
  const raw = await fetch(`${BASE_URL}/inspections/nta/${ntaId}${params}`);
  if (!raw.ok) throw new Error(`API error ${raw.status}`);
  const json = await raw.json();
  return (json as unknown[]).map((item) => InspectionItemSchema.parse(item));
}

// POST /chat — returns an EventSource-like ReadableStream
export function streamChat(
  question: string,
  sessionId?: string
): EventSource {
  const url = new URL(`${BASE_URL}/chat`);
  // Use GET with query params for EventSource compatibility
  // The backend accepts POST — we use fetch + ReadableStream instead
  void url; // EventSource only supports GET; use readChatStream below
  throw new Error("Use readChatStream for streaming");
}

export async function* readChatStream(
  question: string,
  sessionId?: string,
  signal?: AbortSignal
): AsyncGenerator<string> {
  const body: Record<string, string> = { question };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) throw new Error(`Chat error ${res.status}`);
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          yield data;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
