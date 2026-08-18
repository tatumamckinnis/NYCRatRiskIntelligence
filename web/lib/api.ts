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
  type Citation,
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

export type ChatStreamEvent =
  | { type: "token"; data: string }
  | { type: "citations"; data: Citation[] };

export async function* readChatStream(
  question: string,
  sessionId?: string,
  signal?: AbortSignal
): AsyncGenerator<ChatStreamEvent> {
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
  // Tracks the preceding `event:` line per the SSE spec — resets to the
  // default "message" type on each blank line (dispatch boundary). Without
  // this, the `event: citations` frame would be misread as literal chat
  // text and dumped into the message bubble as raw JSON.
  let currentEvent: "message" | "citations" = "message";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line === "") {
          currentEvent = "message";
          continue;
        }
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim() as "message" | "citations";
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (currentEvent === "citations") {
          try {
            yield { type: "citations", data: JSON.parse(data) as Citation[] };
          } catch {
            // malformed citations frame — ignore, keep streaming tokens
          }
          currentEvent = "message";
          continue;
        }
        if (data === "[DONE]") return;
        yield { type: "token", data };
      }
    }
  } finally {
    reader.releaseLock();
  }
}
