import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const BASE_URL = __ENV.API_URL || "https://rat-api-g3lf.onrender.com";

// Sample NTA IDs spread across boroughs
const NTA_IDS = [
  "BK0101", "BK0102", "BK0201", "BK0301",
  "MN0101", "MN0201", "MN0301", "MN1101",
  "QN0101", "QN0201", "QN0301",
  "BX0101", "BX0201",
  "SI0101",
];

const CURRENT_WEEK = "2026-05-11";

export const options = {
  stages: [
    { duration: "30s", target: 50 },   // ramp up to 50 VUs
    { duration: "2m",  target: 50 },   // hold at 50 VUs
    { duration: "30s", target: 0  },   // ramp down
  ],
  thresholds: {
    // Spec targets
    "http_req_duration{endpoint:risk_map}":  ["p(95)<800"],
    "http_req_duration{endpoint:risk_nta}":  ["p(95)<800"],
    "http_req_duration{endpoint:health}":    ["p(95)<500"],
    http_req_failed: ["rate<0.01"],          // <1% error rate
  },
};

export default function () {
  const nta = NTA_IDS[Math.floor(Math.random() * NTA_IDS.length)];

  // GET /health
  const health = http.get(`${BASE_URL}/health`, {
    tags: { endpoint: "health" },
  });
  check(health, { "health 200": (r) => r.status === 200 });

  sleep(0.5);

  // GET /risk/map
  const map = http.get(`${BASE_URL}/risk/map?week=${CURRENT_WEEK}`, {
    tags: { endpoint: "risk_map" },
  });
  check(map, {
    "risk/map 200": (r) => r.status === 200,
    "risk/map has data": (r) => {
      try { return JSON.parse(r.body).length > 0; } catch { return false; }
    },
  });

  sleep(0.5);

  // GET /risk/nta/{id}
  const nta_resp = http.get(`${BASE_URL}/risk/nta/${nta}`, {
    tags: { endpoint: "risk_nta" },
  });
  check(nta_resp, {
    "risk/nta 200": (r) => r.status === 200,
    "risk/nta has score": (r) => {
      try { return JSON.parse(r.body).risk_score !== undefined; } catch { return false; }
    },
  });

  sleep(1);
}
