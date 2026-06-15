"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import type { MapRiskItem } from "@/lib/types";

const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_KEY ?? "";

interface RiskMapProps {
  items: MapRiskItem[];
  week: string;
  onWeekChange: (week: string) => void;
  weeks: string[];
}

type MaplibreMap = {
  addSource: (id: string, source: unknown) => void;
  addLayer: (layer: unknown) => void;
  getSource: (id: string) => { setData: (d: unknown) => void } | undefined;
  isStyleLoaded: () => boolean;
  on: (event: string, layerOrCallback: string | ((e: unknown) => void), callback?: (e: unknown) => void) => void;
  getCanvas: () => HTMLCanvasElement;
  remove: () => void;
};

export function RiskMap({ items, week, onWeekChange, weeks }: RiskMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MaplibreMap | null>(null);
  const router = useRouter();

  // Build GeoJSON from items (using approximate NTA centroids from bounding box center)
  // In production this should use actual NTA centroid coordinates from PostGIS
  function buildGeoJSON(items: MapRiskItem[]) {
    return {
      type: "FeatureCollection",
      features: items.map((item, idx) => {
        // Spread items across NYC bounds as placeholder until real centroids are served
        const lat = 40.58 + (idx % 30) * 0.012;
        const lng = -74.25 + Math.floor(idx / 30) * 0.08;
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [lng, lat] },
          properties: {
            nta_id: item.nta_id,
            risk_score: item.risk_score,
            risk_decile: item.risk_decile,
          },
        };
      }),
    };
  }

  useEffect(() => {
    if (!mapContainer.current) return;
    let destroyed = false;

    const initMap = async () => {
      const { default: maplibregl } = await import("maplibre-gl");

      const map = new maplibregl.Map({
        container: mapContainer.current!,
        style: MAPTILER_KEY
          ? `https://api.maptiler.com/maps/streets/style.json?key=${MAPTILER_KEY}`
          : "https://demotiles.maplibre.org/style.json",
        center: [-73.944, 40.678],
        zoom: 10.5,
      }) as unknown as MaplibreMap;

      mapRef.current = map;

      map.on("load", () => {
        if (destroyed) return;

        map.addSource("nta-risk", {
          type: "geojson",
          data: buildGeoJSON(items),
        });

        map.addLayer({
          id: "nta-circles",
          type: "circle",
          source: "nta-risk",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 4, 13, 14],
            "circle-color": [
              "interpolate",
              ["linear"],
              ["get", "risk_score"],
              0, "#fef08a",
              0.5, "#f97316",
              1, "#dc2626",
            ],
            "circle-opacity": 0.85,
            "circle-stroke-width": 1,
            "circle-stroke-color": "#ffffff40",
          },
        });

        map.on("click", "nta-circles", (e: unknown) => {
          const event = e as { features?: Array<{ properties?: { nta_id?: string } }> };
          const ntaId = event.features?.[0]?.properties?.nta_id;
          if (ntaId) router.push(`/nta/${ntaId}`);
        });

        map.on("mouseenter", "nta-circles", () => {
          map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", "nta-circles", () => {
          map.getCanvas().style.cursor = "";
        });
      });
    };

    initMap().catch(console.error);

    return () => {
      destroyed = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Update circles when items change
  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded?.()) return;
    map.getSource("nta-risk")?.setData(buildGeoJSON(items));
  }, [items]); // eslint-disable-line react-hooks/exhaustive-deps

  const weekIndex = weeks.indexOf(week);

  return (
    <div className="relative w-full h-full min-h-[500px]">
      <div ref={mapContainer} className="absolute inset-0 rounded-lg" />

      {/* Legend */}
      <div className="absolute bottom-20 right-4 bg-background/90 border rounded-lg p-3 text-xs shadow">
        <div className="font-medium mb-2">Rat Risk Score</div>
        <div
          className="w-24 h-3 rounded"
          style={{
            background: "linear-gradient(to right, #fef08a, #f97316, #dc2626)",
          }}
        />
        <div className="flex justify-between w-24 mt-1 text-muted-foreground">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>

      {/* Time slider */}
      {weeks.length > 1 && (
        <div className="absolute bottom-4 left-4 right-32 bg-background/90 border rounded-lg p-3 shadow">
          <div className="text-xs font-medium mb-2">
            Week: <span className="text-primary">{week}</span>
          </div>
          <input
            type="range"
            min={0}
            max={weeks.length - 1}
            value={weekIndex >= 0 ? weekIndex : weeks.length - 1}
            onChange={(e) => onWeekChange(weeks[Number(e.target.value)])}
            className="w-full accent-primary"
            aria-label="Select week"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>{weeks[0]}</span>
            <span>{weeks[weeks.length - 1]}</span>
          </div>
        </div>
      )}
    </div>
  );
}
