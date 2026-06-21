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

type MaplibrePopup = {
  setLngLat: (lnglat: [number, number]) => MaplibrePopup;
  setHTML: (html: string) => MaplibrePopup;
  addTo: (map: MaplibreMap) => MaplibrePopup;
  remove: () => void;
};

export function RiskMap({ items, week, onWeekChange, weeks }: RiskMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MaplibreMap | null>(null);
  const popupRef = useRef<MaplibrePopup | null>(null);
  const router = useRouter();

  function buildGeoJSON(items: MapRiskItem[]) {
    const features = items
      .filter((item) => item.centroid_lat != null && item.centroid_lon != null)
      .map((item) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [item.centroid_lon!, item.centroid_lat!],
        },
        properties: {
          nta_id: item.nta_id,
          nta_name: item.nta_name ?? item.nta_id,
          risk_score: item.risk_score,
          risk_decile: item.risk_decile,
        },
      }));
    return { type: "FeatureCollection", features };
  }

  useEffect(() => {
    if (!mapContainer.current) return;
    let destroyed = false;

    const initMap = async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ml = await import("maplibre-gl") as any;

      const map = new ml.Map({
        container: mapContainer.current!,
        style: MAPTILER_KEY
          ? `https://api.maptiler.com/maps/streets/style.json?key=${MAPTILER_KEY}`
          : "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        center: [-73.944, 40.678],
        zoom: 10.5,
      }) as MaplibreMap;

      mapRef.current = map;

      const popup = new ml.Popup({
        closeButton: false,
        closeOnClick: false,
        maxWidth: "220px",
      }) as MaplibrePopup;
      popupRef.current = popup;

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
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 5, 13, 16],
            "circle-color": [
              "interpolate",
              ["linear"],
              ["get", "risk_score"],
              0, "#4ade80",
              0.4, "#fbbf24",
              0.7, "#f97316",
              1, "#dc2626",
            ],
            "circle-opacity": 0.85,
            "circle-stroke-width": 1.5,
            "circle-stroke-color": "#ffffff60",
          },
        });

        map.on("click", "nta-circles", (e: unknown) => {
          const event = e as { features?: Array<{ properties?: { nta_id?: string } }> };
          const ntaId = event.features?.[0]?.properties?.nta_id;
          if (ntaId) router.push(`/nta/${ntaId}`);
        });

        map.on("mouseenter", "nta-circles", (e: unknown) => {
          const event = e as {
            lngLat: { lng: number; lat: number };
            features?: Array<{ properties?: { nta_name?: string; risk_score?: number; risk_decile?: number } }>;
          };
          map.getCanvas().style.cursor = "pointer";
          const props = event.features?.[0]?.properties;
          if (!props) return;
          const pct = Math.round((props.risk_score ?? 0) * 100);
          popup
            .setLngLat([event.lngLat.lng, event.lngLat.lat])
            .setHTML(
              `<div style="font-family:sans-serif;font-size:13px;line-height:1.5">
                <strong style="display:block;margin-bottom:2px">${props.nta_name ?? ""}</strong>
                <span style="color:#888">Risk score:</span> <strong>${pct}%</strong> &nbsp; Decile ${props.risk_decile ?? "—"}
                <br/><span style="font-size:11px;color:#aaa">Click for full details →</span>
              </div>`
            )
            .addTo(map);
        });

        map.on("mouseleave", "nta-circles", () => {
          map.getCanvas().style.cursor = "";
          popup.remove();
        });
      });
    };

    initMap().catch(console.error);

    return () => {
      destroyed = true;
      popupRef.current?.remove();
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded?.()) return;
    map.getSource("nta-risk")?.setData(buildGeoJSON(items));
  }, [items]); // eslint-disable-line react-hooks/exhaustive-deps

  const weekIndex = weeks.indexOf(week);

  return (
    <div className="relative w-full h-full min-h-[500px]">
      <div ref={mapContainer} className="absolute inset-0 rounded-lg" />

      {/* Instruction hint */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-background/90 border rounded-full px-4 py-1.5 text-xs text-muted-foreground shadow pointer-events-none z-10">
        Hover a dot to see the neighborhood · Click to explore details
      </div>

      {/* Legend */}
      <div className="absolute bottom-20 right-4 bg-background/90 border rounded-lg p-3 text-xs shadow z-10">
        <div className="font-medium mb-2">Rat Risk Score</div>
        <div
          className="w-28 h-3 rounded"
          style={{
            background: "linear-gradient(to right, #4ade80, #fbbf24, #f97316, #dc2626)",
          }}
        />
        <div className="flex justify-between w-28 mt-1 text-muted-foreground">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>

      {/* Time slider */}
      {weeks.length > 1 && (
        <div className="absolute bottom-4 left-4 right-36 bg-background/90 border rounded-lg p-3 shadow z-10">
          <div className="text-xs font-medium mb-2">
            Showing week of <span className="text-primary font-mono">{week}</span>
            <span className="text-muted-foreground ml-2">— drag to travel through time</span>
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
            <span>Today</span>
          </div>
        </div>
      )}
    </div>
  );
}
