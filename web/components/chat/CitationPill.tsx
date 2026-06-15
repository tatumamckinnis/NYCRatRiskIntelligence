"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import type { Citation } from "@/lib/types";

interface CitationPillProps {
  citation: Citation;
}

export function CitationPill({ citation }: CitationPillProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <span className="inline-block">
      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
      >
        <Badge
          variant="outline"
          className="cursor-pointer text-xs hover:bg-accent transition-colors"
        >
          {citation.citation}
        </Badge>
      </button>
      {expanded && (
        <span className="block mt-1 ml-1 text-xs text-muted-foreground border-l-2 border-orange-400 pl-2 max-w-sm">
          <span className="font-medium">{citation.document}</span>
          <br />
          {citation.content.slice(0, 200)}
          {citation.content.length > 200 ? "…" : ""}
        </span>
      )}
    </span>
  );
}
