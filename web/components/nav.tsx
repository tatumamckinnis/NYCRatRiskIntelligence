import Link from "next/link";
import { Badge } from "@/components/ui/badge";

export function Nav() {
  return (
    <header className="border-b bg-background/95 backdrop-blur sticky top-0 z-50">
      <div className="container mx-auto flex h-14 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-sm">
          <span className="text-xl">🐀</span>
          <span>NYC Rat Risk Intelligence</span>
          <Badge variant="secondary" className="text-xs">
            beta
          </Badge>
        </Link>
        <nav className="flex items-center gap-6 text-sm text-muted-foreground">
          <Link href="/" className="hover:text-foreground transition-colors">
            Map
          </Link>
          <Link
            href="/chat"
            className="hover:text-foreground transition-colors"
          >
            Chat
          </Link>
          <Link
            href="/about"
            className="hover:text-foreground transition-colors"
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
