import { Nav } from "@/components/nav";
import { ChatThread } from "@/components/chat/ChatThread";

export const metadata = {
  title: "Regulation Chat · NYC Rat Risk Intelligence",
  description: "Ask questions about NYC rodent regulations — Health Code, HMC, RCNY — with cited answers.",
};

export default function ChatPage() {
  return (
    <div className="flex flex-col h-screen">
      <Nav />
      <div className="flex flex-1 min-h-0 flex-col">
        <div className="border-b px-4 py-3 bg-muted/30">
          <h1 className="text-sm font-semibold">NYC Regulation Assistant</h1>
          <p className="text-xs text-muted-foreground">
            Powered by NYC Health Code, Housing Maintenance Code, RCNY §81.23, and ECB Penalty Schedule
          </p>
        </div>
        <ChatThread />
      </div>
    </div>
  );
}
