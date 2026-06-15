"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CitationPill } from "./CitationPill";
import { readChatStream } from "@/lib/api";
import type { ChatMessage, Citation } from "@/lib/types";

// Parse citations from assistant response text
// Matches §xx.xx and §xx-xx patterns
function parseCitations(text: string): Citation[] {
  const regex = /§[\d]+[\.\-][\d]+[\w\.\-]*/g;
  const matches = [...new Set(text.match(regex) ?? [])];
  return matches.map((m) => ({
    citation: m,
    authority: "NYC",
    document: "",
    content: "",
  }));
}

const EXAMPLE_PROMPTS = [
  "What does 'active rat signs' mean under the NYC Health Code?",
  "What are a landlord's obligations under the Housing Maintenance Code for rodent control?",
  "What penalties apply for rodent violations in a food establishment?",
  "What is Integrated Pest Management under 24 RCNY §81.23?",
];

export function ChatThread() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isStreaming) return;

      const userMsg: ChatMessage = { role: "user", content: question };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsStreaming(true);

      const assistantMsg: ChatMessage = { role: "assistant", content: "" };
      setMessages((prev) => [...prev, assistantMsg]);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        let full = "";
        for await (const token of readChatStream(question, sessionId, ctrl.signal)) {
          full += token;
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { role: "assistant", content: full };
            return next;
          });
        }
        // Extract citations from final response
        const citations = parseCitations(full);
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: full,
            citations,
          };
          return next;
        });
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: "assistant",
              content: "Sorry, an error occurred. Please try again.",
            };
            return next;
          });
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
        void sessionId;
        void setSessionId;
      }
    },
    [isStreaming, sessionId]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Message list */}
      <ScrollArea className="flex-1 px-4 py-4">
        {messages.length === 0 && (
          <div className="text-center text-sm text-muted-foreground py-12 space-y-6">
            <div className="text-4xl">⚖️</div>
            <p className="max-w-sm mx-auto">
              Ask questions about NYC rodent regulations. Every answer is cited
              from the NYC Health Code, Housing Maintenance Code, and RCNY.
            </p>
            <div className="grid sm:grid-cols-2 gap-2 max-w-lg mx-auto">
              {EXAMPLE_PROMPTS.map((p) => (
                <button
                  key={p}
                  onClick={() => sendMessage(p)}
                  className="text-left text-xs border rounded-lg p-3 hover:bg-accent transition-colors"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-4 max-w-2xl mx-auto">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`rounded-xl px-4 py-2.5 text-sm max-w-[85%] ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted border"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {msg.citations.map((c) => (
                      <CitationPill key={c.citation} citation={c} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t px-4 py-3">
        <form onSubmit={handleSubmit} className="flex gap-2 max-w-2xl mx-auto">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about NYC rodent regulations…"
            className="min-h-[44px] max-h-32 resize-none"
            rows={1}
            disabled={isStreaming}
            aria-label="Question input"
          />
          <Button
            type="submit"
            disabled={!input.trim() || isStreaming}
            aria-label="Send"
          >
            {isStreaming ? (
              <span className="animate-pulse">●</span>
            ) : (
              "Send"
            )}
          </Button>
        </form>
        <p className="text-center text-xs text-muted-foreground mt-2">
          Answers are grounded in NYC Health Code, HMC, and RCNY — always verify with official sources.
        </p>
      </div>
    </div>
  );
}
