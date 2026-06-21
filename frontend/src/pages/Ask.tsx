import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ask, synthesize, type AskRequest, type SynthesizeRequest } from "../api/search";
import SegmentedControl from "../components/SegmentedControl";
import { Loader2, Send, Sparkles, AlertCircle, RotateCw, StopCircle } from "lucide-react";

/** Map a thrown error to a user-readable message. Returns null for an abort
 *  (a superseded request), so the caller can skip surfacing it. */
function friendlyMessage(e: unknown): string | null {
  if (e instanceof DOMException && e.name === "AbortError") return null;
  const status = (e as { status?: number } | null)?.status;
  if (status === 429) return "Rate limited. Wait a moment and try again.";
  if (status === 503 || status === 504) return "The model took too long to respond. Try again, or simplify the prompt.";
  if (status && status >= 500) return "The server hit an error. Try again in a moment.";
  if (status === 400) return "That request couldn't be processed. Try rephrasing.";
  return "Couldn't complete that request.";
}

export default function AskPage() {
  const [searchParams] = useSearchParams();
  const topic = searchParams.get("topic") || "";
  const initialMode = searchParams.get("mode") === "synthesize" ? "synthesize" : "ask";
  const [question, setQuestion] = useState(
    topic
      ? initialMode === "synthesize"
        ? topic
        : `What should I understand about ${topic}?`
      : "",
  );
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState<"ask" | "synthesize">(initialMode);

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    // Cancel any in-flight request so rapid re-submits don't race.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const res =
        mode === "ask"
          ? await ask(
              { question: question.trim(), include_references: true } satisfies AskRequest,
              controller.signal,
            )
          : await synthesize(
              { topic: question.trim(), response_type: "Comprehensive Markdown Article" } satisfies SynthesizeRequest,
              controller.signal,
            );
      if (!controller.signal.aborted) setAnswer(res.answer);
    } catch (e) {
      if (controller.signal.aborted) return; // superseded by a newer submit
      setError(friendlyMessage(e) ?? "Request failed");
    } finally {
      if (abortRef.current === controller) setLoading(false);
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Ask</h1>
        <p className="text-xs text-ink-muted">
          Ask questions about your content base or generate synthesis articles.
        </p>
      </div>

      <SegmentedControl
        value={mode}
        onChange={(v) => setMode(v as "ask" | "synthesize")}
        options={[
          { value: "ask", label: "Ask a question" },
          { value: "synthesize", label: "Synthesize article" },
        ]}
        className="w-fit"
      />

      <div>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode === "ask"
              ? "What is the main argument in…?"
              : "Generate a comprehensive article about…"
          }
          rows={3}
          aria-label={mode === "ask" ? "Question" : "Topic to synthesize"}
          className="w-full resize-none rounded-lg border border-ink-border bg-pure px-4 py-3 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/70 focus:border-cobalt focus:ring-1 focus:ring-primary"
        />

        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-xs text-ink-muted">
            {mode === "ask"
              ? "Ask a question about your indexed content."
              : "Generate a comprehensive article from your knowledge base."}
            <kbd className="ml-2 rounded border border-ink-border bg-pure px-1 py-0.5 text-[10px] font-mono text-ink-muted">
              ⌘ Enter
            </kbd>
          </p>
          {loading ? (
            <button
              onClick={handleStop}
              className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-ink-border bg-pure px-4 py-2 text-sm font-semibold text-ink-muted transition-colors hover:bg-accent hover:text-ink focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <StopCircle className="h-4 w-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!question.trim()}
              className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-cobalt px-4 py-2 text-sm font-semibold text-pure transition-opacity hover:opacity-90 focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
            >
              {mode === "ask" ? (
                <>
                  <Send className="h-4 w-4" />
                  Ask
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Synthesize
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-3 rounded-md border border-ink-border bg-paper p-4">
          <Loader2 className="h-4 w-4 animate-spin text-cobalt" />
          <span className="text-sm text-ink-muted">
            {mode === "ask" ? "Searching and generating answer…" : "Synthesizing article…"}
          </span>
          <span className="ml-auto text-[11px] text-ink-muted">This can take 30-60s</span>
        </div>
      )}

      {error && !loading && (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4">
          <div className="flex items-start gap-2 text-red-700">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm">{error}</p>
              <button
                onClick={handleSubmit}
                className="mt-2 inline-flex items-center gap-1.5 rounded border border-red-200 bg-pure px-2.5 py-1 text-xs font-semibold text-red-700 transition-colors hover:bg-red-50 focus:outline-none focus:ring-1 focus:ring-red-500"
              >
                <RotateCw className="h-3.5 w-3.5" />
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {answer && (
        <div className="rounded-md border border-ink-border bg-pure p-6">
          <div className="max-w-none">
            {answer.split("\n").map((para, i) => {
              const trimmed = para.trim();
              if (!trimmed) return <br key={i} />;
              if (trimmed.startsWith("## ") || trimmed.startsWith("### ")) {
                const level = trimmed.startsWith("### ") ? 3 : 2;
                const Tag = `h${level}` as keyof JSX.IntrinsicElements;
                return (
                  <Tag key={i} className="mt-5 mb-2 text-sm font-semibold text-ink">
                    {trimmed.replace(/^#{2,3}\s*/, "")}
                  </Tag>
                );
              }
              if (trimmed.startsWith("- ")) {
                return (
                  <li key={i} className="ml-4 text-sm leading-relaxed text-ink-muted">
                    {trimmed.replace(/^- /, "")}
                  </li>
                );
              }
              return (
                <p key={i} className="text-sm leading-relaxed text-ink">
                  {trimmed}
                </p>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
