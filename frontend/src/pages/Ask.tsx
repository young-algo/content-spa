import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ask, synthesize, type AskRequest, type SynthesizeRequest } from "../api/search";
import { Loader2, Send, Sparkles, AlertCircle } from "lucide-react";

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

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setAnswer("");

    try {
      if (mode === "ask") {
        const req: AskRequest = {
          question: question.trim(),
          include_references: true,
        };
        const res = await ask(req);
        setAnswer(res.answer);
      } else {
        const req: SynthesizeRequest = {
          topic: question.trim(),
          response_type: "Comprehensive Markdown Article",
        };
        const res = await synthesize(req);
        setAnswer(res.answer);
      }
    } catch (e: any) {
      setError(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const filterBtn = (active: boolean) =>
    `rounded px-4 py-2 text-sm font-medium transition-colors focus:outline-none ${
      active
        ? "bg-cobalt-light/60 text-cobalt font-semibold"
        : "text-ink-muted hover:text-ink"
    }`;

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Ask</h1>
        <p className="text-xs text-ink-muted">
          Ask questions about your content base or generate synthesis articles.
        </p>
      </div>

      <div className="flex rounded-md border border-ink-border bg-paper p-0.5 w-fit">
        <button onClick={() => setMode("ask")} className={filterBtn(mode === "ask")}>
          Ask a question
        </button>
        <button onClick={() => setMode("synthesize")} className={filterBtn(mode === "synthesize")}>
          Synthesize article
        </button>
      </div>

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
            <span className="ml-1 text-ink-border">·</span>
            <kbd className="ml-1 rounded border border-ink-border bg-pure px-1 py-0.5 text-[10px] font-mono text-ink-muted">
              ⌘ Enter
            </kbd>
          </p>
          <button
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
            className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-cobalt px-4 py-2 text-sm font-semibold text-pure transition-opacity hover:opacity-90 focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : mode === "ask" ? (
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
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-3 rounded-md border border-ink-border bg-paper p-4">
          <Loader2 className="h-4 w-4 animate-spin text-cobalt" />
          <span className="text-sm text-ink-muted">
            {mode === "ask" ? "Searching and generating answer…" : "Synthesizing article…"}
          </span>
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50/30 p-4">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-sm">{error}</span>
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
