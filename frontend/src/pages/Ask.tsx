import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ask, synthesize, type AskRequest, type SynthesizeRequest } from "../api/search";
import { Loader2, Send, Sparkles } from "lucide-react";

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

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Ask questions about your content base or generate synthesis articles
      </p>

      <div className="mt-6 flex rounded-lg border border-border bg-card p-0.5 w-fit">
        <button
          onClick={() => setMode("ask")}
          className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            mode === "ask"
              ? "bg-primary/10 text-primary"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Ask a question
        </button>
        <button
          onClick={() => setMode("synthesize")}
          className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            mode === "synthesize"
              ? "bg-primary/10 text-primary"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Synthesize article
        </button>
      </div>

      <div className="mt-4">
        <div className="flex gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              mode === "ask"
                ? "What is the main argument in...?"
                : "Generate a comprehensive article about..."
            }
            rows={3}
            className="flex-1 resize-none rounded-lg border border-input bg-card px-4 py-3 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-ring focus:ring-1 focus:ring-ring"
          />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            {mode === "ask"
              ? "Ask a question about your indexed content"
              : "Generate a comprehensive article from your knowledge base"}
          </p>
          <button
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
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
              </>
            )}
          </button>
        </div>
      </div>

      {loading && (
        <div className="mt-8 flex items-center gap-3 rounded-lg border border-border bg-card p-4">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          <span className="text-sm text-muted-foreground">
            {mode === "ask" ? "Searching and generating answer..." : "Synthesizing article..."}
          </span>
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/5 p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {answer && (
        <div className="mt-6 rounded-lg border border-border bg-card p-6">
          <div className="prose prose-sm prose-invert max-w-none">
            {answer.split("\n").map((para, i) => {
              const trimmed = para.trim();
              if (!trimmed) return <br key={i} />;
              if (trimmed.startsWith("## ") || trimmed.startsWith("### ")) {
                const level = trimmed.startsWith("### ") ? 3 : 2;
                const Tag = `h${level}` as keyof JSX.IntrinsicElements;
                return (
                  <Tag key={i} className="mt-4 mb-2 font-semibold text-foreground">
                    {trimmed.replace(/^#{2,3}\s*/, "")}
                  </Tag>
                );
              }
              if (trimmed.startsWith("- ")) {
                return (
                  <li key={i} className="ml-4 text-sm text-muted-foreground">
                    {trimmed.replace(/^- /, "")}
                  </li>
                );
              }
              return (
                <p key={i} className="text-sm leading-relaxed text-muted-foreground">
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
