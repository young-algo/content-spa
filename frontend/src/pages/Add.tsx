import { useState, useRef, type FormEvent, type ChangeEvent } from "react";
import { Upload, Link as LinkIcon, Loader2 } from "lucide-react";
import { ingestUrl, ingestFile } from "../api/system";
import { useTask } from "../hooks/useTasks";
import TaskProgress from "../components/TaskProgress";

export default function AddPage() {
  const [url, setUrl] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileTaskId, setFileTaskId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const { data: urlTask } = useTask(taskId);
  const { data: fileTask } = useTask(fileTaskId);

  const handleUrlSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    const task = await ingestUrl(url.trim());
    setTaskId(task.task_id);
    setUrl("");
  };

  const handleFile = async (file: File) => {
    const task = await ingestFile(file);
    setFileTaskId(task.task_id);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-6 md:space-y-8 animate-fade-in">
      <div className="flex flex-col gap-1.5 border-b border-ink-border pb-4">
        <h1 className="text-xl font-bold tracking-tight text-ink">Add Content</h1>
        <p className="text-xs text-ink-muted">Ingest a URL or upload a local file into your index.</p>
      </div>

      <div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
          From URL
        </h2>
        <form onSubmit={handleUrlSubmit} className="mt-2 flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/article..."
            aria-label="URL to ingest"
            className="flex-1 rounded-lg border border-ink-border bg-pure px-4 py-2.5 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/70 focus:border-cobalt focus:ring-1 focus:ring-primary"
          />
          <button
            type="submit"
            disabled={!url.trim() || !!taskId}
            className="inline-flex items-center gap-2 rounded-lg bg-cobalt px-4 py-2.5 text-sm font-semibold text-pure transition-opacity hover:opacity-90 focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
          >
            {taskId ? <Loader2 className="h-4 w-4 animate-spin" /> : <LinkIcon className="h-4 w-4" />}
            Ingest
          </button>
        </form>

        {urlTask && (
          <div className="mt-3">
            <TaskProgress task={urlTask} />
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-ink-muted font-mono">
          From File
        </h2>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileRef.current?.click(); } }}
          className={`mt-2 flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed p-12 transition-colors focus:outline-none focus:ring-1 focus:ring-primary ${
            dragOver
              ? "border-cobalt bg-cobalt-light/40"
              : "border-ink-border hover:border-ink-muted/60 hover:bg-accent/30"
          }`}
        >
          <Upload className="h-8 w-8 text-ink-muted" />
          <p className="mt-3 text-sm text-ink">Drop a file here or click to browse</p>
          <p className="mt-1 text-xs text-ink-muted">PDF, Markdown, or plain text</p>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.md,.txt,.markdown,.html"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {fileTask && (
          <div className="mt-3">
            <TaskProgress task={fileTask} />
          </div>
        )}
      </div>
    </div>
  );
}
