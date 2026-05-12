import { useState, useRef, type FormEvent, type ChangeEvent } from "react";
import { Upload, Link, Loader2, CheckCircle2 } from "lucide-react";
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
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Add Content</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Ingest a URL or upload a local file into your index
      </p>

      <div className="mt-8">
        <h2 className="text-sm font-medium text-foreground">From URL</h2>
        <form onSubmit={handleUrlSubmit} className="mt-2 flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/article..."
            className="flex-1 rounded-lg border border-input bg-card px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-ring focus:ring-1 focus:ring-ring"
          />
          <button
            type="submit"
            disabled={!url.trim() || !!taskId}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            <Link className="h-4 w-4" />
            Ingest
          </button>
        </form>
      </div>

      {urlTask && (
        <div className="mt-3">
          <TaskProgress task={urlTask} />
        </div>
      )}

      <div className="mt-8">
        <h2 className="text-sm font-medium text-foreground">From File</h2>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className={`mt-2 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-colors ${
            dragOver
              ? "border-primary bg-primary/5"
              : "border-border hover:border-muted-foreground/50"
          }`}
        >
          <Upload className="h-8 w-8 text-muted-foreground" />
          <p className="mt-3 text-sm text-muted-foreground">
            Drop a file here or click to browse
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            PDF, Markdown, or plain text
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.md,.txt,.markdown,.html"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </div>

      {fileTask && (
        <div className="mt-3">
          <TaskProgress task={fileTask} />
        </div>
      )}
    </div>
  );
}
