import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import type { TaskStatus } from "../api/system";

interface TaskProgressProps {
  task: TaskStatus;
}

function resultMessage(result: unknown): string {
  if (typeof result === "object" && result !== null && "doc_id" in result) {
    const docId = (result as Record<string, unknown>).doc_id;
    if (docId != null) {
      return `Document #${String(docId)} indexed`;
    }
  }
  return "Done";
}

export default function TaskProgress({ task }: TaskProgressProps) {
  const isRunning = task.status === "pending" || task.status === "running";
  const isComplete = task.status === "complete";
  const isFailed = task.status === "failed";

  return (
    <div className="flex items-center gap-3 rounded-md border border-ink-border bg-paper p-3">
      {isRunning && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-cobalt" />}
      {isComplete && <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />}
      {isFailed && <XCircle className="h-4 w-4 shrink-0 text-red-600" />}

      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-ink">{task.message || task.status}</p>
        {isRunning && task.progress > 0 && (
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-cobalt transition-all duration-500"
              style={{ width: `${task.progress}%` }}
            />
          </div>
        )}
        {isFailed && task.error && (
          <p className="mt-0.5 text-xs text-red-600">{task.error}</p>
        )}
        {isComplete && task.result != null && (
          <p className="mt-0.5 text-xs text-emerald-600">
            {resultMessage(task.result)}
          </p>
        )}
      </div>
    </div>
  );
}
