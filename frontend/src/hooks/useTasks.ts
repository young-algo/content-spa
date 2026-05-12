import { useQuery } from "@tanstack/react-query";
import { getTask, type TaskStatus } from "../api/system";

export function useTask(taskId: string | null) {
  return useQuery<TaskStatus>({
    queryKey: ["task", taskId],
    queryFn: () => getTask(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed") return false;
      return 2000;
    },
  });
}
