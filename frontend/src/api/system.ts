import { get, post, uploadFile } from "./client";

export interface TaskStatus {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  result: unknown | null;
  error: string | null;
}

export interface StatsResponse {
  total: number;
  unread_count: number;
  read_count: number;
  by_source_type: Array<{ source_type: string; count: number }>;
  top_tags: Array<[string, number]>;
  oldest_unread: Record<string, unknown> | null;
}

export interface TagCount {
  tag: string;
  count: number;
}

export interface TopicCluster {
  name: string;
  description?: string | null;
  tags: string[];
}

export interface TopicsResponse {
  tags: TagCount[];
  clusters: TopicCluster[] | null;
  cluster_created_at?: string | null;
}

export async function fetchStats(): Promise<StatsResponse> {
  return get<StatsResponse>("/stats");
}

export async function fetchTopics(
  cluster = false,
  source_type?: string,
  refresh = false,
): Promise<TopicsResponse> {
  const params = new URLSearchParams();
  if (cluster) params.set("cluster", "true");
  if (source_type) params.set("source_type", source_type);
  if (refresh) params.set("refresh", "true");
  const qs = params.toString();
  return get<TopicsResponse>(`/topics${qs ? `?${qs}` : ""}`);
}

export async function ingestUrl(url: string): Promise<TaskStatus> {
  return post<TaskStatus>("/ingest/url", { url });
}

export async function ingestFile(file: File): Promise<TaskStatus> {
  return uploadFile<TaskStatus>("/ingest/file", file);
}

export async function reindex(reset = true, resume = true): Promise<TaskStatus> {
  return post<TaskStatus>("/reindex", { reset, resume });
}

export async function getTask(taskId: string): Promise<TaskStatus> {
  return get<TaskStatus>(`/tasks/${taskId}`);
}
