import { get, post } from "./client";

export interface SearchResult {
  id: number;
  title: string | null;
  url: string | null;
  source_type: string | null;
  summary: string | null;
  tags?: string | null;
  is_read: number;
  score: number | null;
  chunk_count: number;
  entity_count: number;
  relation_count: number;
  created_at?: string | null;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  semantic: boolean;
}

export interface AskRequest {
  question: string;
  source_type?: string;
  include_references?: boolean;
  save?: boolean;
}

export interface SynthesizeRequest {
  topic: string;
  response_type?: string;
  save?: boolean;
}

export interface LLMResponse {
  answer: string;
  references: unknown | null;
}

export async function search(params: {
  q: string;
  semantic?: boolean;
  source_type?: string;
  limit?: number;
}): Promise<SearchResponse> {
  const searchParams = new URLSearchParams();
  searchParams.set("q", params.q);
  searchParams.set("semantic", String(params.semantic ?? true));
  if (params.source_type) searchParams.set("source_type", params.source_type);
  if (params.limit) searchParams.set("limit", String(params.limit));

  return get<SearchResponse>(`/search?${searchParams.toString()}`);
}

export async function ask(req: AskRequest): Promise<LLMResponse> {
  return post<LLMResponse>("/ask", req);
}

export async function synthesize(req: SynthesizeRequest): Promise<LLMResponse> {
  return post<LLMResponse>("/synthesize", req);
}
