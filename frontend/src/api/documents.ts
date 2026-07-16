import { get, patch, del, post } from "./client";

export interface DocumentItem {
  id: number;
  url: string;
  title: string | null;
  source_type: string | null;
  summary: string | null;
  tags: string | null;
  content: string | null;
  is_read: number;
  read_at: string | null;
  created_at: string | null;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

export interface DocumentListParams {
  page?: number;
  per_page?: number;
  is_read?: boolean;
  source_type?: string;
  tag?: string;
  sort?: string;
}

export async function fetchDocuments(params: DocumentListParams = {}): Promise<DocumentListResponse> {
  const searchParams = new URLSearchParams();
  if (params.page) searchParams.set("page", String(params.page));
  if (params.per_page) searchParams.set("per_page", String(params.per_page));
  if (params.is_read !== undefined) searchParams.set("is_read", String(params.is_read));
  if (params.source_type) searchParams.set("source_type", params.source_type);
  if (params.tag) searchParams.set("tag", params.tag);
  if (params.sort) searchParams.set("sort", params.sort);

  const qs = searchParams.toString();
  return get<DocumentListResponse>(`/documents${qs ? `?${qs}` : ""}`);
}

export async function fetchDocument(id: number): Promise<DocumentItem> {
  return get<DocumentItem>(`/documents/${id}`);
}

export async function updateDocument(id: number, data: { is_read?: boolean }): Promise<DocumentItem> {
  return patch<DocumentItem>(`/documents/${id}`, data);
}

export async function deleteDocument(id: number): Promise<void> {
  return del(`/documents/${id}`);
}

export async function openDocumentSource(id: number): Promise<void> {
  return post<void>(`/documents/${id}/open`);
}
