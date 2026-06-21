import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { QueryKey } from "@tanstack/react-query";
import {
  fetchDocuments,
  fetchDocument,
  updateDocument,
  deleteDocument,
  type DocumentItem,
  type DocumentListParams,
  type DocumentListResponse,
} from "../api/documents";

export function useDocuments(params: DocumentListParams = {}) {
  return useQuery({
    queryKey: ["documents", params],
    queryFn: () => fetchDocuments(params),
  });
}

export function useDocument(id: number) {
  return useQuery({
    queryKey: ["document", id],
    queryFn: () => fetchDocument(id),
    enabled: id > 0,
  });
}

export function useUpdateDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: { is_read?: boolean } }) =>
      updateDocument(id, data),
    // Optimistic update: flip is_read in every cached document list and the
    // single-doc cache before the request lands. On error, restore the prior
    // cache so a failed toggle never leaves the UI lying about read state.
    onMutate: async ({ id, data }) => {
      await queryClient.cancelQueries({ queryKey: ["documents"] });
      await queryClient.cancelQueries({ queryKey: ["document", id] });

      const listQueries = queryClient.getQueryCache().findAll({ queryKey: ["documents"] });
      const previousLists: Array<{ queryKey: QueryKey; data: unknown }> = listQueries.map((q) => ({
        queryKey: q.queryKey,
        data: q.state.data,
      }));
      const previousDoc = queryClient.getQueryData(["document", id]);

      const nextIsRead = data.is_read ? 1 : 0;
      const patchList = (old: unknown): DocumentListResponse | unknown => {
        if (!old || !Array.isArray((old as DocumentListResponse).items)) return old;
        return {
          ...(old as DocumentListResponse),
          items: (old as DocumentListResponse).items.map((d) =>
            d.id === id ? { ...d, is_read: nextIsRead } : d,
          ),
        };
      };
      listQueries.forEach((q) => queryClient.setQueryData(q.queryKey, patchList(q.state.data)));
      queryClient.setQueryData(["document", id], (old: unknown) =>
        old ? { ...(old as DocumentItem), is_read: nextIsRead } : old,
      );

      return { previousLists, previousDoc };
    },
    onError: (_err, { id }, context) => {
      context?.previousLists.forEach(({ queryKey, data }) =>
        queryClient.setQueryData(queryKey, data),
      );
      if (context?.previousDoc !== undefined) {
        queryClient.setQueryData(["document", id], context.previousDoc);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document"] });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}
