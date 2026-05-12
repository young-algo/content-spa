import { useQuery } from "@tanstack/react-query";
import { search } from "../api/search";

export function useSearch(params: {
  q: string;
  semantic?: boolean;
  source_type?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["search", params],
    queryFn: () => search(params),
    enabled: params.q.length > 0,
  });
}
