import type { Topic } from "./catalog.generated";

export type Filters = {
  query: string;
  topicId: string;
};

export const EMPTY_FILTERS: Filters = { query: "", topicId: "" };

export function parseFilters(params: URLSearchParams): Filters {
  return {
    query: params.get("q") ?? "",
    topicId: params.get("topic") ?? "",
  };
}

export function filtersToSearchParams(filters: Filters): URLSearchParams {
  const params = new URLSearchParams();
  const query = filters.query.trim();
  if (query) params.set("q", query);
  if (filters.topicId) params.set("topic", filters.topicId);
  return params;
}

function fold(value: string): string {
  return value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();
}

export function applyFilters(topics: Topic[], filters: Filters): Topic[] {
  const needle = fold(filters.query.trim());
  return topics
    .filter((topic) => !filters.topicId || topic.id === filters.topicId)
    .flatMap((topic) => {
      const notebooks = topic.notebooks.filter(
        (notebook) =>
          !needle ||
          fold(
            `${notebook.title} ${notebook.summary} ${notebook.path}`,
          ).includes(needle),
      );
      return notebooks.length > 0 ? [{ ...topic, notebooks }] : [];
    });
}
