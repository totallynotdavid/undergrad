import type { Course } from "./catalog.generated";

export type Filters = {
  query: string;
  courseId: string;
};

export const EMPTY_FILTERS: Filters = { query: "", courseId: "" };

export function parseFilters(params: URLSearchParams): Filters {
  return {
    query: params.get("q") ?? "",
    courseId: params.get("curso") ?? "",
  };
}

export function filtersToSearchParams(filters: Filters): URLSearchParams {
  const params = new URLSearchParams();
  const query = filters.query.trim();
  if (query) params.set("q", query);
  if (filters.courseId) params.set("curso", filters.courseId);
  return params;
}

function fold(value: string): string {
  return value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();
}

export function applyFilters(courses: Course[], filters: Filters): Course[] {
  const needle = fold(filters.query.trim());
  return courses
    .filter((course) => !filters.courseId || course.id === filters.courseId)
    .map((course) => ({
      ...course,
      sections: course.sections
        .map((section) => ({
          ...section,
          notebooks: section.notebooks.filter(
            (notebook) =>
              !needle ||
              fold(
                `${notebook.title} ${notebook.summary} ${notebook.path}`,
              ).includes(needle),
          ),
        }))
        .filter((section) => section.notebooks.length > 0),
    }))
    .filter((course) => course.sections.length > 0);
}
