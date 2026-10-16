import { expect, test } from "vitest";

import type { Course } from "./catalog.generated";
import {
  applyFilters,
  filtersToSearchParams,
  parseFilters,
  type Filters,
} from "./filters";

const courses: Course[] = [
  {
    id: "dinamica",
    title: "Dinamica",
    folderUrl: "https://example.test/dinamica",
    sections: [
      {
        id: "caos",
        title: "Caos",
        notebooks: [
          nb(
            "Atractor de Lorenz",
            "Sistema caotico clasico",
            "a/lorenz.py",
            true,
          ),
          nb(
            "Mapa logistico",
            "Bifurcaciones y periodo",
            "a/logistico.py",
            false,
          ),
        ],
      },
    ],
  },
  {
    id: "estadistica",
    title: "Mecanica estadistica",
    folderUrl: "https://example.test/estadistica",
    sections: [
      {
        id: "montecarlo",
        title: "Monte Carlo",
        notebooks: [
          nb("Modelo de Ising", "Metropolis en una red", "b/ising.py", true),
        ],
      },
    ],
  },
];

function nb(title: string, summary: string, path: string, exported: boolean) {
  return {
    title,
    summary,
    slug: path,
    path,
    export: exported,
    sourceUrl: "",
    assetPath: "",
  };
}

function count(result: Course[]): number {
  return result.reduce(
    (sum, course) =>
      sum +
      course.sections.reduce((n, section) => n + section.notebooks.length, 0),
    0,
  );
}

function filters(partial: Partial<Filters>): Filters {
  return { query: "", courseId: "", ...partial };
}

test("parseFilters reads the two query keys with empty defaults", () => {
  expect(parseFilters(new URLSearchParams("q=lorenz&curso=dinamica"))).toEqual({
    query: "lorenz",
    courseId: "dinamica",
  });
  expect(parseFilters(new URLSearchParams(""))).toEqual({
    query: "",
    courseId: "",
  });
});

test("filtersToSearchParams omits empty and trimmed-empty values", () => {
  expect(
    filtersToSearchParams(filters({ query: "  ", courseId: "" })).toString(),
  ).toBe("");
  expect(
    filtersToSearchParams(
      filters({ query: "  lorenz  ", courseId: "dinamica" }),
    ).toString(),
  ).toBe("q=lorenz&curso=dinamica");
});

test("applyFilters keeps the notebooks matching the course and the folded query", () => {
  const cases: [string, Partial<Filters>, number][] = [
    ["empty filters keep every notebook", {}, 3],
    ["query folds diacritics in the title", { query: "LÓRENZ" }, 1],
    ["query matches the summary", { query: "metropolis" }, 1],
    ["query matches the path", { query: "ising.py" }, 1],
    ["course keeps only its notebooks", { courseId: "estadistica" }, 1],
    [
      "course and query intersect",
      { courseId: "dinamica", query: "lorenz" },
      1,
    ],
    [
      "course and query disjoint",
      { courseId: "estadistica", query: "lorenz" },
      0,
    ],
  ];
  for (const [name, partial, expected] of cases) {
    expect(count(applyFilters(courses, filters(partial))), name).toBe(expected);
  }
});

test("a filter that matches nothing yields an empty catalog, not empty sections", () => {
  expect(applyFilters(courses, filters({ query: "zzz" }))).toEqual([]);
  const pruned = applyFilters(
    courses,
    filters({ courseId: "dinamica", query: "lorenz" }),
  );
  expect(pruned.map((course) => course.id)).toEqual(["dinamica"]);
  expect(pruned[0].sections.map((section) => section.id)).toEqual(["caos"]);
});
