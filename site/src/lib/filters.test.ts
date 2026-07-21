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
            "interactive",
          ),
          nb(
            "Mapa logistico",
            "Bifurcaciones y periodo",
            "a/logistico.py",
            "source",
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
          nb(
            "Modelo de Ising",
            "Metropolis en una red",
            "b/ising.py",
            "interactive",
          ),
        ],
      },
    ],
  },
];

function nb(
  title: string,
  summary: string,
  path: string,
  mode: Course["sections"][number]["notebooks"][number]["mode"],
) {
  return {
    title,
    summary,
    slug: path,
    path,
    export: mode !== "source",
    mode,
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

test("parse then serialize round-trips a trimmed query", () => {
  const params = filtersToSearchParams(filters({ query: "ising" }));
  expect(parseFilters(params)).toEqual({ query: "ising", courseId: "" });
});

test("empty filters return every course unchanged", () => {
  expect(count(applyFilters(courses, filters({})))).toBe(3);
});

test("search folds diacritics and matches title, summary, and path", () => {
  expect(count(applyFilters(courses, filters({ query: "LÓRENZ" })))).toBe(1);
  expect(count(applyFilters(courses, filters({ query: "metropolis" })))).toBe(
    1,
  );
  expect(count(applyFilters(courses, filters({ query: "ising.py" })))).toBe(1);
});

test("course filter keeps only the matching course", () => {
  const result = applyFilters(courses, filters({ courseId: "estadistica" }));
  expect(result.map((course) => course.id)).toEqual(["estadistica"]);
  expect(count(result)).toBe(1);
});

test("course and search intersect and prune empty sections and courses", () => {
  expect(
    count(
      applyFilters(courses, filters({ courseId: "dinamica", query: "lorenz" })),
    ),
  ).toBe(1);
  expect(
    count(
      applyFilters(
        courses,
        filters({ courseId: "estadistica", query: "lorenz" }),
      ),
    ),
  ).toBe(0);
  const result = applyFilters(
    courses,
    filters({ courseId: "dinamica", query: "lorenz" }),
  );
  expect(
    result.every((course) =>
      course.sections.every((section) => section.notebooks.length > 0),
    ),
  ).toBe(true);
});

test("no match yields an empty catalog, not empty sections", () => {
  expect(applyFilters(courses, filters({ query: "zzz" }))).toEqual([]);
});
