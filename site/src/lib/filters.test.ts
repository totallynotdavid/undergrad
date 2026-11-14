import { expect, test } from "vitest";

import type { Topic } from "./catalog.generated";
import {
  applyFilters,
  filtersToSearchParams,
  parseFilters,
  type Filters,
} from "./filters";

const topics: Topic[] = [
  {
    id: "dynamics",
    title: "Dynamics",
    folderUrl: "https://example.test/dynamics",
    notebooks: [
      nb("Lorenz attractor", "A classic chaotic system", "a/lorenz.py"),
      nb("Logistic map", "Bifurcations and period doubling", "a/logistic.py"),
    ],
  },
  {
    id: "statistical-physics",
    title: "Statistical physics",
    folderUrl: "https://example.test/statistical-physics",
    notebooks: [nb("Ising model", "Metropolis on a lattice", "b/ising.py")],
  },
];

function nb(title: string, summary: string, path: string) {
  return { title, summary, slug: path, path, sourceUrl: "", assetPath: "" };
}

function count(result: Topic[]): number {
  return result.reduce((sum, topic) => sum + topic.notebooks.length, 0);
}

function filters(partial: Partial<Filters>): Filters {
  return { query: "", topicId: "", ...partial };
}

test("parseFilters reads the two query keys with empty defaults", () => {
  expect(parseFilters(new URLSearchParams("q=lorenz&topic=dynamics"))).toEqual({
    query: "lorenz",
    topicId: "dynamics",
  });
  expect(parseFilters(new URLSearchParams(""))).toEqual({
    query: "",
    topicId: "",
  });
});

test("filtersToSearchParams omits empty and trimmed-empty values", () => {
  expect(
    filtersToSearchParams(filters({ query: "  ", topicId: "" })).toString(),
  ).toBe("");
  expect(
    filtersToSearchParams(
      filters({ query: "  lorenz  ", topicId: "dynamics" }),
    ).toString(),
  ).toBe("q=lorenz&topic=dynamics");
});

test("applyFilters keeps the notebooks matching the topic and the folded query", () => {
  const cases: [string, Partial<Filters>, number][] = [
    ["empty filters keep every notebook", {}, 3],
    ["query folds diacritics in the title", { query: "LÓRENZ" }, 1],
    ["query matches the summary", { query: "metropolis" }, 1],
    ["query matches the path", { query: "ising.py" }, 1],
    ["topic keeps only its notebooks", { topicId: "statistical-physics" }, 1],
    ["topic and query intersect", { topicId: "dynamics", query: "lorenz" }, 1],
    [
      "topic and query disjoint",
      { topicId: "statistical-physics", query: "lorenz" },
      0,
    ],
  ];
  for (const [name, partial, expected] of cases) {
    expect(count(applyFilters(topics, filters(partial))), name).toBe(expected);
  }
});

test("a filter that matches nothing yields an empty catalog, not empty topics", () => {
  expect(applyFilters(topics, filters({ query: "zzz" }))).toEqual([]);
  const pruned = applyFilters(topics, filters({ query: "lorenz" }));
  expect(pruned.map((topic) => topic.id)).toEqual(["dynamics"]);
});
