import tailwindcss from "@tailwindcss/vite";
import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

const configuredBase = process.env.BASE_PATH ?? "";
const base = (
  configuredBase === ""
    ? ""
    : configuredBase.startsWith("/")
      ? configuredBase
      : `/${configuredBase}`
) as "" | `/${string}`;

export default defineConfig({
  plugins: [
    tailwindcss(),
    sveltekit({
      compilerOptions: {
        // Force runes mode for the project, except for libraries. Can be removed in svelte 6.
        runes: ({ filename }) =>
          filename.split(/[/\\]/).includes("node_modules") ? undefined : true,
        experimental: { async: true },
      },
      adapter: adapter(),
      paths: {
        base,
      },
      experimental: { remoteFunctions: true, handleRenderingErrors: true },
    }),
  ],
});
