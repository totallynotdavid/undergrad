import type { RunMode } from "./catalog.generated";

export const runModeLabels: Record<RunMode, string> = {
  interactive: "Navegador",
  results: "Resultados",
  source: "Local",
};

export const runModeTitles: Record<RunMode, string> = {
  interactive: "Se ejecuta en el navegador al abrirlo",
  results: "Resultados exportados, solo lectura",
  source: "Requiere ejecucion local con uv",
};
