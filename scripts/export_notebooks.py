from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from notebooks_manifest import (
    ROOT,
    SITE_DIR,
    Notebook,
    Redirect,
    load_manifest,
    validate_manifest,
    write_frontend_data,
)

TMP_ROOT = Path(tempfile.gettempdir()).resolve()
MARIMO_RUN = ["uv", "run", "--locked"]
STATIC_NOTEBOOKS_DIR = SITE_DIR / "static" / "notebooks"

# Figures are embedded in the page as base64 PNG, so the page grows with every
# figure. marimo's frontend and fonts load from a shared CDN and do not count.
MAX_PAGE_BYTES = 1_000_000

# The export template includes marimo's favicon, manifest, touch icon, and two
# decorative image preloads. Each rule must match so a template change fails
# the build instead of publishing marimo's defaults.
EMPTY_FAVICON = '<link rel="icon" href="data:," />'
SHELL_RULES: tuple[tuple[str, str], ...] = (
    (r'<link rel="icon"[^>]*>', EMPTY_FAVICON),
    (r'[ \t]*<link rel="apple-touch-icon"[^>]*>\n?', ""),
    (r'[ \t]*<link rel="manifest"[^>]*>\n?', ""),
    (r'[ \t]*<link rel="preload"[^>]*as="image"[^>]*>\n?', ""),
)


def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def clean_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        return

    inside_repo = ROOT in output_dir.parents
    inside_tmp = TMP_ROOT in output_dir.parents
    if not inside_repo and not inside_tmp:
        raise SystemExit(
            f"Refusing to clean a path outside the repository: {output_dir}"
        )
    if output_dir in {ROOT, TMP_ROOT, SITE_DIR.resolve()}:
        raise SystemExit(f"Refusing to clean root directory: {output_dir}")

    shutil.rmtree(output_dir)


def replace_all(pattern: str, replacement: str, page: str, what: str) -> str:
    page, count = re.subn(pattern, lambda _: replacement, page)
    if count == 0:
        raise SystemExit(f"marimo export template changed: no {what} to rewrite")
    return page


def apply_page_shell(page: str, notebook: Notebook) -> str:
    """Give an exported page the catalog's language, title and description."""
    page = replace_all(r'<html lang="[^"]*"', '<html lang="es"', page, "html lang")
    page = replace_all(
        r"<title>[^<]*</title>",
        f"<title>{html.escape(notebook.title)}</title>",
        page,
        "title",
    )
    page = replace_all(
        r'<meta name="description"[^>]*>',
        f'<meta name="description" content="{html.escape(notebook.summary)}" />',
        page,
        "description",
    )
    for pattern, replacement in SHELL_RULES:
        page = replace_all(pattern, replacement, page, pattern)
    return page


def export_results(notebook: Notebook, output_dir: Path) -> None:
    """Run the notebook and write its results as one static HTML page.

    `marimo export html` exits non-zero when a cell fails. The page is then
    discarded so a build cannot publish it. So is a page larger than
    `MAX_PAGE_BYTES`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    page_path = output_dir / "index.html"
    status = run(
        [
            *MARIMO_RUN,
            "--package",
            notebook.package,
            "marimo",
            "export",
            "html",
            notebook.path,
            "-o",
            str(page_path),
            "-f",
        ]
    )
    if status:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise SystemExit(f"Export failed for {notebook.path}")

    page = apply_page_shell(page_path.read_text(encoding="utf-8"), notebook)
    size = len(page.encode("utf-8"))
    if size > MAX_PAGE_BYTES:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise SystemExit(
            f"{notebook.path} exports a {size}-byte page, over the "
            f"{MAX_PAGE_BYTES}-byte limit. Lower the figure resolution or "
            "plot fewer points."
        )
    page_path.write_text(page, encoding="utf-8")


REDIRECT_PAGE = """\
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <title>Pagina movida</title>
    <meta name="robots" content="noindex" />
    <meta http-equiv="refresh" content="0; url={url}" />
  </head>
  <body>
    <p>Esta pagina se movio. <a href="{url}">Ir a la nueva direccion</a>.</p>
  </body>
</html>
"""


def write_redirect(redirect: Redirect, output_dir: Path) -> None:
    """Write a static page that sends readers from a retired address to its target.

    The link is relative so the page works under any base path the site is served
    from.
    """
    url = html.escape(posixpath.relpath(redirect.target, redirect.source) + "/")
    page_dir = output_dir / redirect.source
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "index.html").write_text(
        REDIRECT_PAGE.format(url=url), encoding="utf-8"
    )


def export_notebooks(clean: bool) -> None:
    site, notebooks, redirects = load_manifest()
    validate_manifest(notebooks, redirects)
    write_frontend_data(site, notebooks)

    if clean:
        clean_output_dir(STATIC_NOTEBOOKS_DIR)

    STATIC_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for notebook in notebooks:
        if not notebook.export:
            continue

        check = run(
            [
                *MARIMO_RUN,
                "--package",
                notebook.package,
                "marimo",
                "check",
                notebook.path,
            ]
        )
        if check:
            raise SystemExit(f"Check failed for {notebook.path}")
        export_results(notebook, STATIC_NOTEBOOKS_DIR / notebook.slug)

    for redirect in redirects:
        write_redirect(redirect, STATIC_NOTEBOOKS_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and export published notebooks."
    )
    parser.add_argument(
        "command",
        choices=["validate", "data", "export"],
        help="Action to run.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove exported notebook assets before exporting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site, notebooks, redirects = load_manifest()
    validate_manifest(notebooks, redirects)

    if args.command == "validate":
        return 0
    if args.command == "data":
        write_frontend_data(site, notebooks)
        return 0
    if args.command == "export":
        export_notebooks(args.clean)
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
