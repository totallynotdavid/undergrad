from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from notebooks_manifest import (
    ROOT,
    SITE_DIR,
    load_manifest,
    validate_manifest,
    write_frontend_data,
)

TMP_ROOT = Path(tempfile.gettempdir()).resolve()
MARIMO_RUN = ["uv", "run", "--locked", "--with", "marimo==0.23.9"]
STATIC_NOTEBOOKS_DIR = SITE_DIR / "static" / "notebooks"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


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


def export_notebooks(clean: bool) -> None:
    site, notebooks = load_manifest()
    validate_manifest(notebooks)
    write_frontend_data(site, notebooks)

    if clean:
        clean_output_dir(STATIC_NOTEBOOKS_DIR)

    STATIC_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for notebook in notebooks:
        if not notebook.export:
            continue

        output_dir = STATIC_NOTEBOOKS_DIR / notebook.slug
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                *MARIMO_RUN,
                "--package",
                notebook.package,
                "marimo",
                "check",
                notebook.path,
            ]
        )
        if notebook.mode == "results":
            export_static_results(notebook, output_dir)
        else:
            export_interactive(notebook, output_dir)


def export_interactive(notebook, output_dir: Path) -> None:
    run(
        [
            *MARIMO_RUN,
            "--package",
            notebook.package,
            "marimo",
            "export",
            "html-wasm",
            notebook.path,
            "-o",
            str(output_dir),
            "--mode",
            "run",
            "-f",
        ]
    )


def export_static_results(notebook, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            *MARIMO_RUN,
            "--package",
            notebook.package,
            "marimo",
            "export",
            "html",
            notebook.path,
            "-o",
            str(output_dir / "index.html"),
        ]
    )


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
    site, notebooks = load_manifest()
    validate_manifest(notebooks)

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
