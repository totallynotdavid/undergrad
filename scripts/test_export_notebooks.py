from __future__ import annotations

import dataclasses
import re
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest import mock
from urllib.parse import urljoin

import export_notebooks
import notebooks_manifest
from notebooks_manifest import Notebook, Redirect, redirect_errors

NOTEBOOK = dedent(
    """\
    import marimo

    app = marimo.App()


    @app.cell
    def _():
        import marimo as mo

        mo.md("# Hola")
        return


    {extra}


    if __name__ == "__main__":
        app.run()
    """
)
FAILING_CELL = dedent(
    """\
    @app.cell
    def _():
        raise ValueError("boom")
        return
    """
)


OVERSIZED_CELL = dedent(
    f"""\
    @app.cell
    def _():
        import marimo as _mo

        _mo.md("a" * {export_notebooks.MAX_PAGE_BYTES + 1})
        return
    """
)


def catalog_entry(path: Path) -> Notebook:
    return Notebook(
        course_id="curso",
        course="Curso",
        section_id="seccion",
        section="Seccion",
        title="Titulo <de> prueba",
        summary="Resumen de prueba",
        path=str(path),
        package="fisica-computacional",
        package_dir="packages/fisica-computacional",
        slug="curso/seccion/prueba",
        export=True,
    )


class ExportResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def write_notebook(self, extra: str) -> Notebook:
        path = self.root / "prueba.py"
        path.write_text(NOTEBOOK.format(extra=extra), encoding="utf-8")
        return catalog_entry(path)

    def test_page_carries_the_catalog_title_language_and_no_marimo_icons(self):
        notebook = self.write_notebook("")
        output = self.root / "out"

        export_notebooks.export_results(notebook, output)

        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn('<html lang="es"', page)
        self.assertIn("<title>Titulo &lt;de&gt; prueba</title>", page)
        self.assertIn('content="Resumen de prueba"', page)
        self.assertIn('<link rel="icon" href="data:," />', page)
        self.assertNotIn("a marimo app", page)
        self.assertNotIn('rel="manifest"', page)
        self.assertNotIn('rel="apple-touch-icon"', page)
        self.assertNotIn('as="image"', page)

    def test_a_failing_cell_fails_the_export_and_publishes_nothing(self):
        notebook = self.write_notebook(FAILING_CELL)
        output = self.root / "out"

        with self.assertRaises(SystemExit):
            export_notebooks.export_results(notebook, output)

        self.assertFalse(output.exists())

    def test_a_page_over_the_size_limit_fails_the_export_and_publishes_nothing(self):
        notebook = self.write_notebook(OVERSIZED_CELL)
        output = self.root / "out"

        with self.assertRaises(SystemExit) as raised:
            export_notebooks.export_results(notebook, output)

        self.assertIn("limit", str(raised.exception))
        self.assertFalse(output.exists())

    def test_a_changed_marimo_template_fails_instead_of_publishing_defaults(self):
        notebook = self.write_notebook("")

        with self.assertRaises(SystemExit):
            export_notebooks.apply_page_shell("<html><body></body></html>", notebook)


class CatalogDataTest(unittest.TestCase):
    def run_data(self, generated: Path, *flags: str) -> int:
        argv = ["export_notebooks.py", "data", *flags]
        with (
            mock.patch.object(notebooks_manifest, "GENERATED_DATA", generated),
            mock.patch.object(sys, "argv", argv),
        ):
            return export_notebooks.main()

    def test_check_fails_on_a_stale_catalog_without_writing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "catalog.generated.ts"
            generated.write_text("stale", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                self.run_data(generated, "--check")

            self.assertIn("mise run data", str(raised.exception))
            self.assertEqual(generated.read_text(encoding="utf-8"), "stale")

    def test_check_passes_on_the_catalog_that_data_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "catalog.generated.ts"
            self.run_data(generated)
            written = generated.read_text(encoding="utf-8")

            self.assertEqual(self.run_data(generated, "--check"), 0)
            self.assertEqual(generated.read_text(encoding="utf-8"), written)


class RedirectTest(unittest.TestCase):
    def notebooks(self, *slugs: str, export: bool = True) -> list[Notebook]:
        return [
            dataclasses.replace(catalog_entry(Path(slug)), slug=slug, export=export)
            for slug in slugs
        ]

    def test_redirect_page_resolves_to_the_new_page_under_any_base_path(self):
        redirect = Redirect(source="curso/viejo/pagina", target="curso/nuevo/pagina")

        with tempfile.TemporaryDirectory() as directory:
            export_notebooks.write_redirect(redirect, Path(directory))
            page = (Path(directory) / redirect.source / "index.html").read_text(
                encoding="utf-8"
            )

        url = re.search(r'http-equiv="refresh" content="0; url=([^"]+)"', page)
        self.assertIsNotNone(url)
        for base in ("https://sitio.test", "https://sitio.test/undergrad"):
            old_page = f"{base}/notebooks/{redirect.source}/"
            self.assertEqual(
                urljoin(old_page, url.group(1)), f"{base}/notebooks/{redirect.target}/"
            )
        self.assertIn(f'<a href="{url.group(1)}">', page)

    def test_redirect_validation(self):
        new = "curso/nuevo/pagina"
        old = Redirect("curso/viejo/pagina", new)
        cases = {
            "valid": (True, [old], None),
            "target is not exported": (False, [old], "not an exported notebook"),
            "replaces a page": (True, [Redirect(new, new)], "collides"),
            "sits above a page": (True, [Redirect("curso/nuevo", new)], "collides"),
            "sits inside a page": (True, [Redirect(f"{new}/x", new)], "collides"),
            "duplicate address": (True, [old, old], "Duplicate"),
        }
        for name, (export, redirects, message) in cases.items():
            with self.subTest(name):
                errors = redirect_errors(self.notebooks(new, export=export), redirects)

                if message is None:
                    self.assertEqual(errors, [])
                else:
                    self.assertEqual(len(errors), 1)
                    self.assertIn(message, errors[0])


if __name__ == "__main__":
    unittest.main()
