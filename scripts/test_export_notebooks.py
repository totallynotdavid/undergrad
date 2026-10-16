from __future__ import annotations

import dataclasses
import re
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from urllib.parse import urljoin

import export_notebooks
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

    def test_a_valid_redirect_passes_validation(self):
        errors = redirect_errors(
            self.notebooks("curso/nuevo/pagina"),
            [Redirect("curso/viejo/pagina", "curso/nuevo/pagina")],
        )

        self.assertEqual(errors, [])

    def test_a_redirect_must_point_to_an_exported_notebook(self):
        errors = redirect_errors(
            self.notebooks("curso/nuevo/pagina", export=False),
            [Redirect("curso/viejo/pagina", "curso/nuevo/pagina")],
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("not an exported notebook", errors[0])

    def test_a_redirect_cannot_replace_or_sit_above_or_inside_a_notebook_page(self):
        pages = self.notebooks("curso/nuevo/pagina")
        for source in ("curso/nuevo/pagina", "curso/nuevo", "curso/nuevo/pagina/x"):
            with self.subTest(source=source):
                errors = redirect_errors(
                    pages, [Redirect(source, "curso/nuevo/pagina")]
                )

                self.assertEqual(len(errors), 1)
                self.assertIn("collides", errors[0])

    def test_two_redirects_cannot_share_an_address(self):
        errors = redirect_errors(
            self.notebooks("curso/nuevo/pagina"),
            [Redirect("curso/viejo/pagina", "curso/nuevo/pagina")] * 2,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Duplicate", errors[0])


if __name__ == "__main__":
    unittest.main()
