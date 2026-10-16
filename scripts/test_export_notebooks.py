from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

import export_notebooks
from notebooks_manifest import Notebook

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
        package="fisica-general",
        package_dir="packages/fisica-general",
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


if __name__ == "__main__":
    unittest.main()
