from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "notebooks.toml"
SITE_DIR = ROOT / "site"
GENERATED_DATA = SITE_DIR / "src" / "lib" / "notebooks.generated.json"

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:[-/][a-z0-9]+)*$")


@dataclass(frozen=True)
class SiteConfig:
    title: str
    description: str
    repository: str
    branch: str


@dataclass(frozen=True)
class Notebook:
    course_id: str
    course: str
    section_id: str
    section: str
    title: str
    summary: str
    path: str
    package: str
    slug: str
    export: bool


def load_manifest() -> tuple[SiteConfig, list[Notebook]]:
    with MANIFEST.open("rb") as file:
        data = tomllib.load(file)

    site_data = data["site"]
    site = SiteConfig(
        title=site_data["title"],
        description=site_data["description"],
        repository=site_data["repository"].rstrip("/"),
        branch=site_data["branch"],
    )
    root = data.get("defaults", {}).get("root", "notebooks").strip("/")
    notebooks = expand_notebooks(root, data.get("courses", []))
    return site, notebooks


def expand_notebooks(root: str, courses: list[dict]) -> list[Notebook]:
    notebooks: list[Notebook] = []
    for course in courses:
        course_id = course["id"]
        package = course.get("package", course_id)
        for section in course.get("sections", []):
            section_id = section["id"]
            for entry in section.get("notebooks", []):
                notebook_id = entry["id"]
                file_path = entry["file"].strip("/")
                notebooks.append(
                    Notebook(
                        course_id=course_id,
                        course=course["title"],
                        section_id=section_id,
                        section=section["title"],
                        title=entry["title"],
                        summary=entry["summary"],
                        path=f"{root}/{package}/{file_path}",
                        package=package,
                        slug=f"{course_id}/{section_id}/{notebook_id}",
                        export=entry["export"],
                    )
                )
    return notebooks


def validate_manifest(notebooks: list[Notebook]) -> None:
    slugs: set[str] = set()
    paths: set[str] = set()
    errors: list[str] = []

    for notebook in notebooks:
        if notebook.slug in slugs:
            errors.append(f"Duplicate slug: {notebook.slug}")
        slugs.add(notebook.slug)

        if notebook.path in paths:
            errors.append(f"Duplicate path: {notebook.path}")
        paths.add(notebook.path)

        if not SLUG_PATTERN.fullmatch(notebook.slug):
            errors.append(f"Invalid slug: {notebook.slug}")

        if Path(notebook.path).is_absolute() or ".." in Path(notebook.path).parts:
            errors.append(f"Invalid notebook path: {notebook.path}")

        notebook_path = ROOT / notebook.path
        if not notebook_path.exists():
            errors.append(f"Missing notebook: {notebook.path}")

        if notebook.export and not notebook.package:
            errors.append(f"Missing package for exported notebook: {notebook.path}")

    if errors:
        raise SystemExit("\n".join(errors))


def write_frontend_data(site: SiteConfig, notebooks: list[Notebook]) -> None:
    GENERATED_DATA.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": asdict(site),
        "notebooks": [
            {
                **asdict(notebook),
                "sourceUrl": source_url(site, notebook),
                "assetPath": f"/notebooks/{notebook.slug}/" if notebook.export else "",
            }
            for notebook in notebooks
        ],
    }
    GENERATED_DATA.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def source_url(site: SiteConfig, notebook: Notebook) -> str:
    encoded_path = quote(notebook.path)
    return f"{site.repository}/blob/{site.branch}/{encoded_path}"
