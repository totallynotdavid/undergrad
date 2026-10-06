from __future__ import annotations

import subprocess
import sys

from notebooks_manifest import ROOT, load_manifest, validate_manifest

MARIMO_VERSION = "0.25.0"


def main() -> int:
    _, notebooks = load_manifest()
    validate_manifest(notebooks)
    failures = 0

    for notebook in notebooks:
        command = [
            "uvx",
            "--from",
            f"marimo=={MARIMO_VERSION}",
            "marimo",
            "check",
            notebook.path,
        ]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            failures += 1
            print(f"FAILED: {notebook.path}", flush=True)
        else:
            print(f"PASSED: {notebook.path}", flush=True)

    if failures:
        print(f"{failures} notebook check(s) failed.", file=sys.stderr)
        return 1

    print(f"Checked {len(notebooks)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
