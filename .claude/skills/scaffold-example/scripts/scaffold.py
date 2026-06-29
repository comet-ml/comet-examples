#!/usr/bin/env python3
"""Scaffold a new Comet example by copying and renaming the canonical template.

One template lives under `templates/integration-example/`: a `pyproject.toml`, a single `comet_ml`
script, and a README in the house structure. It is a real, runnable example under a
sentinel identity (`example_integration` / `example-integration`). This script copies it to the
target directory and rewrites those sentinels to the new name — a pure identifier rename, no
template language. Run it, then fill in the TODO stubs.

  python scaffold.py pytorch-amp-example --description "..." --dest integrations/model-training/pytorch
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

TEMPLATE_DIR = ("templates", "integration-example")
PKG_SENTINEL = "example_integration"  # snake_case: the .py module name
PROJECT_SENTINEL = "example-integration"  # kebab-case: folder + comet project name

EXCLUDE = shutil.ignore_patterns(
    ".venv", "uv.lock", "__pycache__", "*.pyc", ".ruff_cache", ".env", "*.log", ".tmp"
)


def to_snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    if not s:
        raise ValueError(f"cannot derive a name from {name!r}")
    if s[0].isdigit():
        s = f"x_{s}"
    return s


def to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


def is_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new Comet example from templates/integration-example."
    )
    parser.add_argument("name", help="Example name, e.g. 'pytorch-amp-example'.")
    parser.add_argument(
        "--description",
        default=None,
        help="One-line description (written to pyproject.toml).",
    )
    parser.add_argument(
        "--dest",
        default="integrations",
        help="Parent directory for the new example (default: integrations). "
        "Relative paths are resolved from the repo root; absolute paths are used as-is.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (default: inferred from this script).",
    )
    args = parser.parse_args()

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[4]
    )
    template = repo_root.joinpath(*TEMPLATE_DIR)

    pkg = to_snake(args.name)
    kebab = to_kebab(pkg)

    if not template.is_dir():
        print(f"error: template not found at {template}", file=sys.stderr)
        return 1

    dest_parent = (
        Path(args.dest) if Path(args.dest).is_absolute() else repo_root / args.dest
    )
    dest = dest_parent / kebab
    if dest.exists():
        print(f"error: {dest} already exists — refusing to overwrite.", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, dest, ignore=EXCLUDE)

    # Rename the module file (script kind).
    mod_file = dest / f"{PKG_SENTINEL}.py"
    if mod_file.is_file():
        mod_file.rename(dest / f"{pkg}.py")

    # Rewrite sentinels in every text file. Snake and kebab sentinels don't overlap
    # (underscore vs hyphen), so independent replacement is safe.
    for path in dest.rglob("*"):
        if not path.is_file() or not is_text(path):
            continue
        text = path.read_text(encoding="utf-8")
        new = text.replace(PKG_SENTINEL, pkg).replace(PROJECT_SENTINEL, kebab)
        if new != text:
            path.write_text(new, encoding="utf-8")

    # Description -> pyproject.toml.
    if args.description:
        pyproject = dest / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        text = re.sub(
            r'^description = ".*"$',
            f'description = "{args.description}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        pyproject.write_text(text, encoding="utf-8")

    try:
        rel = dest.relative_to(repo_root)
    except ValueError:
        rel = dest
    print(f"Scaffolded {rel}")
    print(f"  folder:        {kebab}/")
    print(f"  script:        {pkg}.py")
    print(f"  comet project: comet-example-{kebab}")
    if args.description:
        print(f"  description:   {args.description}")
    print("\nNext steps:")
    print(f"  1. cd {rel} && uv sync")
    print(
        f"  2. Fill in {pkg}.py (real logic), pyproject.toml deps, and the README sections."
    )
    print(
        f"  3. Run it: uv run python {pkg}.py   (or: COMET_MODE=offline uv run python {pkg}.py)"
    )
    print("  4. To test it in CI, add it to .github/workflows/test-examples.yml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
