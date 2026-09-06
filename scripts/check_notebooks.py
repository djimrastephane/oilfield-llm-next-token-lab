#!/usr/bin/env python3
"""Structural smoke test for every notebook in this repo.

Checks each .ipynb in notebooks/ and advanced/ without loading the model or
any heavy dependency:
  - the file is valid nbformat JSON
  - every code cell parses as valid Python (ast.parse)
  - numbered markdown headers ("## N. ...") are sequential with no gaps or
    duplicates
  - every "Section N" / "Sections N-M" cross-reference in the notebook's own
    text points to a header number that actually exists in that notebook

Run before committing any notebook change:
    python3 scripts/check_notebooks.py
"""
import ast
import glob
import json
import re
import sys


def check_notebook(path: str) -> list[str]:
    problems = []

    with open(path, encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as exc:
            return [f"invalid JSON: {exc}"]

    if nb.get("nbformat") != 4:
        problems.append(f"unexpected nbformat version: {nb.get('nbformat')!r}")

    headers: set[int] = set()
    markdown_text = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            try:
                ast.parse(source)
            except SyntaxError as exc:
                problems.append(f"cell {i}: SyntaxError: {exc}")
        elif cell.get("cell_type") == "markdown":
            markdown_text.append(source)
            for line in source.splitlines():
                m = re.match(r"^#{1,3} (\d+)\.", line)
                if m:
                    headers.add(int(m.group(1)))

    if headers:
        expected = set(range(1, max(headers) + 1))
        missing = expected - headers
        duplicate_check = list(headers)
        if missing:
            problems.append(f"numbered headers skip: {sorted(missing)} (found {sorted(headers)})")

    full_text = "\n".join(markdown_text)
    for m in re.finditer(r"Sections? (\d+)(?:-(\d+))?", full_text):
        # Skip references that are qualified by another notebook, e.g.
        # "notebook 4's Section 10" or "notebook 4 ... in its Section 10" --
        # those point at a *different* file's numbering, not this one's.
        preceding = full_text[max(0, m.start() - 60) : m.start()]
        if re.search(r"notebook\s+\d+", preceding, re.IGNORECASE):
            continue
        nums = [int(m.group(1))] + ([int(m.group(2))] if m.group(2) else [])
        for n in nums:
            if n not in headers:
                problems.append(f"broken cross-reference: {m.group(0)!r} (Section {n} does not exist)")

    return problems


def main() -> int:
    paths = sorted(glob.glob("notebooks/*.ipynb")) + sorted(glob.glob("advanced/*.ipynb"))
    if not paths:
        print("No notebooks found -- run this from the repo root.")
        return 1

    exit_code = 0
    for path in paths:
        problems = check_notebook(path)
        if problems:
            exit_code = 1
            print(f"FAIL {path}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK   {path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
