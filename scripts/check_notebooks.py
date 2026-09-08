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
    header_occurrences: list[int] = []
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
                    n = int(m.group(1))
                    headers.add(n)
                    header_occurrences.append(n)

    if headers:
        expected = set(range(1, max(headers) + 1))
        missing = expected - headers
        if missing:
            problems.append(f"numbered headers skip: {sorted(missing)} (found {sorted(headers)})")
        duplicates = sorted(n for n in headers if header_occurrences.count(n) > 1)
        if duplicates:
            problems.append(f"duplicate numbered headers: {duplicates}")

    full_text = "\n".join(markdown_text)
    for m in re.finditer(r"Sections? (\d+)(?:\s*(?:-|through)\s*(\d+))?", full_text):
        # A "Section N" reference can be explicitly marked as pointing at
        # *this* notebook ("... Section 12 here") or *another* one ("...
        # Section 9 there") -- those markers take precedence over anything
        # else nearby, since a notebook can mention its own and another
        # notebook's sections in the same sentence (e.g. "notebook 1
        # (Section 9 there) ... Section 12 here").
        following = full_text[m.end() : m.end() + 15]
        if re.match(r"\s+there\b", following):
            continue
        if not re.match(r"\s+here\b", following):
            # No explicit marker: fall back to whether a "notebook N"
            # qualifier appears earlier in the *same sentence* -- bounded
            # by the previous sentence/paragraph break, not a fixed
            # character count, so it can't bleed into an unrelated,
            # later sentence's own self-reference.
            sentence_start = max(
                full_text.rfind(".", 0, m.start()) + 1,
                full_text.rfind("\n\n", 0, m.start()) + 1,
            )
            preceding = full_text[sentence_start : m.start()]
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
