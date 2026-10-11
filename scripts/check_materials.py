"""Validate lesson links and execute all runnable Python examples locally."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    files = [ROOT / "README.md", ROOT / "CHEATSHEET.md", ROOT / "READING_LIST.md"]
    files += sorted((ROOT / "chapters").glob("*.md"))
    files += sorted((ROOT / "answers").glob("*.md"))
    links = examples = diagrams = 0
    for path in files:
        source = path.read_text()
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", source):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                raise ValueError(f"broken link: {path.name}: {target}")
            links += 1
        # Code snippets within one document share variables (some answers use
        # the model created in an earlier block). Separate documents do not.
        namespace = {"__name__": "lesson_example"}
        for code in re.findall(r"```python\n(.*?)```", source, re.S):
            with redirect_stdout(StringIO()):
                exec(compile(code, str(path), "exec"), namespace)
            examples += 1
        diagrams += source.count("```mermaid\n")
    print(
        f"Checked {len(files)} documents, {links} local links, "
        f"{examples} Python examples; {diagrams} Mermaid diagrams found."
    )


if __name__ == "__main__":
    main()
