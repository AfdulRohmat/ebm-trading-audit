"""Verify frozen derived artifacts and repository-local Markdown links."""

import json
import re
from pathlib import Path

from ebm_audit.data import checksum

ROOT = Path(__file__).resolve().parents[1]


def main():
    hashes = json.loads((ROOT / "results/checksums.json").read_text())
    for name, expected in hashes.items():
        assert checksum(ROOT / "results" / name) == expected, name
    count = 0
    for path in [ROOT / "README.md", *list((ROOT / "docs").glob("*.md"))]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path.name, target)
            count += 1
    print(f"PASS: {len(hashes)} evidence hashes and {count} local Markdown links")


if __name__ == "__main__":
    main()
