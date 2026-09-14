"""Create a metadata-only inventory of DeepWind assets.

The tool never opens array or checkpoint payloads. It records paths, sizes,
timestamps, file counts, extensions, and largest files for migration planning.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def inventory(root: Path, largest: int) -> dict:
    extensions: Counter[str] = Counter()
    files: list[tuple[int, str, float]] = []
    logical_bytes = 0
    errors: list[str] = []

    for base, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(base) / name
            try:
                stat = path.lstat()
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            logical_bytes += stat.st_size
            extensions[path.suffix.lower() or "<none>"] += 1
            files.append((stat.st_size, str(path), stat.st_mtime))

    files.sort(reverse=True)
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "extensions": dict(extensions.most_common()),
        "largest_files": [
            {
                "path": path,
                "bytes": size,
                "modified_utc": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
            }
            for size, path, mtime in files[:largest]
        ],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--largest", type=int, default=50)
    args = parser.parse_args()

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "roots": [inventory(root, args.largest) for root in args.roots],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
