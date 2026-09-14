# Contributing

Use a focused branch and keep model-behaviour changes separate from packaging
or documentation changes. Every pull request that changes numerical behaviour
must include a regression test and report the affected checkpoint compatibility.

Before submitting:

```bash
python -m pytest
ruff check .
```

Do not commit datasets, checkpoints, credentials, cluster account identifiers,
or generated experiment outputs. Report security issues privately as described
in `SECURITY.md`.
