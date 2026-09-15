#!/usr/bin/env python
# tools/export_report.py
"""Export the leaderboard to a paper-ready results table.

Formats: latex (booktabs), markdown, csv.

Usage:
    python tools/export_report.py --format latex
    python tools/export_report.py --format markdown --variant small,base,large
    python tools/export_report.py --format csv --all --out results_table.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard_lib import HEADLINE, LOWER_IS_BETTER, _finite, load_leaderboard

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"


def _fmt(v):
    if v is None or not _finite(v):
        return ""
    return f"{v:.4f}"


def _pick(rows, args):
    if args.model_ids:
        ids = set(args.model_ids)
        rows = [r for r in rows if r.get("model_id") in ids]
    elif args.variant:
        vs = {v.strip() for v in args.variant.split(",")}
        rows = [r for r in rows if r.get("variant") in vs]
    elif args.family:
        rows = [r for r in rows if r.get("config_hash") == args.family]
    seen = {}
    for r in rows:
        seen.setdefault(r.get("model_id"), r)
    return list(seen.values())


def _best_idx(k, vals):
    idxs = [i for i, v in enumerate(vals) if _finite(v)]
    if not idxs:
        return None
    return (min if k in LOWER_IS_BETTER else max)(idxs, key=lambda i: vals[i])


def _macro_table(rows):
    labels = [r.get("model_id") for r in rows]
    means = [r.get("metrics", {}).get("mean", {}) for r in rows]
    table = []
    for k in HEADLINE:
        vals = [m.get(k) for m in means]
        table.append((k, vals))
    return labels, table


def _per_dataset_table(rows, metric):
    labels = [r.get("model_id") for r in rows]
    per_ds = [r.get("metrics", {}).get("per_dataset", {}) for r in rows]
    ds_order = sorted({ds for p in per_ds for ds in p})
    return labels, [(ds, [p.get(ds, {}).get(metric) for p in per_ds]) for ds in ds_order]


def _render_latex(title, labels, table):
    n = len(labels)
    colspec = "l" + "c" * n
    out = []
    out.append("\\begin{table}[t]")
    out.append("  \\centering")
    out.append(f"  \\caption{{{title}}}")
    out.append(f"  \\label{{tab:{title.replace(' ', '-').replace(':', '').lower()}}}")
    out.append(f"  \\begin{{tabular}}{{{colspec}}}")
    out.append("    \\toprule")
    out.append("    " + " & ".join(["Metric"] + labels) + " \\\\")
    out.append("    \\midrule")
    for row, vals in table:
        best = _best_idx(row, vals) if n > 1 else None
        cells = []
        for i, v in enumerate(vals):
            s = _fmt(v)
            if best is not None and i == best and s:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append("    " + " & ".join([row] + cells) + " \\\\")
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def _render_markdown(title, labels, table):
    n = len(labels)
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|--------|" + "--------|" * n
    out = [f"### {title}", "", header, sep]
    for row, vals in table:
        best = _best_idx(row, vals) if n > 1 else None
        cells = []
        for i, v in enumerate(vals):
            s = _fmt(v)
            if best is not None and i == best and s:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join([row] + cells) + " |")
    return "\n".join(out)


def _render_csv(title, labels, table):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([title])
    w.writerow(["Metric"] + labels)
    for row, vals in table:
        w.writerow([row] + [_fmt(v) for v in vals])
    return buf.getvalue()


def _err(msg):
    print(msg, file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_ids", nargs="*", help="explicit model_id(s) to export")
    ap.add_argument("--all", action="store_true",
                    help="export every registered model (default when no selection given)")
    ap.add_argument("--variant", default=None, help="comma-separated variants")
    ap.add_argument("--family", default=None, help="config_hash to export seeds of")
    ap.add_argument("--format", default="latex", choices=["latex", "markdown", "csv"])
    ap.add_argument("--out", default=None, help="output file (default: stdout)")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    args = ap.parse_args()

    rows = load_leaderboard(args.leaderboard)
    if not rows:
        return _err(f"no rows in {args.leaderboard}; run register_eval.py first")
    picked = _pick(rows, args)
    if not picked:
        return _err("no models matched the selection")

    render = {"latex": _render_latex, "markdown": _render_markdown, "csv": _render_csv}[args.format]

    sections = [render("Macro metrics (mean over 8 datasets x 6 horizons)",
                       *_macro_table(picked))]
    for metric in ("nCRPS", "nMAE"):
        sections.append(render(f"Per-dataset {metric}",
                               *_per_dataset_table(picked, metric)))

    text = "\n\n".join(sections) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
