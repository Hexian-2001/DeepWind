#!/usr/bin/env python3
"""Build a machine-readable catalogue from historical Trainer artefacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def run_directory(state_path: Path) -> Path:
    parent = state_path.parent
    return parent.parent if CHECKPOINT_RE.fullmatch(parent.name) else parent


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def latest_metrics(history: list[dict]) -> dict:
    result: dict = {}
    for item in history:
        for key, value in item.items():
            if key not in {"epoch", "step"}:
                result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selected: dict[str, tuple[int, Path, dict]] = {}
    for root in args.roots:
        for state_path in root.rglob("trainer_state.json"):
            state = read_json(state_path)
            step = int(state.get("global_step", 0) or 0)
            run_dir = run_directory(state_path)
            key = str(run_dir)
            if key not in selected or step > selected[key][0]:
                selected[key] = (step, state_path, state)

    runs = []
    for run_dir_text, (step, state_path, state) in selected.items():
        run_dir = Path(run_dir_text)
        config = read_json(run_dir / "config.json")
        if not config:
            config = read_json(state_path.parent / "config.json")
        runs.append(
            {
                "run_dir": run_dir_text,
                "state_path": str(state_path),
                "global_step": step,
                "best_metric": state.get("best_metric"),
                "best_model_checkpoint": state.get("best_model_checkpoint"),
                "model": {
                    key: config.get(key)
                    for key in (
                        "d_model",
                        "num_layers",
                        "num_heads",
                        "d_ff",
                        "num_experts",
                        "num_experts_per_token",
                        "pred_head_type",
                        "use_rotary_emb",
                        "aux_loss_weight",
                    )
                    if key in config
                },
                "latest_metrics": latest_metrics(state.get("log_history", [])),
            }
        )

    runs.sort(key=lambda item: (item["global_step"], item["run_dir"]), reverse=True)
    payload = {"schema_version": 1, "run_count": len(runs), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"catalogued {len(runs)} runs -> {args.output}")


if __name__ == "__main__":
    main()
