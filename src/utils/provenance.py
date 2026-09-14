"""
src/utils/provenance.py
Run provenance manifest for reproducible experiments.

Every training / fine-tuning / evaluation run writes a ``run_info.json`` into
its output directory capturing everything needed to reproduce it:

    * git commit SHA + dirty-tree flag (from the repository root)
    * the fully-resolved Hydra config
    * a ``pip freeze`` snapshot
    * a filtered environment snapshot (no secret material)

The environment filter only whitelists a small set of safe variables
(``DEEPWIND_*``, Slurm job metadata, ROCm/Torch runtime knobs) and explicitly
excludes anything whose name hints at credentials (``TOKEN``, ``KEY``,
``SECRET``, ``PASSWORD``).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Environment variables safe to capture verbatim. Anything matching a secret
# hint is dropped before prefix matching, so credentials never leak.
_SAFE_ENV_PREFIXES = (
    "DEEPWIND_",
    "SLURM_",
    "ROCM_",
    "ROCR_",
    "MIOPEN_",
    "HSA_",
    "NCCL_",
    "MASTER_",
    "OMP_",
    "TORCH_",
)
_SAFE_ENV_EXACT = {
    "CUDA_VISIBLE_DEVICES",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LANG",
    "LC_ALL",
}
_SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE")


def _repo_root(repo_root: Optional[str]) -> Optional[str]:
    """Resolve the repository root: explicit arg → env var → cwd."""
    return repo_root or os.environ.get("DEEPWIND_PROJECT_ROOT")


def _git(repo_root: str, *args: str) -> Optional[str]:
    """Run ``git -C <root> <args>`` and return stripped stdout, or ``None``."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception as exc:  # noqa: BLE001 — git may be absent; never fatal
        logger.warning("git unavailable (%s)", exc)
        return None


def get_git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    """Return the current commit SHA, or ``None`` if not a git repo."""
    root = _repo_root(repo_root)
    if not root:
        return None
    return _git(root, "rev-parse", "HEAD")


def get_git_dirty(repo_root: Optional[str] = None) -> Optional[bool]:
    """Return ``True`` if the working tree has uncommitted changes.

    ``None`` means git could not be queried (not a repo / git missing).
    """
    root = _repo_root(repo_root)
    if not root:
        return None
    out = _git(root, "status", "--porcelain")
    if out is None:
        return None
    return out != ""


def get_pip_freeze() -> list[str]:
    """Return ``pip freeze`` output as a list of lines (empty on failure)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if out.returncode == 0:
            return out.stdout.splitlines()
    except Exception as exc:  # noqa: BLE001 — pip may be unavailable
        logger.warning("pip freeze failed (%s)", exc)
    return []


def get_env_snapshot() -> dict[str, str]:
    """Return a filtered environment snapshot with secret material removed."""
    snapshot: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        up = key.upper()
        if any(hint in up for hint in _SECRET_HINTS):
            continue
        if up in _SAFE_ENV_EXACT or any(
            up.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES
        ):
            snapshot[key] = value
    return snapshot


def make_run_name(
    size: str,
    variant: str,
    seed: int,
    when: Optional[datetime] = None,
) -> str:
    """Standard run-name convention: ``deepwind-{size}-{variant}-seed{seed}-{date}``."""
    when = when or datetime.now(timezone.utc)
    return f"deepwind-{size}-{variant}-seed{seed}-{when.strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_run_info(
    output_dir: str | Path,
    cfg: Optional[Any] = None,
    repo_root: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Write ``run_info.json`` into ``output_dir`` and return its path.

    Args:
        output_dir: Directory that will receive the manifest (created if
            missing).
        cfg:        Optional Hydra ``DictConfig``; serialised fully-resolved.
        repo_root:  Optional repository root override; defaults to
            ``DEEPWIND_PROJECT_ROOT``.
        extra:      Optional extra dict merged into the manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg: Any = None
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception as exc:  # noqa: BLE001 — config must never block the run
            logger.warning("Could not resolve config for provenance (%s)", exc)

    import getpass

    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "unknown"

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": user,
        "python": sys.version.split()[0],
        "git": {
            "repo_root": _repo_root(repo_root),
            "commit": get_git_commit(repo_root),
            "dirty": get_git_dirty(repo_root),
        },
        "config": resolved_cfg,
        "pip_freeze": get_pip_freeze(),
        "env": get_env_snapshot(),
    }
    if extra:
        manifest["extra"] = extra

    path = output_dir / "run_info.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Run provenance manifest → %s", path)
    return path
