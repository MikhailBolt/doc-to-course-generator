import copy
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from course_generator.config_loader import apply_config_file
from course_generator.pipeline import run_pipeline
from course_generator.presets import apply_cli_preset

ProgressCallback = Callable[[str, str], None]


def list_batch_configs(batch_dir: str) -> List[Path]:
    root = Path(batch_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    configs = sorted(root.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No *.json config files in '{batch_dir}'")
    return configs


def run_batch(
    batch_dir: str,
    base_args: Namespace,
    defaults: Namespace,
    progress: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Run the pipeline once per JSON config in batch_dir."""
    results: List[Dict[str, Any]] = []
    configs = list_batch_configs(batch_dir)

    for idx, cfg_path in enumerate(configs, start=1):
        if progress:
            progress("batch", f"{idx}/{len(configs)}: {cfg_path.name}")

        args = copy.deepcopy(base_args)
        apply_config_file(args, str(cfg_path), defaults)
        try:
            apply_cli_preset(args)
        except ValueError as exc:
            results.append({"config": str(cfg_path), "ok": False, "error": str(exc)})
            continue

        try:
            result = run_pipeline(args, progress=progress)
            results.append({"config": str(cfg_path), "ok": True, "result": result})
        except Exception as exc:
            results.append({"config": str(cfg_path), "ok": False, "error": str(exc)})

    return results
