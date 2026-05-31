import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict

from course_generator import __version__
from course_generator.io import build_output_path


def _write_delivery_manifest(paths: Dict[str, str], output_dir: str, output_prefix: str) -> Path:
    manifest_path = build_output_path(output_dir, "delivery_manifest.txt", output_prefix)
    lines = [
        "Doc-to-Course Generator — delivery package",
        f"Generator version: {__version__}",
        f"Packed at: {datetime.now().isoformat()}",
        "",
        "Files:",
    ]
    for key, file_path in sorted(paths.items()):
        path = Path(file_path)
        if path.is_file() and path.exists():
            lines.append(f"  [{key}] {path.name} ({path.stat().st_size} bytes)")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def create_delivery_zip(paths: Dict[str, str], output_dir: str, output_prefix: str = "") -> str:
    """Pack generated artifacts into a single ZIP for download."""
    zip_path = build_output_path(output_dir, "course_delivery.zip", output_prefix)
    manifest_path = _write_delivery_manifest(paths, output_dir, output_prefix)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname=manifest_path.name)
        for file_path in paths.values():
            path = Path(file_path)
            if path.is_file() and path.exists():
                archive.write(path, arcname=path.name)
    return str(zip_path)
