import zipfile
from pathlib import Path
from typing import Dict

from course_generator.io import build_output_path


def create_delivery_zip(paths: Dict[str, str], output_dir: str, output_prefix: str = "") -> str:
    """Pack generated artifacts into a single ZIP for download."""
    zip_path = build_output_path(output_dir, "course_delivery.zip", output_prefix)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in paths.values():
            path = Path(file_path)
            if path.is_file() and path.exists():
                archive.write(path, arcname=path.name)
    return str(zip_path)
