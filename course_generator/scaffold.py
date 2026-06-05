from pathlib import Path

from course_generator.constants import (
    DEFAULT_DOCS_PATH,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
)


def init_project_directories() -> list[str]:
    """Create standard folders and a sample doc if missing."""
    created: list[str] = []
    dirs = [
        DEFAULT_DOCS_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_LOG_DIR,
        "vectorstore",
    ]
    for name in dirs:
        path = Path(name)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))

    sample = Path(DEFAULT_DOCS_PATH) / "sample-topic.txt"
    if not sample.exists():
        sample.write_text(
            "Sample topic for Doc-to-Course Generator\n\n"
            "Replace this file with your PDF, TXT, MD, or DOCX materials.\n",
            encoding="utf-8",
        )
        created.append(str(sample))

    return created
