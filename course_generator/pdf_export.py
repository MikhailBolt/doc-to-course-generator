from pathlib import Path
from typing import Optional


def _find_unicode_font() -> Optional[str]:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def export_markdown_to_pdf(markdown_text: str, output_path: Path) -> str:
    """Convert a simple Markdown summary into a PDF file (Unicode when a system font is found)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise ImportError("reportlab is required for PDF export. Run: pip install reportlab") from exc

    font_path = _find_unicode_font()
    font_name = "Helvetica"
    if font_path:
        font_name = "CourseUnicode"
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName=font_name, fontSize=10, leading=14)
    h1 = ParagraphStyle("H1", parent=body, fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=body, fontSize=13, leading=17, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=body, fontSize=11, leading=15, spaceAfter=4)

    story = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("### "):
            story.append(Paragraph(text[4:].strip(), h3))
        elif line.startswith("## "):
            story.append(Paragraph(text[3:].strip(), h2))
        elif line.startswith("# "):
            story.append(Paragraph(text[2:].strip(), h1))
        elif line.startswith("- "):
            story.append(Paragraph(f"• {text[2:].strip()}", body))
        else:
            story.append(Paragraph(text, body))

    doc.build(story)
    return str(output_path)
