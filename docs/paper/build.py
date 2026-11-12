import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BalancedColumns, Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT_DIR = Path("/mnt/c/Windows/Fonts")

MARGIN = 1.8 * cm
GAP = 0.6 * cm
PAGE_WIDTH = A4[0] - 2 * MARGIN - 12
COLUMN_WIDTH = (PAGE_WIDTH - GAP) / 2

if (FONT_DIR / "times.ttf").exists():
    for name, file in {"Serif": "times", "Serif-Bold": "timesbd", "Serif-Italic": "timesi", "Serif-BoldItalic": "timesbi", "Mono": "cour"}.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f"{file}.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")
    SERIF, SERIF_BOLD, SERIF_ITALIC, MONO = "Serif", "Serif-Bold", "Serif-Italic", "Mono"
else:
    SERIF, SERIF_BOLD, SERIF_ITALIC, MONO = "Times-Roman", "Times-Bold", "Times-Italic", "Courier"

BODY = ParagraphStyle("body", fontName=SERIF, fontSize=10, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=5)
TITLE = ParagraphStyle("title", fontName=SERIF_BOLD, fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=5)
SUBTITLE = ParagraphStyle("subtitle", fontName=SERIF_ITALIC, fontSize=11.5, leading=14, alignment=TA_CENTER, spaceAfter=9)
H1 = ParagraphStyle("h1", fontName=SERIF_BOLD, fontSize=12, leading=15, spaceBefore=9, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName=SERIF_BOLD, fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.5, leading=10.5, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9)
REFERENCE = ParagraphStyle("reference", parent=BODY, fontSize=8.5, leading=10.5, alignment=0, leftIndent=16, firstLineIndent=-16, spaceAfter=3)
TABLE_CAPTION = ParagraphStyle("tablecaption", parent=CAPTION, spaceBefore=3, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName=SERIF, fontSize=8.5, leading=10.5)
CELL_HEAD = ParagraphStyle("cellhead", parent=CELL, fontName=SERIF_BOLD)
AUTHORS = ParagraphStyle("authors", fontName=SERIF, fontSize=11.5, leading=14, alignment=TA_CENTER, spaceAfter=14)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=9.5, leading=11.8, leftIndent=0.8 * cm, rightIndent=0.8 * cm, spaceAfter=5)
WIDE_FROM_CM = 12


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="%s" size="8.5">\1</font>' % MONO, text)
    text = re.sub(r"==(.+?)==", r'<font backColor="#ffe066">\1</font>', text)
    text = re.sub(r"\^(.+?)\^", r"<super>\1</super>", text)
    text = re.sub(r"~(.+?)~", r"<sub>\1</sub>", text)
    return text


def figure(path: str, caption: str, width: float) -> KeepTogether:
    image = Image(str(ROOT / path))
    scale = width / image.imageWidth
    image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
    return KeepTogether([Spacer(1, 3), image, Paragraph(inline(caption), CAPTION)])


class Unsplittable(Table):
    def split(self, availWidth, availHeight):
        return []


def boxed(items: list, width: float) -> Table:
    box = Unsplittable([[item] for item in items], colWidths=[width])
    box.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return box


def column_figure(path: str, caption: str, width: float) -> Table:
    image = Image(str(ROOT / path))
    scale = width / image.imageWidth
    image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
    return boxed([Spacer(1, 3), image, Paragraph(inline(caption), CAPTION)], COLUMN_WIDTH)


def table(rows: list[list[str]], width: float) -> Table:
    data = [[Paragraph(inline(c), CELL_HEAD if i == 0 else CELL) for c in row] for i, row in enumerate(rows)]
    weights = [max(max(len(row[c]) ** 0.6, max(len(w) for w in row[c].split() or [""]) * 0.7) for row in rows) for c in range(len(rows[0]))]
    result = Table(data, colWidths=[width * w / sum(weights) for w in weights], hAlign="CENTER", repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return result


def parse(source: str) -> list:
    blocks, column, bullets, rows, paragraph, captions, heads = [], [], [], [], [], [], []

    def add(flowable):
        if heads and isinstance(flowable, Paragraph):
            flowable.wrap(COLUMN_WIDTH, 1e6)
            style = flowable.style
            parts = flowable.split(COLUMN_WIDTH, 2 * style.leading + 1) or [flowable]
            column.append(boxed(heads + [parts[0]], COLUMN_WIDTH))
            column.extend(parts[1:])
        elif heads and isinstance(flowable, Unsplittable):
            column.append(boxed(heads + [flowable], COLUMN_WIDTH))
        else:
            column.extend(heads)
            column.append(flowable)
        heads.clear()

    def close_column():
        if heads:
            column.extend(heads)
            heads.clear()
        if column:
            blocks.append(BalancedColumns(list(column), nCols=2, endSlack=0, innerPadding=GAP, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0))
            column.clear()

    def wide(flowable):
        close_column()
        if blocks and isinstance(blocks[-1], BalancedColumns):
            blocks.append(Spacer(1, 14))
        blocks.append(flowable)

    def flush():
        if paragraph:
            text = " ".join(paragraph)
            add(Paragraph(inline(text), REFERENCE if re.match(r"\[\d+\] ", text) else BODY))
            paragraph.clear()
        if bullets:
            items = [ListItem(Paragraph(inline(b), BODY), leftIndent=11) for b in bullets]
            add(ListFlowable(items, bulletType="bullet", start="•", leftIndent=11, bulletFontSize=8))
            bullets.clear()
        if rows:
            spans, caption = captions.pop() if captions else (False, None)
            width = PAGE_WIDTH if spans else COLUMN_WIDTH
            parts = [Paragraph(inline(caption), TABLE_CAPTION)] if caption else []
            if spans:
                wide(KeepTogether(parts + [table(rows, width), Spacer(1, 9)]))
            else:
                add(boxed(parts + [table(rows, width), Spacer(1, 9)], COLUMN_WIDTH))
            rows.clear()

    titles, authors, abstract = [], [], []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("%% "):
            authors.append(stripped[3:])
        elif stripped.startswith("% "):
            titles.append(stripped[2:])
        elif stripped.startswith("> "):
            abstract.append(stripped[2:])
        elif not stripped or stripped == "!<":
            flush()
        elif stripped == "\\newpage":
            flush()
            close_column()
            blocks.append(PageBreak())
        elif stripped.startswith("## "):
            flush()
            heads.append(Paragraph(inline(stripped[3:]), H2))
        elif stripped.startswith("# "):
            flush()
            heads.append(Paragraph(inline(stripped[2:]), H1))
        elif match := re.match(r"!>?\[(.*)\]\((.+?)\)(?:\{(\d+(?:\.\d+)?)\})?", stripped):
            flush()
            width_cm = float(match[3] or 8)
            if width_cm >= WIDE_FROM_CM:
                wide(figure(match[2], match[1], min(width_cm * cm, PAGE_WIDTH)))
            else:
                add(column_figure(match[2], match[1], min(width_cm * cm, COLUMN_WIDTH)))
        elif stripped.startswith("|"):
            if paragraph or bullets:
                flush()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        elif match := re.match(r"Table(\*?):\s*(.*)", stripped):
            captions.append((match[1] == "*", match[2]))
        elif stripped.startswith("- "):
            if paragraph:
                flush()
            bullets.append(stripped[2:])
        elif bullets and line.startswith("  "):
            bullets[-1] += " " + stripped
        else:
            paragraph.append(stripped)
    flush()
    close_column()

    head = [Paragraph(inline(titles[0]), TITLE)] if titles else []
    head += [Paragraph(inline(t), SUBTITLE) for t in titles[1:]]
    head += [Paragraph(inline(a), AUTHORS) for a in authors]
    head += [Paragraph(inline(a), ABSTRACT) for a in abstract]
    if abstract:
        head.append(Spacer(1, 10))
    return head + blocks


def page_number(canvas, doc):
    canvas.setFont(SERIF, 9)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))


def build(source: Path, output: Path) -> None:
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
        title="WATCH: Wakefulness Assessment through Tracking of Cues & Head",
    )
    doc.build(parse(source.read_text(encoding="utf-8")), onFirstPage=page_number, onLaterPages=page_number)
    print(f"Saved {output}")


if __name__ == "__main__":
    build(HERE / "paper.md", Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "watch-paper.pdf")
