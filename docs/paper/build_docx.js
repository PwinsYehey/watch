const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Bookmark, BorderStyle, Document, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak, PageNumber,
  Footer, Paragraph, SectionType, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const FONT = "Times New Roman";
const MARGIN = 1020;
const GAP = 340;
const TEXT_WIDTH = 11906 - 2 * MARGIN;
const COLUMN_WIDTH = Math.floor((TEXT_WIDTH - GAP) / 2);
const DXA_PER_CM = 1440 / 2.54;

function inline(text, base = {}) {
  const runs = [];
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|~(.+?)~|\^(.+?)\^)/g;
  let last = 0;
  for (const m of text.matchAll(pattern)) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    if (m[2] !== undefined) runs.push(...inline(m[2], { ...base, bold: true }));
    else if (m[3] !== undefined) runs.push(...inline(m[3], { ...base, italics: true }));
    else if (m[4] !== undefined) runs.push(new TextRun({ text: m[4], ...base, font: "Courier New", size: base.size ? base.size - 2 : 20 }));
    else if (m[5] !== undefined) runs.push(new TextRun({ text: m[5], ...base, subScript: true }));
    else if (m[6] !== undefined) runs.push(new TextRun({ text: m[6], ...base, superScript: true }));
    last = m.index + m[0].length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

function figure(file, caption, widthCm, maxWidth) {
  const full = path.join(ROOT, file);
  const [w, h] = pngSize(full);
  const cm = Math.min(widthCm, maxWidth / DXA_PER_CM);
  const px = (cm / 2.54) * 96;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: Math.round(px), height: Math.round((px * h) / w) } })],
    }),
    new Paragraph({ style: "Caption", children: inline(caption) }),
  ];
}

function table(rows, TEXT_WIDTH) {
  const n = rows[0].length;
  const weights = [...Array(n).keys()].map((c) =>
    Math.max(...rows.map((r) => Math.max(r[c].length ** 0.6, Math.max(...r[c].split(/\s+/).map((w) => w.length)) * 0.7))));
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((TEXT_WIDTH * w) / total));
  widths[n - 1] += TEXT_WIDTH - widths.reduce((a, b) => a + b, 0);
  const line = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({
    width: { size: TEXT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, i) => new TableRow({
      tableHeader: i === 0,
      cantSplit: true,
      children: row.map((cell, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined,
        borders: {
          top: i === 0 ? line : none,
          bottom: i === 0 ? thin : i === rows.length - 1 ? line : none,
          left: none, right: none,
        },
        children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: inline(cell, { size: 17, bold: i === 0 || undefined }) })],
      })),
    })),
  });
}

function parse(source) {
  const sections = [{ cols: 1, children: [] }];
  const headings = [];
  let paragraph = [], bullets = [], rows = [], captions = [];
  const current = () => sections[sections.length - 1];
  const use = (cols) => {
    if (current().cols !== cols) sections.push({ cols, children: [] });
    return current().children;
  };
  const flush = () => {
    if (paragraph.length) {
      const text = paragraph.join(" ");
      use(2).push(new Paragraph({ style: /^\[\d+\] /.test(text) ? "Reference" : "Body", children: inline(text) }));
      paragraph = [];
    }
    for (const b of bullets) use(2).push(new Paragraph({ style: "Body", numbering: { reference: "bullets", level: 0 }, children: inline(b) }));
    bullets = [];
    if (rows.length) {
      const [wide, caption] = captions.pop() || [false, null];
      const target = use(wide ? 1 : 2);
      if (caption) target.push(new Paragraph({ style: "Caption", keepNext: true, spacing: { before: 160, after: 80 }, children: inline(caption) }));
      target.push(table(rows, wide ? TEXT_WIDTH : COLUMN_WIDTH));
      target.push(new Paragraph({ style: "Body", spacing: { after: 60 }, children: [] }));
      rows = [];
    }
  };
  let titles = 0;
  for (const line of source.split(/\r?\n/)) {
    const s = line.trim();
    let m;
    if (s.startsWith("%% ")) {
      use(1).push(new Paragraph({ style: "Authors", children: inline(s.slice(3)) }));
    } else if (s.startsWith("% ")) {
      use(1).push(titles++ === 0 ? new Paragraph({ heading: HeadingLevel.TITLE, children: inline(s.slice(2)) }) : new Paragraph({ style: "Subtitle", children: inline(s.slice(2)) }));
    } else if (s.startsWith("> ")) {
      flush();
      use(1).push(new Paragraph({ style: "Abstract", children: inline(s.slice(2)) }));
    } else if (!s || s === "!<") {
      flush();
    } else if (s === "\\newpage") {
      flush();
      current().children.push(new Paragraph({ children: [new PageBreak()] }));
    } else if ((m = s.match(/^(#{1,2}) (.*)/))) {
      flush();
      const id = `h${headings.length + 1}`;
      headings.push({ level: m[1].length, text: m[2], id });
      use(2).push(new Paragraph({
        heading: m[1].length === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
        children: [new Bookmark({ id, children: inline(m[2]) })],
      }));
    } else if ((m = s.match(/^!>?\[(.*)\]\((.+?)\)(?:\{(\d+(?:\.\d+)?)\})?/))) {
      flush();
      const cm = parseFloat(m[3] || "8");
      if (cm >= 12) use(1).push(...figure(m[2], m[1], cm + 1, TEXT_WIDTH));
      else use(2).push(...figure(m[2], m[1], cm, COLUMN_WIDTH));
    } else if (s.startsWith("|")) {
      if (paragraph.length || bullets.length) flush();
      const cells = s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
    } else if ((m = s.match(/^Table(\*?):\s*(.*)/))) {
      captions.push([m[1] === "*", m[2]]);
    } else if (s.startsWith("- ")) {
      if (paragraph.length) flush();
      bullets.push(s.slice(2));
    } else if (bullets.length && line.startsWith("  ")) {
      bullets[bullets.length - 1] += " " + s;
    } else {
      paragraph.push(s);
    }
  }
  flush();
  return { sections, headings };
}

const built = parse(fs.readFileSync(path.join(HERE, "paper.md"), "utf8"));
const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] });
const heading = (size, before, after) => ({ run: { font: FONT, size, bold: true, color: "000000" }, paragraph: { spacing: { before, after }, keepNext: true, keepLines: true } });

const doc = new Document({
  creator: "Prince Herrera, Ormin Cariaso, John Casili, Christenzien Lalimo",
  title: "WATCH: Wakefulness Assessment through Tracking of Cues & Head",
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 } },
      title: { run: { font: FONT, size: 36, bold: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 }, keepNext: true } },
      heading1: heading(24, 240, 100),
      heading2: heading(21, 160, 60),
    },
    paragraphStyles: [
      { id: "Subtitle", name: "Subtitle", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24, italics: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 } } },
      { id: "Authors", name: "Authors", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 320 } } },
      { id: "Abstract", name: "Abstract", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 21 }, paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { left: 567, right: 567 }, spacing: { before: 0, after: 160, line: 264 } } },
      { id: "Body", name: "Body Text", basedOn: "Normal", next: "Body", quickFormat: true, run: { font: FONT, size: 20 }, paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 100, line: 252 } } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Body", run: { font: FONT, size: 17, italics: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 } } },
      { id: "Reference", name: "Reference", basedOn: "Normal", next: "Reference", run: { font: FONT, size: 17 }, paragraph: { indent: { left: 397, hanging: 397 }, spacing: { before: 0, after: 80 } } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 284 } } } }] }] },
  sections: built.sections.map((section, i) => ({
    properties: {
      type: i === 0 ? undefined : SectionType.CONTINUOUS,
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1020, bottom: 1080, left: MARGIN, right: MARGIN } },
      column: { count: section.cols, space: GAP, equalWidth: true },
    },
    footers: { default: footer() },
    children: section.children,
  })),
});

const output = process.argv[2] || path.join(HERE, "watch-paper.docx");
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(output, buffer);
  console.log(`Saved ${output}`);
});
