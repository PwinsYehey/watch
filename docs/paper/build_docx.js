const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak, PageNumber,
  Footer, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const FONT = "Times New Roman";
const TEXT_WIDTH = 9026;
const MAX_IMAGE_CM = 15.9;

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

function figure(file, caption, widthCm) {
  const full = path.join(ROOT, file);
  const [w, h] = pngSize(full);
  const cm = Math.min(widthCm, MAX_IMAGE_CM);
  const px = (cm / 2.54) * 96;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: Math.round(px), height: Math.round((px * h) / w) } })],
    }),
    new Paragraph({ style: "Caption", children: inline(caption) }),
  ];
}

function table(rows) {
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
        children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: inline(cell, { size: 19, bold: i === 0 || undefined }) })],
      })),
    })),
  });
}

function parse(source) {
  const out = [];
  let paragraph = [], bullets = [], rows = [], captions = [];
  const flush = () => {
    if (paragraph.length) {
      const text = paragraph.join(" ");
      out.push(/^\[\d+\] /.test(text) ? new Paragraph({ style: "Reference", children: inline(text) }) : new Paragraph({ style: "Body", children: inline(text) }));
      paragraph = [];
    }
    for (const b of bullets) out.push(new Paragraph({ style: "Body", numbering: { reference: "bullets", level: 0 }, children: inline(b) }));
    bullets = [];
    if (rows.length) {
      const caption = captions.pop();
      if (caption) out.push(new Paragraph({ style: "Caption", keepNext: true, spacing: { before: 160, after: 80 }, children: inline(caption) }));
      out.push(table(rows));
      out.push(new Paragraph({ style: "Body", spacing: { after: 60 }, children: [] }));
      rows = [];
    }
  };
  let titles = 0;
  for (const line of source.split(/\r?\n/)) {
    const s = line.trim();
    let m;
    if (s.startsWith("%% ")) {
      out.push(new Paragraph({ style: "Authors", children: inline(s.slice(3)) }));
    } else if (s.startsWith("% ")) {
      out.push(titles++ === 0 ? new Paragraph({ heading: HeadingLevel.TITLE, children: inline(s.slice(2)) }) : new Paragraph({ style: "Subtitle", children: inline(s.slice(2)) }));
    } else if (s.startsWith("> ")) {
      flush();
      out.push(new Paragraph({ style: "Abstract", children: inline(s.slice(2)) }));
    } else if (!s || s === "!<") {
      flush();
    } else if (s === "\\newpage") {
      flush();
      out.push(new Paragraph({ children: [new PageBreak()] }));
    } else if (s.startsWith("## ")) {
      flush();
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inline(s.slice(3)) }));
    } else if (s.startsWith("# ")) {
      flush();
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: inline(s.slice(2)) }));
    } else if ((m = s.match(/^!>?\[(.*)\]\((.+?)\)(?:\{(\d+(?:\.\d+)?)\})?/))) {
      flush();
      const cm = parseFloat(m[3] || "8");
      out.push(...figure(m[2], m[1], cm >= 12 ? MAX_IMAGE_CM : Math.max(cm * 1.4, 9)));
    } else if (s.startsWith("|")) {
      if (paragraph.length || bullets.length) flush();
      const cells = s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
    } else if ((m = s.match(/^Table\*?:\s*(.*)/))) {
      captions.push(m[1]);
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
  return out;
}

const heading = (size, before, after) => ({ run: { font: FONT, size, bold: true, color: "000000" }, paragraph: { spacing: { before, after }, keepNext: true, keepLines: true } });

const doc = new Document({
  creator: "Prince Herrera, Ormin Cariaso, John Casili, Christenzien Lalimo",
  title: "WATCH: Wakefulness Assessment through Tracking of Cues & Head",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      title: { run: { font: FONT, size: 36, bold: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 }, keepNext: true } },
      heading1: heading(28, 360, 120),
      heading2: heading(24, 240, 80),
    },
    paragraphStyles: [
      { id: "Subtitle", name: "Subtitle", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24, italics: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 } } },
      { id: "Authors", name: "Authors", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 320 } } },
      { id: "Abstract", name: "Abstract", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 21 }, paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { left: 567, right: 567 }, spacing: { before: 0, after: 160, line: 264 } } },
      { id: "Body", name: "Body Text", basedOn: "Normal", next: "Body", quickFormat: true, run: { font: FONT, size: 22 }, paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 120, line: 276 } } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Body", run: { font: FONT, size: 19, italics: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 } } },
      { id: "Reference", name: "Reference", basedOn: "Normal", next: "Reference", run: { font: FONT, size: 19 }, paragraph: { indent: { left: 454, hanging: 454 }, spacing: { before: 0, after: 80 } } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 284 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: parse(fs.readFileSync(path.join(HERE, "paper.md"), "utf8")),
  }],
});

const output = process.argv[2] || path.join(HERE, "watch-paper.docx");
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(output, buffer);
  console.log(`Saved ${output}`);
});
