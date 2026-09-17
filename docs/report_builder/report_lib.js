/**
 * Shared document builder for the four capstone reports.
 *
 * Content is expressed as a flat list of blocks:
 *   { h1 | h2 | h3: "text" }
 *   { p: "text" }                       plain paragraph (supports **bold** spans)
 *   { bullets: ["a", "b"] }
 *   { numbered: ["a", "b"] }
 *   { table: { header: [...], rows: [[...]], widths: [dxa...], align: ["left","right"] } }
 *   { image: "path.png", width: 600, caption: "Figure 1 — ..." }
 *   { callout: "text" }                 shaded box for a key finding
 *   { pagebreak: true }
 */
const fs = require("fs");
const path = require("path");
const sizeOf = (buf) => {
  // Minimal PNG dimension reader (IHDR chunk).
  if (buf.readUInt32BE(0) !== 0x89504e47) return null;
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
};
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, PageBreak, LevelFormat, Footer, Header, PageNumber,
  TabStopType, TabStopPosition,
} = require("docx");

const ACCENT = "1F3A5F";
const MUTED = "52514E";
const RULE = "D9D9D6";
const SHADE = "EEF3F9";
const FONT = "Calibri";
const PAGE_WIDTH = 11906;   // A4
const PAGE_HEIGHT = 16838;
const MARGIN = 907;         // 1.6 cm
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN;   // 9638 DXA

function runs(text, base = {}) {
  // Split on **bold** markers.
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return new TextRun({ text: part.slice(2, -2), bold: true, ...base });
    }
    return new TextRun({ text: part, ...base });
  });
}

function paragraph(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts.run || {}),
    spacing: { after: 90, line: 250, ...(opts.spacing || {}) },
    alignment: opts.alignment,
    ...(opts.para || {}),
  });
}

function heading(text, level) {
  const map = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 };
  return new Paragraph({
    text,
    heading: map[level],
    spacing: { before: level === 1 ? 200 : 160, after: 70 },
  });
}

function bulletList(items, reference) {
  return items.map((item) => new Paragraph({
    children: runs(item),
    numbering: { reference, level: 0 },
    spacing: { after: 40, line: 252 },
  }));
}

function cell(text, { header = false, align = "left", width, shade } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
    margins: { top: 45, bottom: 45, left: 90, right: 90 },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    children: [new Paragraph({
      children: runs(text, { bold: header, size: 18, color: header ? ACCENT : undefined }),
      alignment: align === "right" ? AlignmentType.RIGHT : AlignmentType.LEFT,
      spacing: { after: 0 },
    })],
  });
}

function table({ header, rows, widths, align }) {
  const n = header.length;
  const w = widths || Array(n).fill(Math.floor(CONTENT_WIDTH / n));
  const al = align || Array(n).fill("left");
  return new Table({
    width: { size: w.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((h, i) => cell(h, { header: true, align: al[i], width: w[i], shade: SHADE })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, { align: al[i], width: w[i] })),
      })),
    ],
  });
}

function image(file, width, caption) {
  const buf = fs.readFileSync(file);
  const dims = sizeOf(buf) || { width: 1600, height: 900 };
  const height = Math.round(width * dims.height / dims.width);
  const out = [new Paragraph({
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
  })];
  if (caption) {
    out.push(new Paragraph({
      children: runs(caption, { italics: true, size: 18, color: MUTED }),
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }));
  }
  return out;
}

function callout(text) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: SHADE, color: "auto" },
        margins: { top: 120, bottom: 120, left: 180, right: 180 },
        borders: {
          top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
          bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
          right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
          left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT },
        },
        children: [new Paragraph({ children: runs(text, { size: 20 }), spacing: { after: 0, line: 259 } })],
      })],
    })],
  });
}

function render(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.h1) out.push(heading(b.h1, 1));
    else if (b.h2) out.push(heading(b.h2, 2));
    else if (b.h3) out.push(heading(b.h3, 3));
    else if (b.p !== undefined) out.push(paragraph(b.p, b.opts));
    else if (b.bullets) out.push(...bulletList(b.bullets, "bullets"));
    else if (b.numbered) out.push(...bulletList(b.numbered, "numbers"));
    else if (b.table) { out.push(table(b.table)); out.push(paragraph("", { spacing: { after: 20, line: 200 } })); }
    else if (b.image) out.push(...image(b.image, b.width || 560, b.caption));
    else if (b.callout) { out.push(callout(b.callout)); out.push(paragraph("", { spacing: { after: 20, line: 200 } })); }
    else if (b.pagebreak) out.push(new Paragraph({ children: [new PageBreak()] }));
    else if (b.spacer) out.push(paragraph("", { spacing: { after: b.spacer } }));
  }
  return out;
}

function titleBlock({ title, subtitle, meta }) {
  const out = [
    new Paragraph({
      children: [new TextRun({ text: title, bold: true, size: 40, color: ACCENT, font: FONT })],
      spacing: { after: 80 },
    }),
  ];
  if (subtitle) out.push(new Paragraph({
    children: [new TextRun({ text: subtitle, size: 24, color: MUTED, font: FONT })],
    spacing: { after: 100 },
  }));
  if (meta) {
    for (const [k, v] of meta) {
      out.push(new Paragraph({
        children: [new TextRun({ text: `${k}  `, bold: true, size: 19, color: MUTED }),
                   new TextRun({ text: v, size: 19, color: MUTED })],
        spacing: { after: 20 },
      }));
    }
  }
  out.push(new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 4 } },
    spacing: { after: 120 },
  }));
  return out;
}

async function buildDocument({ title, subtitle, meta, blocks, outFile, footerText }) {
  const doc = new Document({
    creator: "Rapipong Sornsakda",
    title,
    styles: {
      default: { document: { run: { font: FONT, size: 20 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, color: ACCENT, font: FONT },
          paragraph: { spacing: { before: 200, after: 70 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: ACCENT, font: FONT },
          paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, color: MUTED, font: FONT },
          paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
        { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 320 } } } }] },
      ],
    },
    sections: [{
      properties: {
        page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
                margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          children: [new TextRun({ text: "Smart City Traffic Intelligence — NUS/Emeritus AMLDS Capstone", size: 16, color: MUTED })],
          alignment: AlignmentType.RIGHT })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          children: [
            new TextRun({ text: footerText || "Rapipong Sornsakda", size: 16, color: MUTED }),
            new TextRun({ text: "\tPage ", size: 16, color: MUTED }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED }),
            new TextRun({ text: " of ", size: 16, color: MUTED }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: MUTED }),
          ] })] }),
      },
      children: [...titleBlock({ title, subtitle, meta }), ...render(blocks)],
    }],
  });
  const buffer = await Packer.toBuffer(doc);
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, buffer);
  return outFile;
}

module.exports = { buildDocument };
