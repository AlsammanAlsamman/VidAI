#!/usr/bin/env node
/**
 * VidAI flowchart.
 *
 * Pure Node, no dependencies: builds the diagram as SVG, writes vidai_flow.svg
 * and vidai_flow.html, then rasterises to vidai_flow.png with headless Chrome
 * (falls back to ImageMagick `convert` if Chrome is not found).
 *
 *   node assets/flowchart/flowchart.js      # -> assets/flowchart/vidai_flow.{svg,html,png}
 */

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// ---------------------------------------------------------------- palette
const C = {
  bg0: "#0e0e15", bg1: "#1a1830",
  you: "#34d399", claude: "#ffbf00", vidai: "#8b6cff", live: "#ff3b5c",
  text: "#f4f4ff", muted: "#a9abc9", line: "#8a8dbf",
  card: "#1f1d36", cardEdge: "#3a3666",
};

// ---------------------------------------------------------------- data
const steps = [
  { n: 1, title: "BRIEF", who: "you + Claude", kind: "claude", file: "brief",
    lines: ["say “vidai” in Claude Code", "Claude asks: topic, language,", "style, length, outline"] },
  { n: 2, title: "ANCHOR SETUP", who: "Claude", kind: "claude", file: "session.json",
    lines: ["Claude picks the few small", "stats worth recording", "for this video, and why"] },
  { n: 3, title: "RECORD", who: "VidAI Recorder", kind: "vidai", file: "video.mkv",
    lines: ["camera · screen · camera-in-screen", "mic · hotkeys · voice", "“VidAI record” / “VidAI stop”"] },
  { n: 4, title: "LIVE", who: "VidAI · every frame", kind: "live", file: "live.jsonl",
    lines: ["effects by voice · picture tools", "AI models · talk · suggest", "stats: speech · OCR · 30 fps"] },
  { n: 5, title: "ANCHORS", who: "VidAI", kind: "vidai", file: "anchors.json",
    lines: ["pauses · speech + transcript", "markers · screen text", "what was applied live"] },
  { n: 6, title: "UNDERSTAND", who: "Claude", kind: "claude", file: "frames.png",
    lines: ["reads the anchors, looks only", "at the frames they point to", "compares with the brief"] },
  { n: 7, title: "EDIT PLAN", who: "Claude", kind: "claude", file: "edit.json",
    lines: ["cut pauses & mistakes · text", "arrows · zoom · audio fixes", "subtitles · chapters · models"] },
  { n: 8, title: "RENDER", who: "VidAI · parallel", kind: "vidai", file: "final.mp4",
    lines: ["split inside silences", "3 chunks at once · C kernels", "YouTube MP4 + .srt + chapters"] },
];
const kindColor = (k) => C[k] || C.line;

// ---------------------------------------------------------------- layout
const W = 1700, H = 940;
const boxW = 270, boxH = 168, gap = 42;
const rowTop = 230, rowBottom = 600;
const leftPad = (W - (5 * boxW + 4 * gap)) / 2;
const pos = {};
for (let i = 0; i < 5; i++) pos[i + 1] = { x: leftPad + i * (boxW + gap), y: rowTop };
pos[6] = { x: pos[5].x, y: rowBottom };
pos[7] = { x: pos[4].x, y: rowBottom };
pos[8] = { x: pos[3].x, y: rowBottom };

// ---------------------------------------------------------------- svg helpers
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, o = {}) =>
  `<text x="${x}" y="${y}" font-size="${o.size || 14}" font-weight="${o.weight || 400}" fill="${o.fill || C.text}" ` +
  `text-anchor="${o.anchor || "start"}" font-family="${o.mono ? "'JetBrains Mono', 'Ubuntu Sans Mono', Menlo, monospace" : "'Ubuntu Sans', Inter, 'Segoe UI', Helvetica, Arial, sans-serif"}" ` +
  `letter-spacing="${o.spacing || 0}" opacity="${o.opacity ?? 1}" ${o.transform ? `transform="${o.transform}"` : ""}>${esc(s)}</text>`;

function card(s) {
  const { x, y } = pos[s.n];
  const col = kindColor(s.kind);
  let out = "";
  out += `<rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="16" fill="${C.card}" stroke="${C.cardEdge}" stroke-width="1.5" filter="url(#shadow)"/>`;
  out += `<rect x="${x}" y="${y}" width="${boxW}" height="6" rx="3" fill="${col}"/>`;
  out += `<circle cx="${x + 30}" cy="${y + 40}" r="17" fill="${col}"/>`;
  out += text(x + 30, y + 46, s.n, { size: 18, weight: 800, fill: C.bg0, anchor: "middle" });
  out += text(x + 58, y + 46, s.title, { size: 19, weight: 800, spacing: 0.5 });
  out += text(x + 58, y + 66, s.who, { size: 12, fill: col, weight: 600 });
  s.lines.forEach((l, i) => { out += text(x + 20, y + 94 + i * 18, l, { size: 12.5, fill: C.muted }); });
  const chipW = s.file.length * 7.4 + 22;
  out += `<rect x="${x + boxW - chipW - 14}" y="${y + boxH - 30}" width="${chipW}" height="20" rx="10" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.6"/>`;
  out += text(x + boxW - chipW / 2 - 14, y + boxH - 16, s.file, { size: 11, mono: true, fill: col, anchor: "middle" });
  return out;
}

function arrow(d, o = {}) {
  const col = o.color || C.line;
  const head = o.color === C.claude ? "claude" : o.color === C.live ? "live" : "line";
  let out = `<path d="${d}" fill="none" stroke="${col}" stroke-width="${o.width || 3}" stroke-linecap="round" ` +
    `stroke-linejoin="round" marker-end="url(#head-${head})" ${o.dash ? `stroke-dasharray="${o.dash}"` : ""}/>`;
  if (o.label) {
    const [lx, ly] = o.labelAt;
    const w = o.label.length * 6.6 + 18;
    out += `<rect x="${lx - w / 2}" y="${ly - 12}" width="${w}" height="20" rx="10" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.7"/>`;
    out += text(lx, ly + 2, o.label, { size: 11, fill: col, anchor: "middle", weight: 600 });
  }
  return out;
}

// ---------------------------------------------------------------- build
let svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="${C.bg0}"/><stop offset="1" stop-color="${C.bg1}"/>
  </linearGradient>
  <filter id="shadow" x="-10%" y="-10%" width="130%" height="140%">
    <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" flood-opacity="0.45"/>
  </filter>
  ${["line", "claude", "live"].map((k) =>
    `<marker id="head-${k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse">
       <path d="M0,0 L10,5 L0,10 z" fill="${k === "line" ? C.line : C[k]}"/></marker>`).join("")}
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0 L0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<rect width="100%" height="100%" fill="url(#grid)"/>
`;

// logo + title
const icon = path.join(__dirname, "..", "icon.png");
if (fs.existsSync(icon)) {
  const b64 = fs.readFileSync(icon).toString("base64");
  svg += `<image x="${W / 2 - 330}" y="22" width="104" height="67" href="data:image/png;base64,${b64}"/>`;
}
svg += text(W / 2 + 60, 68, "VidAI · how it works", { size: 36, weight: 800, anchor: "middle", spacing: 0.5 });
svg += text(W / 2, 124, "say “vidai” → Claude interviews you → you record with live stats & effects → Claude edits → YouTube-ready video",
  { size: 14, fill: C.muted, anchor: "middle" });

// under development stamp
svg += `<g transform="translate(${W - 150},62) rotate(8)">
  <rect x="-118" y="-24" width="236" height="48" rx="12" fill="${C.live}" fill-opacity="0.12" stroke="${C.live}" stroke-width="2.5" stroke-dasharray="7 5"/>
  ${text(0, -2, "UNDER DEVELOPMENT", { size: 16, weight: 800, fill: C.live, anchor: "middle", spacing: 1 })}
  ${text(0, 16, "v0.4 · working prototype", { size: 11.5, fill: C.live, anchor: "middle", weight: 600 })}
</g>`;

// recording bracket (behind cards 3-5)
const rX0 = pos[3].x - 18, rX1 = pos[5].x + boxW + 18;
svg += `<rect x="${rX0}" y="${rowTop - 46}" width="${rX1 - rX0}" height="${boxH + 74}" rx="22" fill="${C.vidai}" fill-opacity="0.06" stroke="${C.vidai}" stroke-opacity="0.45" stroke-dasharray="6 6"/>`;
svg += text(rX0 + 18, rowTop - 22, "WHILE YOU RECORD  ·  all live", { size: 12.5, fill: C.vidai, weight: 700, spacing: 0.4 });

// edit bracket (behind cards 6-8)
const eX0 = pos[8].x - 18, eX1 = pos[6].x + boxW + 18;
svg += `<rect x="${eX0}" y="${rowBottom - 46}" width="${eX1 - eX0}" height="${boxH + 74}" rx="22" fill="${C.claude}" fill-opacity="0.05" stroke="${C.claude}" stroke-opacity="0.4" stroke-dasharray="6 6"/>`;
svg += text(eX0 + 18, rowBottom - 22, "AFTER RECORDING  ·  non-destructive: the original video is never changed", { size: 12.5, fill: C.claude, weight: 700, spacing: 0.4 });

// forward arrows 1→2→3→4→5
for (let i = 1; i < 5; i++) {
  const a = pos[i], b = pos[i + 1], y = a.y + boxH / 2;
  svg += arrow(`M${a.x + boxW + 4},${y} L${b.x - 6},${y}`);
}
// 5 → 6
{
  const x = pos[5].x + boxW / 2;
  svg += arrow(`M${x},${pos[5].y + boxH + 4} L${x},${rowBottom - 52}`, { label: "stop → anchors saved", labelAt: [x, (pos[5].y + boxH + rowBottom - 48) / 2] });
}
// 6 → 7 → 8
for (const [a, b] of [[6, 7], [7, 8]]) {
  const y = rowBottom + boxH / 2;
  svg += arrow(`M${pos[a].x - 4},${y} L${pos[b].x + boxW + 6},${y}`);
}
// Claude's live loop over LIVE (4)
{
  const xR = pos[4].x + boxW / 2 + 80, xL = pos[4].x + boxW / 2 - 80, yTop = rowTop - 80;
  svg += arrow(`M${xR},${rowTop - 2} L${xR},${yTop} L${xL},${yTop} L${xL},${rowTop - 8}`,
    { color: C.claude, dash: "8 6", width: 2.5, label: "“VidAI, <request>” → Claude → command / new effect", labelAt: [(xR + xL) / 2, yTop] });
}
// re-edit loop 8 → 7
{
  const y = rowBottom + boxH + 30, x8 = pos[8].x + boxW / 2, x7 = pos[7].x + boxW / 2;
  svg += `<path d="M${x8},${rowBottom + boxH + 4} C${x8},${y + 20} ${x7},${y + 20} ${x7},${rowBottom + boxH + 8}" fill="none" stroke="${C.claude}" stroke-width="2" stroke-dasharray="4 5" opacity="0.8" marker-end="url(#head-claude)"/>`;
  svg += text((x8 + x7) / 2, y + 30, "look at the result → adjust → render again", { size: 11.5, fill: C.claude, anchor: "middle", weight: 600 });
}

steps.forEach((s) => { svg += card(s); });

// legend / live loop box (bottom left)
{
  const x = leftPad, y = rowBottom - 20, w = pos[8].x - 30 - x;
  svg += `<rect x="${x}" y="${y}" width="${w}" height="${boxH + 20}" rx="16" fill="${C.bg0}" fill-opacity="0.55" stroke="${C.cardEdge}"/>`;
  svg += text(x + 20, y + 32, "WHO DOES WHAT", { size: 13, weight: 800, spacing: 0.5 });
  const rows = [
    [C.you, "You", "talk, record, press markers, say “VidAI …”"],
    [C.claude, "Claude", "asks, decides, edits, writes new effects live"],
    [C.vidai, "VidAI", "records, measures, renders (Python + C, local)"],
    [C.live, "Live loop", "sound · speech · screen text → effects in ms"],
  ];
  rows.forEach(([col, name, desc], i) => {
    const yy = y + 60 + i * 30;
    svg += `<rect x="${x + 20}" y="${yy - 12}" width="14" height="14" rx="4" fill="${col}"/>`;
    svg += text(x + 42, yy, name, { size: 13, weight: 700, fill: col });
    svg += text(x + 130, yy, desc, { size: 12, fill: C.muted });
  });
}

// footer
svg += text(W / 2, H - 46, "free & local · Linux · camera + screen + mic · 🤗 small AI models · VidAI talks · 104 tests passing · more coming",
  { size: 13, fill: C.muted, anchor: "middle" });
svg += text(W / 2, H - 22, "cd VidAI && claude   →   vidai", { size: 12, fill: C.line, anchor: "middle", mono: true });
svg += `</svg>`;

// ---------------------------------------------------------------- write + rasterise
const outDir = __dirname;
const svgPath = path.join(outDir, "vidai_flow.svg");
const htmlPath = path.join(outDir, "vidai_flow.html");
const pngPath = path.join(outDir, "vidai_flow.png");
fs.writeFileSync(svgPath, svg);
fs.writeFileSync(htmlPath, `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:${C.bg0}}svg{display:block}</style></head><body>${svg}</body></html>`);
console.log("wrote", svgPath, "and", htmlPath);

const scale = Number(process.env.SCALE || 2);
let done = false;
for (const bin of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
  try {
    execFileSync(bin, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
      `--force-device-scale-factor=${scale}`, `--window-size=${W},${H + 140}`, `--screenshot=${pngPath}`, `file://${htmlPath}`],
      { stdio: "ignore", timeout: 60000 });
    done = fs.existsSync(pngPath);
    if (done) {
      try { execFileSync("convert", [pngPath, "-crop", `${W * scale}x${H * scale}+0+0`, "+repage", pngPath], { stdio: "ignore" }); }
      catch (e) { /* ImageMagick missing: keep the uncropped capture */ }
      console.log("wrote", pngPath, `(${W * scale}x${H * scale}, via ${bin})`);
      break;
    }
  } catch (e) { /* try next */ }
}
if (!done) {
  try {
    execFileSync("convert", ["-density", String(96 * scale), svgPath, pngPath], { stdio: "ignore" });
    console.log("wrote", pngPath, "(via ImageMagick)");
  } catch (e) {
    console.error("could not rasterise: install Chrome or ImageMagick; vidai_flow.svg is still usable");
    process.exit(1);
  }
}
