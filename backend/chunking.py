"""
Structure-aware, hierarchical PDF chunking for DocChat.

Pipeline
--------
1. PARSE      PyMuPDF -> typed elements (heading / paragraph / list / table),
              each stamped with its page number. Running headers, footers and
              page numbers are removed; two-column pages are put back in
              reading order; tables are extracted as row/column grids.
2. STRUCTURE  Elements are folded into a section tree using heading levels
              (from the PDF outline when it is trustworthy, otherwise from font
              size / weight). Every section knows its full heading path:
              ["3 Methods", "3.2 Data collection"].
3. CHUNK      Each section is chunked on its own, so a chunk never straddles
              two sections. Inside a section paragraphs are packed whole,
              tables become their own chunks (split by rows with the header
              row repeated), and only over-long paragraphs are split at
              sentence boundaries (with overlap).

The output `Chunk` carries everything needed downstream: body text, heading
path, page range, chunk type, and (section_id, seq) so the retriever can pull
in neighbouring chunks of the same section at answer time.

`contextualize()` builds the text that is actually embedded / BM25-indexed:
a one-line breadcrumb ("Section: A > B | Page 4") followed by the body.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pymupdf

logger = logging.getLogger("docchat.chunking")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & PUBLIC DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1200          # target ceiling for a text chunk
    min_chars: int = 250           # a trailing chunk smaller than this is merged back
    overlap_chars: int = 150       # overlap used ONLY when a paragraph must be split
    max_table_chars: int = 3000    # tables are kept whole up to this size
    margin_zone: float = 0.08      # top/bottom fraction of a page searched for headers/footers
    heading_size_ratio: float = 1.12   # font >= body * ratio  => heading candidate
    max_heading_chars: int = 200
    max_heading_levels: int = 6
    detect_tables: bool = True
    table_strategy: str = "lines"  # PyMuPDF: "lines" (ruled tables) | "text" (borderless, noisier)

    # ── OCR fallback for scanned / image-heavy pages (only used if an OCR provider is passed) ──
    ocr_min_chars: int = 40                # a page with fewer extractable chars is "text-poor"
    ocr_textless_coverage: float = 0.10    # text-poor page + images covering >= this => transcribe whole page
    ocr_figure_coverage: float = 0.12      # text page + images covering >= this => transcribe figures only
    ocr_background_fraction: float = 0.80  # one image covering more than this is a backdrop, not a figure
    ocr_min_image_fraction: float = 0.03   # ignore images smaller than this (logos, icons, bullets)
    ocr_figures: bool = True               # also OCR text inside figures on pages that have a text layer
    ocr_max_pages: int = 24                # hard cap per document (protects a free-tier quota)
    ocr_dpi: int = 150
    ocr_max_side_px: int = 1800


@dataclass
class Chunk:
    text: str                      # body text (no breadcrumb prefix)
    kind: str                      # "text" | "table"
    section_id: int
    heading_path: List[str]
    page_start: int                # 1-indexed
    page_end: int                  # 1-indexed, inclusive
    seq: int = -1                  # global order within the document
    overlaps_prev: bool = False    # True if produced by splitting an over-long paragraph

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class OCRTarget:
    """One page image handed to an OCR provider."""
    page: int                      # 1-indexed
    mode: str                      # "full" = transcribe the page | "figures" = only text inside images
    image: bytes
    mime: str = "image/jpeg"


@dataclass
class OCRResult:
    text: Dict[int, str] = field(default_factory=dict)      # page -> markdown ("" = blank page)
    skipped: Dict[int, str] = field(default_factory=dict)   # page -> human-readable reason


# A provider takes every page that needs OCR at once (so it can batch, rate-limit and budget).
OCRProvider = Callable[[List[OCRTarget]], OCRResult]


@dataclass
class ChunkedDocument:
    chunks: List[Chunk]
    page_count: int
    section_count: int
    warnings: List[str] = field(default_factory=list)   # user-facing notes (skipped OCR, etc.)
    ocr_pages: List[int] = field(default_factory=list)  # pages whose text came from OCR


def contextualize(
    breadcrumb: Optional[str],
    page_start: Optional[int],
    page_end: Optional[int],
    chunk_type: str,
    body: str,
) -> Tuple[str, int]:
    """
    Return (text_to_embed, body_offset). The prefix gives both the dense model
    and BM25 the heading context that a bare paragraph lacks, and gives the LLM
    a section/page label to cite. `body_offset` marks where the body starts so
    callers can strip the prefix again (e.g. for source excerpts).
    """
    parts: List[str] = []
    if breadcrumb:
        parts.append(f"Section: {breadcrumb}")
    if page_start:
        if page_end and page_end != page_start:
            parts.append(f"Pages {page_start}-{page_end}")
        else:
            parts.append(f"Page {page_start}")
    if chunk_type == "table":
        parts.append("Table")
    if not parts:
        return body, 0
    header = "[" + " | ".join(parts) + "]\n\n"
    return header + body, len(header)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Block:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    size: float          # dominant font size (rounded to 0.5pt)
    bold: bool
    n_lines: int
    is_list: bool
    chars: int


@dataclass
class _Table:
    bbox: Tuple[float, float, float, float]
    rows: List[List[str]]


@dataclass
class _PageData:
    number: int
    width: float
    height: float
    blocks: List[_Block]
    tables: List[_Table]
    images: List[Tuple[float, float, float, float]] = field(default_factory=list)  # significant images, top-to-bottom
    image_coverage: float = 0.0     # fraction of the page covered by significant images
    biggest_image: float = 0.0      # largest single image, as a fraction of the page


@dataclass
class _FigureNote:
    """Text transcribed from figures on a page that also has a native text layer."""
    bbox: Tuple[float, float, float, float]
    text: str


@dataclass
class _Element:
    kind: str                       # heading | paragraph | list | table
    text: str
    page: int
    page_end: int
    level: int = 0                  # headings only
    size: float = 0.0               # headings only; 0 => emphasis-only heading (bold / CAPS at body size)
    rows: Optional[List[List[str]]] = None   # tables only
    caption: str = ""               # tables only
    at_page_top: bool = False       # tables only (continuation detection)


@dataclass
class _Section:
    id: int
    level: int
    title: str
    path: List[str]
    elements: List[_Element] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
_MARKER = r"(?:[•●▪◦‣∙·\-–—*]|\d{1,3}[.)]|\(\d{1,3}\)|[a-z]\)|\([a-z]\))"
_LIST_RE = re.compile(rf"^\s*{_MARKER}\s+")
_MARKER_ONLY_RE = re.compile(rf"^\s*{_MARKER}\s*$")     # PDFs often emit the bullet glyph as its own line
_CAPTION_START_RE = re.compile(r"^(?:table|tab\.|figure|fig\.|chart|exhibit)\s*[A-Z]?\d+", re.I)
_CAPTION_RE = re.compile(r"^(?:table|tab\.|figure|fig\.|chart|exhibit)\s*[A-Z]?\d+[a-z]?\s*[.:\u2013\u2014-]", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")
_SENT_END = (".", "!", "?", "…", '"', "”", "’")


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "").replace("\x00", "")
    return re.sub(r"[ \t\u00a0]+", " ", s).strip()


def _join_lines(lines: Sequence[str]) -> str:
    """Join wrapped lines; undo end-of-line hyphenation ("exam-" + "ple")."""
    out = lines[0]
    for nxt in lines[1:]:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and nxt[:1].islower():
            out = out[:-1] + nxt
        else:
            out = f"{out} {nxt}"
    return out


def _norm_title(s: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", s).lower()).strip()


def _round_half(x: float) -> float:
    return round(x * 2) / 2


def _overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of rectangle `a` covered by rectangle `b`."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ((x1 - x0) * (y1 - y0)) / area if area > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PARSING: page -> blocks + tables
# ─────────────────────────────────────────────────────────────────────────────

def _span_is_bold(span: dict) -> bool:
    if span.get("flags", 0) & 16:
        return True
    font = span.get("font", "").lower()
    return any(k in font for k in ("bold", "black", "heavy", "semibold", "demi"))


def _make_block(raw: dict, page: int) -> Optional[_Block]:
    lines: List[str] = []
    sizes: Counter = Counter()
    bold_chars = total = 0
    for line in raw.get("lines", []):
        parts: List[str] = []
        for sp in line.get("spans", []):
            t = sp.get("text", "")
            parts.append(t)
            n = len(t.strip())
            if n:
                sizes[_round_half(sp["size"])] += n
                total += n
                if _span_is_bold(sp):
                    bold_chars += n
        lt = _clean("".join(parts))
        if lt:
            lines.append(lt)
    if not lines or total == 0:
        return None

    glued: List[str] = []
    for ln in lines:
        if glued and _MARKER_ONLY_RE.match(glued[-1]):
            glued[-1] = f"{glued[-1].strip()} {ln}"
        else:
            glued.append(ln)
    lines = glued

    is_list = bool(_LIST_RE.match(lines[0]))
    if is_list:
        items: List[str] = []
        for ln in lines:
            if _LIST_RE.match(ln) or not items:
                items.append(ln)
            else:
                items[-1] = _join_lines([items[-1], ln])
        text = "\n".join(items)
    else:
        text = _join_lines(lines)

    return _Block(
        text=text,
        page=page,
        bbox=tuple(raw["bbox"]),
        size=sizes.most_common(1)[0][0],
        bold=(bold_chars / total) >= 0.8,
        n_lines=len(lines),
        is_list=is_list,
        chars=total,
    )


def _clean_rows(rows: Sequence[Sequence[Optional[str]]]) -> Optional[List[List[str]]]:
    """Normalise a raw PyMuPDF table grid; None if it doesn't look like a real table."""
    grid = [
        [_clean((c or "").replace("\n", " ")).replace("|", "\\|") for c in row]
        for row in rows
    ]
    grid = [r for r in grid if any(r)]                       # drop empty rows
    if len(grid) < 2:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] for r in grid)]  # drop empty columns
    grid = [[r[j] for j in keep] for r in grid]
    if len(grid[0]) < 2:
        return None
    filled = sum(1 for r in grid for c in r if c)
    if filled < 0.3 * len(grid) * len(grid[0]):              # mostly empty => layout box, not a table
        return None
    return grid


def _find_tables(page: "pymupdf.Page", cfg: ChunkingConfig) -> List[_Table]:
    if not cfg.detect_tables:
        return []
    try:
        finder = page.find_tables(strategy=cfg.table_strategy)
    except Exception:  # noqa: BLE001 — table detection must never sink an upload
        logger.debug("find_tables failed on page %s", page.number + 1, exc_info=True)
        return []
    out: List[_Table] = []
    for tab in finder.tables:
        try:
            rows = _clean_rows(tab.extract())
        except Exception:  # noqa: BLE001
            continue
        if rows:
            out.append(_Table(bbox=tuple(tab.bbox), rows=rows))
    return out


def _read_page(page: "pymupdf.Page", cfg: ChunkingConfig) -> _PageData:
    number = page.number + 1
    tables = _find_tables(page, cfg)
    raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    blocks: List[_Block] = []
    for rb in raw.get("blocks", []):
        if rb.get("type") != 0:
            continue
        blk = _make_block(rb, number)
        if blk is None:
            continue
        # text that lives inside a detected table is emitted via the table, not twice
        if any(_overlap_ratio(blk.bbox, t.bbox) > 0.5 for t in tables):
            continue
        blocks.append(blk)
    images, coverage, biggest = _page_images(page, cfg)
    return _PageData(number, page.rect.width, page.rect.height, blocks, tables, images, coverage, biggest)


def _page_images(page: "pymupdf.Page", cfg: ChunkingConfig) -> Tuple[List[Tuple[float, float, float, float]], float, float]:
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001
        return [], 0.0, 0.0
    r = page.rect
    page_area = r.width * r.height
    boxes: List[Tuple[float, float, float, float]] = []
    total = biggest = 0.0
    for info in infos:
        x0, y0, x1, y1 = info["bbox"]
        x0, y0, x1, y1 = max(x0, r.x0), max(y0, r.y0), min(x1, r.x1), min(y1, r.y1)
        frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area if page_area else 0.0
        if frac < cfg.ocr_min_image_fraction:
            continue
        boxes.append((x0, y0, x1, y1))
        total += frac
        biggest = max(biggest, frac)
    boxes.sort(key=lambda b: b[1])
    return boxes, min(total, 1.0), biggest


# ─────────────────────────────────────────────────────────────────────────────
# READING ORDER (two-column pages)
# ─────────────────────────────────────────────────────────────────────────────

def _reading_order(blocks: List[_Block], page_w: float) -> List[_Block]:
    blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    mid = page_w / 2

    def wide(b: _Block) -> bool:
        return (b.bbox[2] - b.bbox[0]) > 0.6 * page_w

    def is_left(b: _Block) -> bool:
        return (b.bbox[0] + b.bbox[2]) / 2 < mid

    narrow = [b for b in blocks if not wide(b)]
    left = [b for b in narrow if is_left(b)]
    right = [b for b in narrow if not is_left(b)]
    # Only treat the page as multi-column if left/right blocks genuinely sit side by side.
    side_by_side = any(l.bbox[1] < r.bbox[3] and r.bbox[1] < l.bbox[3] for l in left for r in right)
    if len(left) < 2 or len(right) < 2 or not side_by_side:
        return blocks

    ordered: List[_Block] = []
    seg_l: List[_Block] = []
    seg_r: List[_Block] = []

    def flush() -> None:
        ordered.extend(seg_l)
        ordered.extend(seg_r)
        seg_l.clear()
        seg_r.clear()

    for b in blocks:
        if wide(b):
            flush()
            ordered.append(b)
        elif is_left(b):
            seg_l.append(b)
        else:
            seg_r.append(b)
    flush()
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# RUNNING HEADERS / FOOTERS
# ─────────────────────────────────────────────────────────────────────────────

def _in_margin(b: _Block, page: _PageData, cfg: ChunkingConfig) -> bool:
    zone = cfg.margin_zone * page.height
    return b.bbox[3] <= zone or b.bbox[1] >= page.height - zone


def _furniture_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def _find_furniture(pages: List[_PageData], cfg: ChunkingConfig) -> set:
    """Margin text that repeats on many pages (digits normalised) is header/footer."""
    counts: Counter = Counter()
    for pg in pages:
        seen = set()
        for b in pg.blocks:
            if _in_margin(b, pg, cfg):
                seen.add(_furniture_key(b.text))
        counts.update(seen)
    threshold = max(3, int(0.4 * len(pages)))
    return {k for k, c in counts.items() if c >= threshold}


def _is_furniture(b: _Block, pg: _PageData, keys: set, cfg: ChunkingConfig) -> bool:
    if not _in_margin(b, pg, cfg):
        return False
    return bool(_PAGE_NUM_RE.match(b.text)) or _furniture_key(b.text) in keys


# ─────────────────────────────────────────────────────────────────────────────
# HEADING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _body_font_size(pages: List[_PageData]) -> float:
    weight: Counter = Counter()
    for pg in pages:
        for b in pg.blocks:
            weight[b.size] += b.chars
    return weight.most_common(1)[0][0] if weight else 10.0


def _heading_style(b: _Block, body: float, cfg: ChunkingConfig) -> Optional[float]:
    """
    None  -> not a heading.
    float -> heading; the value is the font size if larger than body text,
             or 0.0 for emphasis-only headings (bold / ALL-CAPS at body size).
    """
    t = b.text
    if b.is_list or b.n_lines > 3 or not (2 <= len(t) <= cfg.max_heading_chars):
        return None
    if _CAPTION_START_RE.match(t) or t[-1] in ",;":
        return None
    letters = sum(c.isalpha() for c in t)
    if letters < 2 or letters / len(t) < 0.4:
        return None
    larger = b.size >= body * cfg.heading_size_ratio
    if t.endswith(".") and b.size < body * 1.3:      # sentences end with periods, headings rarely do
        return None
    if larger:
        return b.size
    short = b.n_lines <= 2
    if b.bold and b.size >= body * 0.98 and short and len(t) <= 120:
        return 0.0
    if t.isupper() and letters >= 3 and b.size >= body * 0.98 and short and len(t) <= 100:
        return 0.0
    return None


def _assign_heading_levels(elements: List[_Element], cfg: ChunkingConfig) -> None:
    """Rank heading font sizes (largest = level 1); emphasis-only headings go last."""
    heuristic = [e for e in elements if e.kind == "heading" and e.level == 0]
    if not heuristic:
        return
    sizes = sorted({e.size for e in heuristic if e.size > 0}, reverse=True)
    rank = {s: i + 1 for i, s in enumerate(sizes)}
    emphasis_level = len(sizes) + 1
    for e in heuristic:
        e.level = min(rank.get(e.size, emphasis_level), cfg.max_heading_levels)


# ── PDF outline (bookmarks) ──────────────────────────────────────────────────

def _toc_entries(doc: "pymupdf.Document") -> List[Tuple[int, str, int]]:
    try:
        raw = doc.get_toc(simple=True)
    except Exception:  # noqa: BLE001
        return []
    return [
        (int(lvl), _norm_title(title), int(page))
        for lvl, title, page in raw
        if title and title.strip() and page >= 1 and _norm_title(title)
    ]


def _new_toc_index(entries: List[Tuple[int, str, int]]) -> Dict[int, List[list]]:
    idx: Dict[int, List[list]] = defaultdict(list)
    for lvl, title, page in entries:
        idx[page].append([title, lvl, False])      # [normalised title, level, consumed]
    return idx


def _match_toc(idx: Dict[int, List[list]], page: int, text: str) -> Optional[int]:
    n = _norm_title(text)
    if not n:
        return None
    for entry in idx.get(page, []):
        if entry[2]:
            continue
        a = entry[0]
        lo, hi = sorted((len(a), len(n)))
        if a == n or (lo >= 6 and (a.startswith(n) or n.startswith(a)) and lo / hi >= 0.6):
            entry[2] = True                          # each outline entry matches once
            return entry[1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def _continues(prev: str, nxt: str) -> bool:
    """Does `nxt` continue the paragraph `prev` (split by a column or page break)?"""
    if prev.endswith("-") and nxt[:1].islower():
        return True
    return not prev.endswith(_SENT_END) and nxt[:1].islower()


def _add_paragraph_or_list(elements: List[_Element], el: _Element) -> None:
    prev = elements[-1] if elements else None
    if prev is not None and el.page - prev.page_end <= 1:
        if el.kind == "paragraph":
            if prev.kind == "paragraph" and _continues(prev.text, el.text):
                prev.text = _join_lines([prev.text, el.text])
                prev.page_end = el.page
                return
            # caption printed *below* a table
            if prev.kind == "table" and not prev.caption and _CAPTION_RE.match(el.text) and len(el.text) <= 300:
                prev.caption = el.text
                return
        elif el.kind == "list" and prev.kind == "list":
            prev.text += "\n" + el.text
            prev.page_end = el.page
            return
    elements.append(el)


def _add_table(elements: List[_Element], table_rows: List[List[str]], page_no: int, at_page_top: bool) -> None:
    el = _Element(
        kind="table", text="", page=page_no, page_end=page_no,
        rows=table_rows, at_page_top=at_page_top,
    )
    prev = elements[-1] if elements else None
    if prev is not None and prev.kind == "table" and prev.page_end == page_no - 1 \
            and el.at_page_top and prev.rows and len(prev.rows[0]) == len(table_rows[0]):
        # table continued on the next page: append rows, drop a repeated header row
        rows = table_rows[1:] if table_rows[0] == prev.rows[0] else table_rows
        prev.rows.extend(rows)
        prev.page_end = page_no
        return
    if prev is not None and prev.kind == "paragraph" and _CAPTION_RE.match(prev.text) and len(prev.text) <= 300:
        el.caption = prev.text                        # caption printed above the table
        el.page = prev.page
        elements.pop()
    elements.append(el)


# ── OCR fallback ─────────────────────────────────────────────────────────────

def _page_text_chars(pg: _PageData, furniture: set, cfg: ChunkingConfig) -> int:
    n = sum(b.chars for b in pg.blocks if not _is_furniture(b, pg, furniture, cfg))
    return n + sum(len(c) for t in pg.tables for row in t.rows for c in row)


def _plan_ocr(pages: List[_PageData], furniture: set, cfg: ChunkingConfig) -> Tuple[List[int], List[int]]:
    """
    Decide which pages need vision OCR. Returns (full_pages, figure_pages).
      full    : almost no extractable text but real images  -> scanned / image-only page
      figures : has a text layer, plus figures that may hold text (charts, screenshots)
    """
    full: List[int] = []
    figs: List[int] = []
    for pg in pages:
        if _page_text_chars(pg, furniture, cfg) < cfg.ocr_min_chars:
            if pg.image_coverage >= cfg.ocr_textless_coverage:
                full.append(pg.number)
        elif (cfg.ocr_figures and pg.image_coverage >= cfg.ocr_figure_coverage
              and pg.biggest_image < cfg.ocr_background_fraction):
            figs.append(pg.number)
    return full, figs


def _render_page(page: "pymupdf.Page", cfg: ChunkingConfig) -> bytes:
    rect = page.rect
    zoom = min(cfg.ocr_dpi / 72.0, cfg.ocr_max_side_px / max(rect.width, rect.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("jpeg", jpg_quality=80)


def _ranges(nums: Sequence[int]) -> str:
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    out: List[str] = []
    nums = sorted(set(nums))
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def _run_ocr(
    doc: "pymupdf.Document", pages: List[_PageData], furniture: set,
    cfg: ChunkingConfig, ocr: Optional[OCRProvider],
) -> Tuple[Dict[int, Tuple[str, str]], List[str]]:
    """Returns ({page: (mode, markdown)}, user-facing warnings)."""
    full, figs = _plan_ocr(pages, furniture, cfg)
    warnings: List[str] = []
    if not full and not figs:
        return {}, warnings

    if ocr is None:
        if full:
            warnings.append(
                f"Pages {_ranges(full)} look like scanned images with no text layer. OCR is not "
                "available, so their content cannot be searched."
            )
        if figs:
            warnings.append(f"Pages {_ranges(figs)} contain images whose text is not indexed (OCR is not available).")
        return {}, warnings

    planned = [(p, "full") for p in full] + [(p, "figures") for p in figs]   # scanned pages first
    skipped: Dict[int, str] = {}
    if len(planned) > cfg.ocr_max_pages:
        for pno, _ in planned[cfg.ocr_max_pages:]:
            skipped[pno] = f"over the per-document OCR limit of {cfg.ocr_max_pages} pages"
        planned = planned[: cfg.ocr_max_pages]

    targets: List[OCRTarget] = []
    for pno, mode in planned:
        try:
            targets.append(OCRTarget(page=pno, mode=mode, image=_render_page(doc[pno - 1], cfg)))
        except Exception:  # noqa: BLE001
            logger.exception("could not render page %s for OCR", pno)
            skipped[pno] = "the page could not be rendered"

    result = OCRResult()
    if targets:
        try:
            result = ocr(targets)
        except Exception:  # noqa: BLE001 — a broken OCR service must not fail the whole upload
            logger.exception("OCR provider raised")
            result = OCRResult(skipped={t.page: "the OCR service failed" for t in targets})
    skipped.update(result.skipped)

    mode_of = {t.page: t.mode for t in targets}
    got = {pno: (mode_of[pno], txt) for pno, txt in result.text.items() if pno in mode_of}
    for t in targets:
        if t.page not in got and t.page not in skipped:
            skipped[t.page] = "OCR returned no output"

    by_reason: Dict[str, List[int]] = defaultdict(list)
    for pno, reason in skipped.items():
        by_reason[reason].append(pno)
    for reason, pnos in by_reason.items():
        warnings.append(f"Pages {_ranges(pnos)} were not OCR'd: {reason}.")
    return got, warnings


# ── OCR markdown -> elements ─────────────────────────────────────────────────

_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_MD_EMPHASIS_RE = re.compile(r"[*_`]+")


def _strip_fence(md: str) -> str:
    md = md.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n?```$", md, re.S)
    return m.group(1) if m else md


def _md_heading_texts(md: str) -> List[str]:
    out = []
    for line in _strip_fence(md).splitlines():
        m = _MD_HEADING_RE.match(line)
        if m:
            out.append(_clean(_MD_EMPHASIS_RE.sub("", m.group(2))))
    return [h for h in out if h]


def _parse_md_table(lines: Sequence[str]) -> Optional[List[List[str]]]:
    rows: List[List[str]] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|") and not s.endswith("\\|"):
            s = s[:-1]
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]
        if cells and all(_MD_SEP_CELL_RE.match(c) for c in cells if c) and any(cells):
            continue                                            # the |---|---| separator row
        rows.append(cells)
    return _clean_rows(rows)


def _add_markdown(
    elements: List[_Element], md: str, page: int, toc_idx: Optional[Dict[int, List[list]]],
) -> None:
    """Feed OCR markdown through the same element pipeline as native text."""
    lines = _strip_fence(md).replace("\r", "").split("\n")
    para: List[str] = []
    emitted = False        # anything emitted for this page yet? (table-continuation detection)

    def flush_para() -> None:
        nonlocal emitted
        text = [_clean(l) for l in para if l.strip()]
        para.clear()
        if text:
            _add_paragraph_or_list(elements, _Element("paragraph", _join_lines(text), page, page))
            emitted = True

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_para()
            i += 1
            continue

        m = _MD_HEADING_RE.match(line)
        if m:
            flush_para()
            title = _clean(_MD_EMPHASIS_RE.sub("", m.group(2)))
            if title:
                level = _match_toc(toc_idx, page, title) if toc_idx is not None else None
                elements.append(_Element("heading", title, page, page, level=level or len(m.group(1))))
                emitted = True
            i += 1
            continue

        if line.lstrip().startswith("|"):
            flush_para()
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = _parse_md_table(lines[i:j])
            if rows:
                _add_table(elements, rows, page, at_page_top=not emitted)
                emitted = True
            else:                                               # not a real table: keep as text
                para.extend(l.strip().strip("|").replace("|", " ") for l in lines[i:j])
                flush_para()
            i = j
            continue

        if _LIST_RE.match(line):
            flush_para()
            items: List[str] = []
            j = i
            while j < len(lines) and lines[j].strip() and not _MD_HEADING_RE.match(lines[j]) \
                    and not lines[j].lstrip().startswith("|"):
                ln = _clean(lines[j])
                if _LIST_RE.match(ln) or not items:
                    items.append(ln)
                else:
                    items[-1] = _join_lines([items[-1], ln])
                j += 1
            _add_paragraph_or_list(elements, _Element("list", "\n".join(items), page, page))
            emitted = True
            i = j
            continue

        para.append(line)
        i += 1
    flush_para()


def _figure_note(text: str) -> Optional[str]:
    text = _strip_fence(text).strip()
    if not text or text.upper().rstrip(".") == "NONE":
        return None
    return "[Text and data from a figure/image on this page]\n" + text


def _insert_by_y(flow: list, item: object, y: float) -> None:
    pos = next((i for i, b in enumerate(flow) if b.bbox[1] >= y - 1), len(flow))
    flow.insert(pos, item)


# ── main assembly ────────────────────────────────────────────────────────────

def _extract_elements(
    doc: "pymupdf.Document", cfg: ChunkingConfig, ocr: Optional[OCRProvider] = None,
) -> Tuple[List[_Element], List[int], List[str]]:
    pages = [_read_page(doc[i], cfg) for i in range(doc.page_count)]
    body = _body_font_size(pages)
    furniture = _find_furniture(pages, cfg)
    ocr_text, warnings = _run_ocr(doc, pages, furniture, cfg, ocr)

    # Trust the PDF outline only if most of its entries can be located in the text
    # (native blocks, or headings found in OCR'd pages).
    entries = _toc_entries(doc)
    toc_idx: Optional[Dict[int, List[list]]] = None
    if len(entries) >= 3:
        probe = _new_toc_index(entries)
        matched = sum(
            1 for pg in pages for b in pg.blocks
            if not b.is_list and _match_toc(probe, pg.number, b.text) is not None
        )
        for pno, (mode, md) in ocr_text.items():
            if mode == "full":
                matched += sum(1 for h in _md_heading_texts(md) if _match_toc(probe, pno, h) is not None)
        if matched / len(entries) >= 0.5:
            toc_idx = _new_toc_index(entries)
    logger.info("chunking: body_font=%.1fpt headings=%s ocr_pages=%d",
                body, "pdf-outline" if toc_idx else "font-heuristic", len(ocr_text))

    elements: List[_Element] = []
    for pg in pages:
        entry = ocr_text.get(pg.number)
        if entry and entry[0] == "full":                # scanned page: OCR markdown replaces the page
            _add_markdown(elements, entry[1], pg.number, toc_idx)
            continue

        flow: list = _reading_order(
            [b for b in pg.blocks if not _is_furniture(b, pg, furniture, cfg)], pg.width
        )
        for tbl in pg.tables:                           # place each table at its vertical position
            _insert_by_y(flow, tbl, tbl.bbox[1])
        if entry and entry[0] == "figures":
            note = _figure_note(entry[1])
            if note:
                anchor = pg.images[0] if pg.images else (0, pg.height, 0, pg.height)
                _insert_by_y(flow, _FigureNote(tuple(anchor), note), anchor[1])

        for item in flow:
            if isinstance(item, _Table):
                _add_table(elements, item.rows, pg.number, item.bbox[1] < 0.25 * pg.height)
                continue
            if isinstance(item, _FigureNote):
                _add_paragraph_or_list(elements, _Element("paragraph", item.text, pg.number, pg.number))
                continue
            b: _Block = item
            if toc_idx is not None:
                level = None if b.is_list else _match_toc(toc_idx, pg.number, b.text)
                if level is not None:
                    elements.append(_Element("heading", b.text, pg.number, pg.number, level=level))
                    continue
            else:
                style = _heading_style(b, body, cfg)
                if style is not None:
                    elements.append(_Element("heading", b.text, pg.number, pg.number, size=style))
                    continue
            _add_paragraph_or_list(
                elements,
                _Element("list" if b.is_list else "paragraph", b.text, pg.number, pg.number),
            )

    if toc_idx is None:
        _assign_heading_levels(elements, cfg)
    return elements, sorted(p for p, (m, t) in ocr_text.items() if t.strip()), warnings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION TREE
# ─────────────────────────────────────────────────────────────────────────────

def _build_sections(elements: List[_Element]) -> List[_Section]:
    """
    Fold the flat element stream into sections. `path` is the chain of ancestor
    headings, so a chunk under "3.2 Data" knows it lives in "3 Methods".
    Section 0 holds content that appears before the first heading.
    """
    sections: List[_Section] = [_Section(id=0, level=0, title="", path=[])]
    stack: List[_Section] = []
    current = sections[0]
    for el in elements:
        if el.kind == "heading":
            while stack and stack[-1].level >= el.level:
                stack.pop()
            title = el.text[:120]
            sec = _Section(
                id=len(sections), level=el.level, title=title,
                path=(stack[-1].path if stack else []) + [title],
            )
            sections.append(sec)
            stack.append(sec)
            current = sec
        else:
            current.elements.append(el)
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Sentence-aware split of an over-long paragraph, with a small overlap."""
    units: List[str] = []
    for s in _SENT_SPLIT_RE.split(text):
        s = s.strip()
        while len(s) > max_chars:                    # a single "sentence" that is still too long
            cut = s.rfind(" ", 0, max_chars)
            if cut < max_chars * 0.5:
                cut = max_chars
            units.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            units.append(s)

    def size(us: List[str]) -> int:
        return sum(len(u) for u in us) + max(len(us) - 1, 0)

    pieces: List[str] = []
    cur: List[str] = []
    for u in units:
        if cur and size(cur) + 1 + len(u) > max_chars:
            pieces.append(" ".join(cur))
            tail: List[str] = []
            for prev in reversed(cur):
                if size(tail) + 1 + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
            cur = tail if size(tail) + 1 + len(u) <= max_chars else []
        cur.append(u)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_list(text: str, max_chars: int) -> List[str]:
    """Split a long list between items (never inside one)."""
    pieces: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for item in text.split("\n"):
        if cur and cur_len + 1 + len(item) > max_chars:
            pieces.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(item)
        cur_len += len(item) + 1
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_chunks(el: _Element, cfg: ChunkingConfig) -> List[str]:
    """Markdown table(s). Oversized tables are split by rows; the header row is repeated."""
    rows = el.rows or []
    header, body = rows[0], rows[1:]
    head_md = _md_row(header) + "\n" + _md_row(["---"] * len(header))
    caption = el.caption.strip()

    def render(part_rows: Sequence[Sequence[str]], continued: bool) -> str:
        label = caption + (" (continued)" if continued and caption else "")
        if continued and not caption:
            label = "(table continued)"
        prefix = f"{label}\n\n" if label else ""
        return prefix + head_md + "".join("\n" + _md_row(r) for r in part_rows)

    whole = render(body, False)
    if len(whole) <= cfg.max_table_chars:
        return [whole]

    parts: List[str] = []
    cur: List[Sequence[str]] = []
    cur_len = len(head_md) + len(caption) + 30
    for r in body:
        rl = len(_md_row(r)) + 1
        if cur and cur_len + rl > cfg.max_table_chars:
            parts.append(render(cur, bool(parts)))
            cur, cur_len = [], len(head_md) + len(caption) + 30
        cur.append(r)
        cur_len += rl
    if cur:
        parts.append(render(cur, bool(parts)))
    return parts


def _chunk_section(sec: _Section, cfg: ChunkingConfig) -> List[Chunk]:
    out: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0
    p0 = p1 = 0

    def emit(text: str, kind: str, ps: int, pe: int, overlaps: bool = False) -> None:
        out.append(Chunk(text=text, kind=kind, section_id=sec.id, heading_path=sec.path,
                         page_start=ps, page_end=pe, overlaps_prev=overlaps))

    def flush() -> None:
        nonlocal buf, buf_len, p0, p1
        if buf:
            emit("\n\n".join(buf), "text", p0, p1)
        buf, buf_len, p0, p1 = [], 0, 0, 0

    for el in sec.elements:
        if el.kind == "table":
            flush()
            for t in _table_chunks(el, cfg):
                emit(t, "table", el.page, el.page_end)
            continue

        if len(el.text) > cfg.max_chars:             # too big for one chunk: split, standalone
            flush()
            pieces = (
                _split_list(el.text, cfg.max_chars) if el.kind == "list"
                else _split_long_text(el.text, cfg.max_chars, cfg.overlap_chars)
            )
            for i, piece in enumerate(pieces):
                emit(piece, "text", el.page, el.page_end, overlaps=(i > 0 and el.kind != "list"))
            continue

        if buf and buf_len + 2 + len(el.text) > cfg.max_chars:
            flush()
        if not buf:
            p0 = el.page
        buf.append(el.text)
        buf_len += len(el.text) + (2 if len(buf) > 1 else 0)
        p1 = el.page_end
    flush()

    # fold a tiny trailing text chunk back into its predecessor
    if len(out) >= 2:
        last, prev = out[-1], out[-2]
        if (last.kind == prev.kind == "text" and not last.overlaps_prev
                and len(last.text) < cfg.min_chars
                and len(prev.text) + 2 + len(last.text) <= cfg.max_chars * 1.25):
            prev.text += "\n\n" + last.text
            prev.page_end = max(prev.page_end, last.page_end)
            out.pop()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def chunk_pdf(
    pdf_bytes: bytes,
    cfg: Optional[ChunkingConfig] = None,
    ocr: Optional[OCRProvider] = None,
) -> ChunkedDocument:
    """
    Parse `pdf_bytes` and return structure-aware chunks.

    `ocr` (optional) is called once with every page that needs vision OCR
    (scanned pages, or pages with figures) and returns markdown per page.
    Without it, such pages are reported in `warnings` and left unindexed.
    Raises ValueError for unreadable / encrypted PDFs.
    """
    cfg = cfg or ChunkingConfig()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise ValueError("The PDF is password-protected.")
        page_count = doc.page_count
        elements, ocr_pages, warnings = _extract_elements(doc, cfg, ocr)
    finally:
        doc.close()

    sections = _build_sections(elements)
    chunks: List[Chunk] = []
    for sec in sections:
        chunks.extend(_chunk_section(sec, cfg))
    for i, c in enumerate(chunks):
        c.seq = i
    return ChunkedDocument(
        chunks=chunks, page_count=page_count, section_count=len(sections) - 1,
        warnings=warnings, ocr_pages=ocr_pages,
    )
