"""
Gemini vision OCR provider for DocChat (built for the free tier).

Free-tier realities this module is designed around
--------------------------------------------------
* Requests-per-day (RPD) is the scarcest quota, not tokens. So several pages are
  sent per request, and a persistent daily counter stops us before Google does.
* Limits are per project AND per model, differ by model, and are only published in
  AI Studio (https://aistudio.google.com/rate-limit). Nothing is hard-coded: RPM,
  daily budget, model and batch size all come from configuration.
* A 429 can mean "too many per minute" (wait, retry) or "daily quota gone" (stop).
  They are told apart here; only the first is retried.
* Unpaid-tier content may be used by Google to improve its products, and some of it
  may be seen by human reviewers. Do not point this at confidential documents on the
  free tier.

The provider is a plain callable matching `chunking.OCRProvider`:
    provider(list[OCRTarget]) -> OCRResult
It never raises for per-page problems; unreadable pages come back in `skipped`
with a human-readable reason so the API can warn the user.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from chunking import OCRResult, OCRTarget

logger = logging.getLogger("docchat.ocr")

DEFAULT_MODEL = "gemini-2.0-flash-lite"   # check your quota for it in AI Studio: https://aistudio.google.com/rate-limit
_PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini daily quotas reset at midnight Pacific


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT + STRUCTURED OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT = """You are the OCR stage of a document search system. You will receive page images, each \
preceded by a label such as "Page 7 (FULL PAGE)". For every page, return its content as Markdown.

Modes
- FULL PAGE: transcribe everything printed or handwritten on the page.
- FIGURES ONLY: the page already has a text layer. Transcribe ONLY text and data that appear INSIDE \
images, charts, diagrams or screenshots (titles, axis labels, legends, callouts, table cells, values). \
Do not transcribe ordinary body text. If a chart's key values are readable, add one short sentence \
stating what it shows. If there is nothing meaningful (photo, logo, decoration), return exactly NONE.

Rules
- Transcribe faithfully. Do not summarise, translate, correct spelling, or add commentary.
- Reading order: top to bottom; for multi-column layouts read each column in full, left to right.
- Headings: use "#" for the most prominent heading on the page, "##" for the next level, and so on. \
Only mark text that is visually a heading. Do not invent headings.
- Tables: GitHub pipe tables with a header row (| a | b |, then | --- | --- |).
- Lists: one "- " line per item.
- Paragraphs: one line per paragraph, blank line between paragraphs. Do not hard-wrap lines.
- Omit running headers, footers and page numbers.
- Text you cannot read: write [illegible]. Never guess numbers.
- Formulas: plain text or inline LaTeX.
"""


class _OcrPage(BaseModel):
    page: int
    markdown: str


class _OcrBatch(BaseModel):
    pages: List[_OcrPage]


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING + DAILY BUDGET
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Spaces request start times >= 60/rpm seconds apart. Thread-safe."""

    def __init__(self, rpm: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._interval = 60.0 / max(rpm, 0.1)
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            self._sleep(start - now)


class MemoryDailyBudget:
    """Per-process daily request counter (resets at Pacific midnight)."""

    def __init__(self, limit: int):
        self._limit = limit
        self._lock = threading.Lock()
        self._day = None
        self._used = 0

    def acquire(self) -> bool:
        today = datetime.now(_PACIFIC).date()
        with self._lock:
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self._limit:
                return False
            self._used += 1
            return True


class PostgresDailyBudget:
    """
    Daily request counter shared by every worker/process, persisted in Postgres so a
    restart doesn't reset it. `get_conn` is a zero-arg callable returning a psycopg
    connection. The check-and-increment is a single atomic UPSERT.
    """

    def __init__(self, get_conn: Callable, limit: int):
        self._get_conn = get_conn
        self._limit = limit
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_table(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            with self._get_conn() as conn:
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('docchat_ocr_usage_schema'))")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS ocr_usage ("
                    " day DATE PRIMARY KEY, requests INTEGER NOT NULL DEFAULT 0)"
                )
                conn.commit()
            self._ready = True

    def acquire(self) -> bool:
        if self._limit <= 0:
            return False
        self._ensure_table()
        day = datetime.now(_PACIFIC).date()
        with self._get_conn() as conn:
            row = conn.execute(
                """
                INSERT INTO ocr_usage (day, requests) VALUES (%s, 1)
                ON CONFLICT (day) DO UPDATE SET requests = ocr_usage.requests + 1
                    WHERE ocr_usage.requests < %s
                RETURNING requests
                """,
                (day, self._limit),
            ).fetchone()
            conn.commit()
        return row is not None

    def used_today(self) -> int:
        self._ensure_table()
        day = datetime.now(_PACIFIC).date()
        with self._get_conn() as conn:
            row = conn.execute("SELECT requests FROM ocr_usage WHERE day = %s", (day,)).fetchone()
        return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_RETRY_DELAY_RES = (
    re.compile(r"retrydelay\W+(\d+(?:\.\d+)?)s"),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s"),
)
_MAX_RETRY_WAIT = 90.0     # if Google asks for longer than this it is not a per-minute limit


def _retry_delay(text: str) -> Optional[float]:
    for rx in _RETRY_DELAY_RES:
        m = rx.search(text)
        if m:
            return float(m.group(1))
    return None


def classify_error(exc: BaseException) -> Tuple[str, Optional[float]]:
    """
    -> ("retry", delay)  transient: per-minute 429, 5xx, timeouts
       ("abort", None)   stop ALL OCR for this document: daily quota, bad key, unknown model
       ("fail",  None)   this batch failed; others may still work (e.g. 400)
    NB: the daily-quota check reads the error text (Google names the quota, e.g. "...PerDay...").
    """
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code == 429:
        compact = text.replace(" ", "").replace("_", "")
        if "perday" in compact or "daily" in text:
            return "abort", None
        delay = _retry_delay(text)
        if delay is not None and delay > _MAX_RETRY_WAIT:
            return "abort", None
        return "retry", delay
    if code in (500, 502, 503, 504):
        return "retry", _retry_delay(text)
    if code in (401, 403, 404):
        return "abort", None
    name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, ConnectionError)) or "timeout" in name or "connect" in name:
        return "retry", None
    return "fail", None


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class _Abort(Exception):
    """Stop OCR for the whole document (quota / auth)."""


class _BatchFailed(Exception):
    """This batch produced nothing usable; other batches may still succeed."""


class _Run:
    """State shared by the worker threads of one OCR call."""

    def __init__(self) -> None:
        self.abort = threading.Event()
        self.reason = ""

    def stop(self, reason: str) -> None:
        if not self.abort.is_set():
            self.reason = reason
            self.abort.set()


def _strip_fence(md: str) -> str:
    md = md.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n?```$", md, re.S)
    return (m.group(1) if m else md).strip()


class GeminiVisionOCR:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        rpm: float = 10,
        pages_per_request: int = 3,
        concurrency: int = 2,
        daily_budget=None,                    # object with .acquire() -> bool, or None for unlimited
        max_retries: int = 3,
        timeout_s: float = 120,
        client=None,                          # injectable for tests
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._client = client
        self.model = model
        self.pages_per_request = max(1, pages_per_request)
        self.concurrency = max(1, concurrency)
        self.budget = daily_budget
        self.max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(rpm, sleep=sleep)

    # -- public ---------------------------------------------------------------

    def __call__(self, targets: List[OCRTarget]) -> OCRResult:
        result = OCRResult()
        if not targets:
            return result
        n = self.pages_per_request
        batches = [targets[i:i + n] for i in range(0, len(targets), n)]
        run = _Run()
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(batches))) as pool:
            outcomes = list(pool.map(lambda b: self._process_batch(b, run), batches))
        for texts, skipped in outcomes:
            result.text.update(texts)
            result.skipped.update(skipped)
        logger.info("OCR: %d/%d pages transcribed, %d skipped%s", len(result.text), len(targets),
                    len(result.skipped), f" (stopped: {run.reason})" if run.reason else "")
        return result

    # -- internals ------------------------------------------------------------

    def _process_batch(self, batch: List[OCRTarget], run: _Run) -> Tuple[Dict[int, str], Dict[int, str]]:
        texts: Dict[int, str] = {}
        skipped: Dict[int, str] = {}
        fail_reason = "the model returned no text for this page"

        def skip_all(reason: str) -> Tuple[Dict[int, str], Dict[int, str]]:
            return texts, {**skipped, **{t.page: reason for t in batch if t.page not in texts}}

        if run.abort.is_set():
            return skip_all(run.reason)
        try:
            texts = self._transcribe(batch, run)
        except _Abort as a:
            run.stop(str(a))
            return skip_all(str(a))
        except _BatchFailed as f:
            fail_reason = str(f)

        missing = [t for t in batch if t.page not in texts]
        if missing and len(batch) > 1:
            # A multi-page answer was truncated/garbled: retry the missing pages one by one.
            for t in missing:
                if run.abort.is_set():
                    skipped[t.page] = run.reason
                    continue
                try:
                    single = self._transcribe([t], run)
                except _Abort as a:
                    run.stop(str(a))
                    skipped[t.page] = str(a)
                    continue
                except _BatchFailed as f:
                    skipped[t.page] = str(f)
                    continue
                if t.page in single:
                    texts[t.page] = single[t.page]
                else:
                    skipped[t.page] = fail_reason
        else:
            for t in missing:
                skipped[t.page] = fail_reason
        return texts, skipped

    def _transcribe(self, batch: List[OCRTarget], run: _Run) -> Dict[int, str]:
        from google.genai import types

        contents: list = [_PROMPT]
        for t in batch:
            label = "FULL PAGE" if t.mode == "full" else "FIGURES ONLY"
            contents.append(f"Page {t.page} ({label}):")
            contents.append(types.Part.from_bytes(data=t.image, mime_type=t.mime))
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_OcrBatch,
            max_output_tokens=8192,
        )

        for attempt in range(self.max_retries + 1):
            if run.abort.is_set():
                raise _Abort(run.reason)
            # Count every attempt against the daily budget: failed calls can still consume quota.
            if self.budget is not None:
                try:
                    allowed = self.budget.acquire()
                except Exception:  # noqa: BLE001
                    logger.exception("OCR budget store failed")
                    raise _Abort("the OCR usage counter is unavailable")
                if not allowed:
                    raise _Abort("the daily OCR request budget is used up (resets at midnight Pacific)")
            self._limiter.wait()
            try:
                resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                kind, delay = classify_error(exc)
                logger.warning("OCR call failed (%s, attempt %d): %s", kind, attempt + 1, exc)
                if kind == "abort":
                    code = getattr(exc, "code", "")
                    reason = ("the Gemini daily quota is used up" if code == 429
                              else f"the Gemini API rejected the request ({code or type(exc).__name__})")
                    raise _Abort(reason) from exc
                if kind == "retry" and attempt < self.max_retries:
                    self._sleep(min(delay if delay is not None else 5.0 * 2 ** attempt, _MAX_RETRY_WAIT))
                    continue
                raise _BatchFailed("the Gemini request failed") from exc
            return self._parse(resp, batch)
        raise _BatchFailed("the Gemini request kept failing")

    @staticmethod
    def _parse(resp, batch: List[OCRTarget]) -> Dict[int, str]:
        raw = getattr(resp, "text", None)
        if not raw:
            raise _BatchFailed("the model returned an empty or blocked response")
        try:
            data = json.loads(raw)
            items = data["pages"] if isinstance(data, dict) else data
            wanted = {t.page for t in batch}
            out: Dict[int, str] = {}
            for it in items:
                page = int(it["page"])
                if page in wanted:
                    out[page] = _strip_fence(str(it.get("markdown", "")))
            return out
        except (ValueError, KeyError, TypeError) as exc:
            raise _BatchFailed("the model's output could not be parsed") from exc
