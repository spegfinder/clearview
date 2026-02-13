"""Parse financial data from PDF accounts using Claude API.

Companies that file PDF accounts (mostly PLCs and larger companies) can't be
parsed with iXBRL tag extraction. Instead we send the PDF to Claude Haiku
which reads the document and extracts structured financial data.

For large annual reports (100+ pages), we trim to the financial statements
section before sending, since financials are typically in the latter portion.
"""

import os
import io
import json
import base64
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
MAX_PDF_PAGES = 90  # Stay under API's 100-page limit

EXTRACTION_PROMPT = """You are a financial data extraction tool. Extract the following financial fields from this UK company accounts document. Return ONLY a JSON array (no other text, no markdown, no explanation).

Each element in the array should represent one financial year found in the document. Most accounts contain the current year and a comparative prior year — extract both.

For each year, return this exact structure:
{
  "year": "2024",
  "period_end": "2024-12-31",
  "turnover": null,
  "cost_of_sales": null,
  "gross_profit": null,
  "ebit": null,
  "net_profit": null,
  "total_assets": null,
  "current_assets": null,
  "fixed_assets": null,
  "total_liabilities": null,
  "current_liabilities": null,
  "non_current_liabilities": null,
  "net_assets": null,
  "cash": null,
  "retained_earnings": null,
  "employees": null,
  "share_capital": null,
  "dividends_paid": null
}

Rules:
- All monetary values in GBP as integers (no decimals, no thousands separators). If stated in £'000, multiply by 1000. If stated in £m, multiply by 1000000.
- Use null for any field you cannot find.
- "year" = the calendar year the period ends in (e.g. if period ends 31 March 2024, year = "2024")
- "period_end" = the exact date the accounting period ends (YYYY-MM-DD)
- "ebit" = operating profit/loss (before interest and tax)
- "net_profit" = profit/loss for the financial year (after tax)
- "dividends_paid" = dividends paid or proposed during the year (as a positive number)
- "employees" = average number of employees (integer)
- "retained_earnings" = profit and loss account reserve / retained earnings from balance sheet
- Return newest year first in the array.
- If the document is unreadable or contains no financial data, return an empty array: []

Return ONLY the JSON array. No commentary."""


def _trim_pdf_to_financials(pdf_bytes):
    """Trim a large PDF to just the pages likely containing financial statements.

    Strategy:
    1. Search all pages for financial keywords (balance sheet, profit/loss, etc.)
    2. If found, extract a window around those pages
    3. If not found, take the last N pages (financials are at the back of annual reports)

    Returns: trimmed PDF bytes
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("[PDF Parser] pypdf not available — sending full PDF")
        return pdf_bytes

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        if total_pages <= MAX_PDF_PAGES:
            return pdf_bytes  # No trimming needed

        print(f"[PDF Parser] PDF has {total_pages} pages — trimming to find financials...")

        # Search for pages containing financial statement keywords
        financial_keywords = [
            "balance sheet", "statement of financial position",
            "profit and loss", "income statement", "statement of comprehensive income",
            "cash flow", "statement of cash flows",
            "total assets", "net assets", "shareholders' funds",
            "retained earnings", "called up share capital",
        ]

        financial_pages = set()
        for i, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").lower()
                for kw in financial_keywords:
                    if kw in text:
                        financial_pages.add(i)
                        break
            except Exception:
                continue

        if financial_pages:
            # Found financial pages — take a window around them
            min_page = max(0, min(financial_pages) - 2)
            max_page = min(total_pages - 1, max(financial_pages) + 5)
            # Ensure we don't exceed the limit
            if max_page - min_page + 1 > MAX_PDF_PAGES:
                max_page = min_page + MAX_PDF_PAGES - 1
            page_range = range(min_page, max_page + 1)
            print(f"[PDF Parser] Found financial content on {len(financial_pages)} pages, extracting pages {min_page+1}-{max_page+1}")
        else:
            # No keywords found — take the last N pages (financials are at the back)
            start = max(0, total_pages - MAX_PDF_PAGES)
            page_range = range(start, total_pages)
            print(f"[PDF Parser] No keywords found, taking last {len(page_range)} pages")

        writer = PdfWriter()
        for i in page_range:
            writer.add_page(reader.pages[i])

        output = io.BytesIO()
        writer.write(output)
        trimmed = output.getvalue()
        print(f"[PDF Parser] Trimmed from {len(pdf_bytes)} to {len(trimmed)} bytes ({len(page_range)} pages)")
        return trimmed

    except Exception as e:
        print(f"[PDF Parser] Trim failed ({e}), trying last {MAX_PDF_PAGES} pages...")
        # Last resort: brute force last N pages
        try:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(io.BytesIO(pdf_bytes))
            writer = PdfWriter()
            start = max(0, len(reader.pages) - MAX_PDF_PAGES)
            for i in range(start, len(reader.pages)):
                writer.add_page(reader.pages[i])
            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()
        except Exception:
            return pdf_bytes


def extract_financials_from_pdf(pdf_bytes):
    """Send a PDF to Claude API and extract structured financial data.

    Strategy: extract text from financial pages with pypdf first, then send
    the text to Claude. This is ~10x cheaper than sending the PDF as a document
    since document pages are processed as images.

    Args:
        pdf_bytes: Raw bytes of the PDF document

    Returns:
        List of financial year records (same format as iXBRL parser), or empty list on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[PDF Parser] No ANTHROPIC_API_KEY set — skipping PDF parsing")
        return []

    # Extract text from financial pages
    extracted_text = _extract_financial_text(pdf_bytes)

    if not extracted_text or len(extracted_text.strip()) < 200:
        print("[PDF Parser] Could not extract meaningful text from PDF")
        # Fall back to sending PDF as document (more expensive but handles scanned docs)
        return _parse_pdf_as_document(pdf_bytes, api_key)

    # Truncate if extremely long (shouldn't happen with targeted extraction)
    if len(extracted_text) > 50000:
        extracted_text = extracted_text[:50000]

    token_est = len(extracted_text) // 4
    print(f"[PDF Parser] Sending {len(extracted_text)} chars (~{token_est} tokens) of extracted text to Claude...")

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Here is the text extracted from a UK company's annual accounts PDF. Extract the financial data.\n\n---\n\n" + extracted_text + "\n\n---\n\n" + EXTRACTION_PROMPT,
                    }
                ],
            },
            timeout=60,
        )

        result = _handle_api_response(resp)
        if result is not None:
            return result

        # If text approach failed, try PDF document approach as fallback
        print("[PDF Parser] Text extraction approach failed, falling back to PDF document...")
        return _parse_pdf_as_document(pdf_bytes, api_key)

    except requests.exceptions.Timeout:
        print("[PDF Parser] API request timed out (60s)")
        return []
    except Exception as e:
        print(f"[PDF Parser] Unexpected error: {e}")
        return []


def _extract_financial_text(pdf_bytes):
    """Extract text from pages containing financial statements."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[PDF Parser] pypdf not available")
        return None

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        print(f"[PDF Parser] PDF has {total_pages} pages, scanning for financial content...")

        financial_keywords = [
            "balance sheet", "statement of financial position",
            "profit and loss", "income statement", "statement of comprehensive income",
            "cash flow", "statement of cash flows",
            "total assets", "net assets", "shareholders' funds", "shareholders' equity",
            "retained earnings", "called up share capital",
            "current liabilities", "non-current liabilities", "current assets",
            "trade and other receivables", "trade and other payables",
            "revenue", "turnover", "cost of sales", "gross profit",
            "operating profit", "profit before tax", "profit for the year",
            "dividends paid", "dividends per share",
        ]

        # Score each page by how many financial keywords it contains
        page_scores = []
        for i, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "")
                text_lower = text.lower()
                score = sum(1 for kw in financial_keywords if kw in text_lower)
                page_scores.append((i, score, text))
            except Exception:
                page_scores.append((i, 0, ""))

        # Get pages with financial content (score > 0), sorted by score
        financial_pages = [(i, score, text) for i, score, text in page_scores if score >= 2]
        financial_pages.sort(key=lambda x: x[0])  # Keep in page order

        if financial_pages:
            # Also include 1 page before and after each financial page for context
            page_indices = set()
            for i, _, _ in financial_pages:
                page_indices.update([max(0, i-1), i, min(total_pages-1, i+1)])
            page_indices = sorted(page_indices)

            # Limit to 40 pages max
            if len(page_indices) > 40:
                # Take pages with highest scores
                top_pages = sorted(financial_pages, key=lambda x: x[1], reverse=True)[:30]
                page_indices = sorted(set(i for i, _, _ in top_pages))

            texts = []
            for i in page_indices:
                texts.append(f"--- Page {i+1} ---\n{page_scores[i][2]}")

            combined = "\n\n".join(texts)
            print(f"[PDF Parser] Extracted text from {len(page_indices)} financial pages (of {total_pages} total)")
            return combined
        else:
            # No financial pages found — try last 20 pages
            start = max(0, total_pages - 20)
            texts = []
            for i in range(start, total_pages):
                text = page_scores[i][2] if page_scores[i][2] else ""
                if text.strip():
                    texts.append(f"--- Page {i+1} ---\n{text}")
            print(f"[PDF Parser] No keyword matches — using last {total_pages - start} pages")
            return "\n\n".join(texts)

    except Exception as e:
        print(f"[PDF Parser] Text extraction failed: {e}")
        return None


def _parse_pdf_as_document(pdf_bytes, api_key):
    """Fallback: send trimmed PDF as a document to Claude (more expensive)."""
    pdf_bytes = _trim_pdf_to_financials(pdf_bytes)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    size_mb = len(pdf_b64) / (1024 * 1024)
    if size_mb > 30:
        print(f"[PDF Parser] PDF too large ({size_mb:.1f}MB) even after trimming")
        return []

    print(f"[PDF Parser] Fallback: sending {size_mb:.1f}MB PDF as document...")

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            },
            timeout=90,
        )

        result = _handle_api_response(resp)
        return result if result is not None else []

    except Exception as e:
        print(f"[PDF Parser] Document fallback error: {e}")
        return []


def _handle_api_response(resp):
    """Parse and validate Claude's response. Returns list of records or None on failure."""
    if resp.status_code != 200:
        print(f"[PDF Parser] API error: HTTP {resp.status_code}")
        try:
            err = resp.json()
            print(f"[PDF Parser] Error detail: {err.get('error', {}).get('message', 'unknown')}")
        except Exception:
            pass
        return None

    data = resp.json()

    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")

    # Parse JSON
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        records = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[PDF Parser] JSON parse failed: {e}")
        return None

    if not isinstance(records, list):
        print(f"[PDF Parser] Expected list, got {type(records)}")
        return None

    # Validate records
    valid = []
    for r in records:
        if not isinstance(r, dict) or not r.get("year"):
            continue

        for field in ["turnover", "cost_of_sales", "gross_profit", "ebit",
                      "net_profit", "total_assets", "current_assets", "fixed_assets",
                      "total_liabilities", "current_liabilities", "non_current_liabilities",
                      "net_assets", "cash", "retained_earnings", "share_capital",
                      "dividends_paid"]:
            val = r.get(field)
            if val is not None:
                try:
                    r[field] = int(round(float(val)))
                except (ValueError, TypeError):
                    r[field] = None

        if r.get("employees") is not None:
            try:
                r["employees"] = int(r["employees"])
            except (ValueError, TypeError):
                r["employees"] = None

        has_data = any(r.get(k) is not None for k in [
            "total_assets", "net_assets", "current_assets", "turnover"
        ])
        if has_data:
            valid.append(r)

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    print(f"[PDF Parser] Extracted {len(valid)} period(s). Tokens: {input_tokens} in / {output_tokens} out")

    return valid
