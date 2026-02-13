"""Parse financial data from PDF accounts using Claude API.

Companies that file PDF accounts (mostly PLCs and larger companies) can't be
parsed with iXBRL tag extraction. Instead we send the PDF to Claude Haiku
which reads the document and extracts structured financial data.

For large annual reports (100+ pages), we extract text from financial pages
first (cheap text-only API call). Falls back to sending trimmed PDF as
document if text extraction fails.
"""

import os
import io
import json
import base64
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
MAX_PDF_PAGES = 30  # Aggressive trim — financials are only 4-6 pages

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


def extract_financials_from_pdf(pdf_bytes):
    """Send a PDF to Claude API and extract structured financial data.

    Strategy:
    1. Try pdfplumber for text extraction (handles complex layouts)
    2. If that fails, try pypdf
    3. If text found, send as cheap text-only API call
    4. If no text, send trimmed PDF as document (more expensive)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[PDF Parser] No ANTHROPIC_API_KEY set — skipping PDF parsing")
        return []

    # Try text extraction first (cheap path)
    extracted_text = _extract_financial_text(pdf_bytes)

    if extracted_text and len(extracted_text.strip()) >= 200:
        # Truncate if very long
        if len(extracted_text) > 50000:
            extracted_text = extracted_text[:50000]

        token_est = len(extracted_text) // 4
        print(f"[PDF Parser] Sending {len(extracted_text)} chars (~{token_est} tokens) of extracted text...")

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
            if result is not None and len(result) > 0:
                return result
            print("[PDF Parser] Text approach returned no data, trying document fallback...")
        except Exception as e:
            print(f"[PDF Parser] Text approach error: {e}")

    else:
        print("[PDF Parser] Could not extract meaningful text from PDF")

    # Fallback: send trimmed PDF as document
    return _parse_pdf_as_document(pdf_bytes, api_key)


def _extract_financial_text(pdf_bytes):
    """Extract text from pages containing financial statements.

    Tries pdfplumber first (better for complex layouts), falls back to pypdf.
    """
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

    # Try pdfplumber first
    page_texts = _extract_with_pdfplumber(pdf_bytes)

    # Fall back to pypdf
    if not page_texts:
        page_texts = _extract_with_pypdf(pdf_bytes)

    if not page_texts:
        return None

    total_pages = len(page_texts)
    print(f"[PDF Parser] Extracted text from {total_pages} pages, scanning for financial content...")

    # Score each page
    scored = []
    for i, text in enumerate(page_texts):
        text_lower = text.lower()
        score = sum(1 for kw in financial_keywords if kw in text_lower)
        scored.append((i, score, text))

    # Get pages with financial content (score >= 2)
    financial_pages = [(i, s, t) for i, s, t in scored if s >= 2]

    if financial_pages:
        # Include 1 page either side for context
        page_indices = set()
        for i, _, _ in financial_pages:
            page_indices.update([max(0, i - 1), i, min(total_pages - 1, i + 1)])
        page_indices = sorted(page_indices)

        # Limit to 40 pages
        if len(page_indices) > 40:
            top = sorted(financial_pages, key=lambda x: x[1], reverse=True)[:30]
            page_indices = sorted(set(i for i, _, _ in top))

        texts = [f"--- Page {i+1} ---\n{scored[i][2]}" for i in page_indices if scored[i][2].strip()]
        print(f"[PDF Parser] Found financial content on {len(financial_pages)} pages, using {len(texts)} pages")
        return "\n\n".join(texts)
    else:
        # No keywords — take last 20 pages
        start = max(0, total_pages - 20)
        texts = [f"--- Page {i+1} ---\n{scored[i][2]}" for i in range(start, total_pages) if scored[i][2].strip()]
        print(f"[PDF Parser] No keyword matches — using last {len(texts)} pages")
        combined = "\n\n".join(texts)
        return combined if len(combined.strip()) >= 200 else None


def _extract_with_pdfplumber(pdf_bytes):
    """Extract text using pdfplumber (better for complex/designed PDFs)."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                    pages.append(text)
                except Exception:
                    pages.append("")
        non_empty = sum(1 for p in pages if len(p.strip()) > 50)
        print(f"[PDF Parser] pdfplumber: {len(pages)} pages, {non_empty} with content")
        if non_empty > 0:
            return pages
        return None
    except ImportError:
        print("[PDF Parser] pdfplumber not available")
        return None
    except Exception as e:
        print(f"[PDF Parser] pdfplumber failed: {e}")
        return None


def _extract_with_pypdf(pdf_bytes):
    """Extract text using pypdf (fallback)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception:
                pages.append("")
        non_empty = sum(1 for p in pages if len(p.strip()) > 50)
        print(f"[PDF Parser] pypdf: {len(pages)} pages, {non_empty} with content")
        if non_empty > 0:
            return pages
        return None
    except ImportError:
        print("[PDF Parser] pypdf not available")
        return None
    except Exception as e:
        print(f"[PDF Parser] pypdf failed: {e}")
        return None


def _trim_pdf_to_financials(pdf_bytes):
    """Trim a large PDF to just the last N pages for document fallback."""
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)

        if total <= MAX_PDF_PAGES:
            return pdf_bytes

        # Take last MAX_PDF_PAGES pages
        writer = PdfWriter()
        start = total - MAX_PDF_PAGES
        for i in range(start, total):
            writer.add_page(reader.pages[i])

        output = io.BytesIO()
        writer.write(output)
        trimmed = output.getvalue()
        print(f"[PDF Parser] Trimmed from {total} to {MAX_PDF_PAGES} pages ({len(trimmed)} bytes)")
        return trimmed
    except Exception as e:
        print(f"[PDF Parser] Trim failed: {e}")
        return pdf_bytes


def _parse_pdf_as_document(pdf_bytes, api_key):
    """Fallback: send trimmed PDF as a document to Claude."""
    pdf_bytes = _trim_pdf_to_financials(pdf_bytes)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    size_mb = len(pdf_b64) / (1024 * 1024)
    if size_mb > 30:
        print(f"[PDF Parser] PDF too large ({size_mb:.1f}MB) even after trimming")
        return []

    print(f"[PDF Parser] Sending {size_mb:.1f}MB trimmed PDF as document (~{MAX_PDF_PAGES} pages)...")

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
    """Parse and validate Claude's response."""
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
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    print(f"[PDF Parser] Extracted {len(valid)} period(s). Tokens: {in_tok} in / {out_tok} out")
    return valid

