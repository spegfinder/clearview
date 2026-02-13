"""Parse financial data from PDF accounts using Claude API.

Companies that file PDF accounts (mostly PLCs and larger companies) can't be
parsed with iXBRL tag extraction. Instead we send the PDF to Claude Haiku
which reads the document and extracts structured financial data.
"""

import os
import json
import base64
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

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

    Args:
        pdf_bytes: Raw bytes of the PDF document

    Returns:
        List of financial year records (same format as iXBRL parser), or empty list on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[PDF Parser] No ANTHROPIC_API_KEY set — skipping PDF parsing")
        return []

    # Encode PDF as base64
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Check size — Claude supports up to ~32MB base64
    size_mb = len(pdf_b64) / (1024 * 1024)
    if size_mb > 30:
        print(f"[PDF Parser] PDF too large ({size_mb:.1f}MB), skipping")
        return []

    print(f"[PDF Parser] Sending {size_mb:.1f}MB PDF to Claude {MODEL}...")

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
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"[PDF Parser] API error: HTTP {resp.status_code}")
            try:
                err = resp.json()
                print(f"[PDF Parser] Error detail: {err.get('error', {}).get('message', 'unknown')}")
            except Exception:
                pass
            return []

        data = resp.json()

        # Extract text from response
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON from response
        text = text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        records = json.loads(text)

        if not isinstance(records, list):
            print(f"[PDF Parser] Expected list, got {type(records)}")
            return []

        # Validate and clean records
        valid = []
        for r in records:
            if not isinstance(r, dict):
                continue
            if not r.get("year"):
                continue

            # Ensure numeric fields are ints or None
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

            # Employees
            if r.get("employees") is not None:
                try:
                    r["employees"] = int(r["employees"])
                except (ValueError, TypeError):
                    r["employees"] = None

            # Check it has at least some data
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

    except requests.exceptions.Timeout:
        print("[PDF Parser] API request timed out (60s)")
        return []
    except json.JSONDecodeError as e:
        print(f"[PDF Parser] Failed to parse JSON response: {e}")
        return []
    except Exception as e:
        print(f"[PDF Parser] Unexpected error: {e}")
        return []
