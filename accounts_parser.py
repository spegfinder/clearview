"""Parse iXBRL (inline XBRL) accounts documents from Companies House.

iXBRL files are HTML documents with embedded XBRL tags. Financial values appear as:
  <ix:nonFraction name="ns:ConceptName" contextRef="ctx1" ...>1,234</ix:nonFraction>

Contexts define the reporting period and entity:
  <xbrli:context id="ctx1">
    <xbrli:period>
      <xbrli:startDate>2023-01-01</xbrli:startDate>
      <xbrli:endDate>2023-12-31</xbrli:endDate>
    </xbrli:period>
  </xbrli:context>

We extract values, match them to the correct period, and build a financial summary.
"""

from bs4 import BeautifulSoup
import re
from datetime import datetime


# ── XBRL concept name mappings ──
# Maps from our internal field names to lists of possible XBRL concept suffixes.
# We match against the local name (after the namespace prefix).
CONCEPT_MAP = {
    "turnover": [
        "Turnover", "TurnoverRevenue", "Revenue",
        "TurnoverGrossIncome", "RevenueFromContractsWithCustomers",
    ],
    "cost_of_sales": [
        "CostSales", "CostOfSales",
    ],
    "gross_profit": [
        "GrossProfitLoss", "GrossProfit",
    ],
    "ebit": [
        "OperatingProfitLoss", "OperatingProfit",
        "ProfitLossOnOrdinaryActivitiesBeforeInterestAndTax",
        "ProfitLossBeforeInterestPayableSimilarCharges",
        "ProfitLossBeforeTax",  # not exactly EBIT but close for small companies
    ],
    "net_profit": [
        "ProfitLossForPeriod", "ProfitLossForYear",
        "ProfitLossForFinancialYear",
        "RetainedProfitLossForFinancialYear",
        "ProfitLoss", "ProfitLossAttributableToOwnersOfParent",
    ],
    "total_assets": [
        "TotalAssets", "TotalAssetsLessCurrentLiabilities",
    ],
    "fixed_assets": [
        "FixedAssets", "NonCurrentAssets", "TangibleFixedAssets",
        "IntangibleFixedAssets",
    ],
    "current_assets": [
        "CurrentAssets", "TotalCurrentAssets",
    ],
    "total_liabilities": [
        "TotalLiabilities",
    ],
    "current_liabilities": [
        "CreditorsDueWithinOneYear", "CurrentLiabilities",
        "CreditorAmountsFallingDueWithinOneYear",
    ],
    "non_current_liabilities": [
        "CreditorsDueAfterOneYear", "NonCurrentLiabilities",
        "CreditorsAmountsFallingDueAfterMoreThanOneYear",
        "CreditorAmountsFallingDueAfterOneYear",
    ],
    "net_assets": [
        "NetAssetsLiabilities", "NetAssets",
        "TotalNetAssets", "NetAssetsIncludingPensionAssetLiability",
    ],
    "cash": [
        "CashBankInHand", "CashCashEquivalents",
        "CashAtBankInHand", "CashBankOnHand",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedLosses",
        "ProfitLossAccountReserve",
        "RetainedEarnings",
    ],
    "employees": [
        "AverageNumberEmployeesDuringPeriod",
        "EntityAverageNumberOfEmployees",
        "AverageNumberOfEmployees",
        "EmployeesTotal",
    ],
    "creditors_due_within_year": [
        "CreditorsDueWithinOneYear",
        "CreditorAmountsFallingDueWithinOneYear",
        "CurrentLiabilities",
    ],
    "share_capital": [
        "CalledUpShareCapital", "ShareCapital",
        "CalledUpShareCapitalNotPaid",
    ],
}


def parse_ixbrl(content, encoding="utf-8"):
    """Parse an iXBRL document and extract all tagged financial values.

    Returns:
        {
            "contexts": { ctx_id: { "start": date, "end": date, "instant": date } },
            "facts": [ { "concept": str, "context_ref": str, "value": float, "sign": str, "decimals": str } ],
        }
    """
    if isinstance(content, bytes):
        # Try to detect encoding from XML declaration
        try:
            content = content.decode(encoding)
        except UnicodeDecodeError:
            content = content.decode("latin-1")

    soup = BeautifulSoup(content, "html.parser")

    # ── Parse contexts ──
    contexts = {}
    for ctx in soup.find_all(re.compile(r"(xbrli:)?context", re.I)):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue

        period = ctx.find(re.compile(r"(xbrli:)?period", re.I))
        if not period:
            continue

        info = {}
        start = period.find(re.compile(r"(xbrli:)?startdate", re.I))
        end = period.find(re.compile(r"(xbrli:)?enddate", re.I))
        instant = period.find(re.compile(r"(xbrli:)?instant", re.I))

        if start and end:
            info["start"] = start.get_text(strip=True)
            info["end"] = end.get_text(strip=True)
        elif instant:
            info["instant"] = instant.get_text(strip=True)

        # Check for dimensions (segments) — helps identify consolidated vs entity
        segment = ctx.find(re.compile(r"(xbrli:)?segment", re.I))
        if segment:
            info["has_dimension"] = True

        contexts[ctx_id] = info

    # ── Parse facts (ix:nonFraction and ix:nonNumeric for some fields) ──
    facts = []

    for tag in soup.find_all(re.compile(r"ix:(nonfraction|nonnumeric)", re.I)):
        name = tag.get("name", "")
        ctx_ref = tag.get("contextref", tag.get("contextRef", ""))
        sign = tag.get("sign", "")
        decimals = tag.get("decimals", "")
        scale_str = tag.get("scale", "0")
        fmt_tag = tag.get("format", "")
        unit = tag.get("unitref", tag.get("unitRef", ""))

        # Get the text value
        text = tag.get_text(strip=True)
        if not text or not name:
            continue

        # Get the local concept name (after namespace prefix)
        local_name = name.split(":")[-1] if ":" in name else name

        # Parse numeric value
        value = _parse_value(text, scale_str, sign, fmt_tag)

        if value is not None:
            facts.append({
                "concept": local_name,
                "full_name": name,
                "context_ref": ctx_ref,
                "value": value,
                "sign": sign,
                "decimals": decimals,
                "unit": unit,
            })

    return {"contexts": contexts, "facts": facts}


def _parse_value(text, scale_str="0", sign="", fmt=""):
    """Parse a numeric value from iXBRL text content."""
    # Remove common formatting
    cleaned = text.replace(",", "").replace(" ", "").replace("\xa0", "")
    cleaned = cleaned.replace("(", "").replace(")", "")  # brackets = negative in accounting
    has_brackets = "(" in text and ")" in text

    # Handle ixt:numdotdecimal and similar formats
    cleaned = re.sub(r"[^\d.\-]", "", cleaned)

    if not cleaned or cleaned == "-" or cleaned == ".":
        return None

    try:
        value = float(cleaned)
    except ValueError:
        return None

    # Apply scale (e.g., scale="3" means value is in thousands)
    try:
        scale = int(scale_str) if scale_str else 0
        value *= 10 ** scale
    except (ValueError, TypeError):
        pass

    # Apply sign
    if sign == "-" or has_brackets:
        value = -abs(value)

    return value


def _match_concept(local_name, field):
    """Check if a local XBRL concept name matches any of our target concepts."""
    targets = CONCEPT_MAP.get(field, [])
    # Case-insensitive match against the end of the concept name
    ln = local_name.lower()
    for t in targets:
        if ln == t.lower() or ln.endswith(t.lower()):
            return True
    return False


def extract_financials_from_ixbrl(content):
    """Extract structured financial data from an iXBRL document.

    Returns a list of financial year records:
    [
        {
            "year": "2023",
            "period_end": "2023-12-31",
            "turnover": 1234567 | None,
            "cost_of_sales": ... | None,
            ...
        }
    ]
    """
    parsed = parse_ixbrl(content)
    contexts = parsed["contexts"]
    facts = parsed["facts"]

    if not facts:
        return []

    # ── Group contexts by period end date ──
    # Duration contexts (start/end) for P&L items
    # Instant contexts for balance sheet items
    periods = {}  # period_end_date -> { "duration_ctx_ids": [...], "instant_ctx_ids": [...] }

    for ctx_id, info in contexts.items():
        if info.get("has_dimension"):
            continue  # skip dimensioned contexts (e.g. segment breakdowns)

        if "end" in info:
            end = info["end"]
            if end not in periods:
                periods[end] = {"duration": [], "instant": []}
            periods[end]["duration"].append(ctx_id)
        elif "instant" in info:
            inst = info["instant"]
            if inst not in periods:
                periods[inst] = {"duration": [], "instant": []}
            periods[inst]["instant"].append(ctx_id)

    # ── For each period, extract values ──
    results = []

    for period_end, ctx_groups in sorted(periods.items(), reverse=True):
        all_ctx_ids = set(ctx_groups["duration"] + ctx_groups["instant"])
        if not all_ctx_ids:
            continue

        record = {
            "period_end": period_end,
            "year": period_end[:4],
        }

        fields_to_extract = [
            "turnover", "cost_of_sales", "gross_profit", "ebit", "net_profit",
            "total_assets", "current_assets", "fixed_assets",
            "total_liabilities", "current_liabilities", "non_current_liabilities",
            "net_assets", "cash", "retained_earnings", "employees",
            "creditors_due_within_year", "share_capital",
        ]

        for field in fields_to_extract:
            value = None
            # P&L fields should come from duration contexts; BS from instant
            if field in ("turnover", "cost_of_sales", "gross_profit", "ebit", "net_profit", "employees"):
                relevant_ctx = ctx_groups["duration"]
            else:
                relevant_ctx = ctx_groups["instant"]

            # Fall back to all contexts if nothing found
            for ctx_set in [relevant_ctx, list(all_ctx_ids)]:
                if value is not None:
                    break
                for fact in facts:
                    if fact["context_ref"] in ctx_set and _match_concept(fact["concept"], field):
                        value = fact["value"]
                        break

            record[field] = value

        # ── Derived fields ──
        # Total liabilities from parts if not directly available
        if record["total_liabilities"] is None:
            cl = record.get("current_liabilities") or 0
            ncl = record.get("non_current_liabilities") or 0
            if cl or ncl:
                record["total_liabilities"] = cl + ncl

        # Total assets from parts
        if record["total_assets"] is None:
            fa = record.get("fixed_assets") or 0
            ca = record.get("current_assets") or 0
            if fa or ca:
                record["total_assets"] = fa + ca

        # creditors_due_within_year fallback to current_liabilities
        if record["creditors_due_within_year"] is None:
            record["creditors_due_within_year"] = record.get("current_liabilities")

        # Only include records that have at least some balance sheet data
        has_data = any(record.get(k) is not None for k in [
            "total_assets", "net_assets", "current_assets", "turnover"
        ])

        if has_data:
            results.append(record)

    # Deduplicate by year (keep the record with more data)
    by_year = {}
    for r in results:
        yr = r["year"]
        if yr not in by_year:
            by_year[yr] = r
        else:
            # Keep the one with more non-None values
            existing_count = sum(1 for v in by_year[yr].values() if v is not None)
            new_count = sum(1 for v in r.values() if v is not None)
            if new_count > existing_count:
                by_year[yr] = r

    return sorted(by_year.values(), key=lambda x: x["year"], reverse=True)[:4]


def format_for_frontend(financials_list):
    """Convert parsed financials into the shape expected by the frontend."""
    result = []
    for f in financials_list:
        result.append({
            "year": f["year"],
            "period_end": f.get("period_end"),
            "turnover": f.get("turnover"),
            "cost_of_sales": f.get("cost_of_sales"),
            "gross_profit": f.get("gross_profit"),
            "ebit": f.get("ebit"),
            "net_profit": f.get("net_profit"),
            "total_assets": f.get("total_assets"),
            "current_assets": f.get("current_assets"),
            "total_liabilities": f.get("total_liabilities"),
            "current_liabilities": f.get("current_liabilities") or f.get("creditors_due_within_year"),
            "net_assets": f.get("net_assets"),
            "retained_earnings": f.get("retained_earnings"),
            "cash": f.get("cash"),
            "creditors_due_within_year": f.get("creditors_due_within_year") or f.get("current_liabilities"),
            "employees": int(f["employees"]) if f.get("employees") is not None else None,
        })
    return result
