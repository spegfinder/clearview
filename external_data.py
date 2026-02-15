"""Clearview — External Data Sources

Integrates:
  1. The London Gazette — insolvency notices (winding up petitions, administration, CVLs)
  2. Contracts Finder — government contract awards and opportunities

Both are free, open data, no API key required.
"""

import requests
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
import time

# ── Rate limiting ──
_last_gazette = 0
_last_contracts = 0


def _gazette_rate_limit():
    """Rate limit gazette requests — 2 seconds between calls."""
    global _last_gazette
    elapsed = time.time() - _last_gazette
    if elapsed < 2:
        time.sleep(2 - elapsed)
    _last_gazette = time.time()


# ═══════════════════════════════════════════════════════════
#  THE GAZETTE — Insolvency Notices
# ═══════════════════════════════════════════════════════════

GAZETTE_NOTICE_TYPES = {
    "2450": "Winding-up petition",
    "2451": "Winding-up order",
    "2452": "Appointment of liquidator",
    "2410": "Administration order",
    "2411": "Appointment of administrator",
    "2421": "Administrative receiver appointed",
    "2432": "Voluntary arrangement",
    "2440": "Creditors' voluntary liquidation",
    "2441": "CVL meeting of creditors",
    "2443": "CVL appointment of liquidator",
    "2447": "CVL deemed consent",
    "2461": "Dismissal of winding-up petition",
    "2600": "Striking off",
}


def search_gazette(company_name, company_number=None):
    """Search The Gazette for insolvency notices about a company.

    Uses the Atom feed endpoint which is more reliable than JSON.
    Falls back to HTML scraping if feed fails.

    Returns: list of {type, title, date, url, notice_code, severity}
    """
    notices = []

    # Clean company name for search
    clean_name = company_name.strip()
    # Remove common suffixes for better matching
    for suffix in [" LIMITED", " LTD", " PLC", " LLP", " CIC"]:
        if clean_name.upper().endswith(suffix):
            clean_name = clean_name[:len(clean_name)-len(suffix)].strip()

    if len(clean_name) < 3:
        return notices

    try:
        _gazette_rate_limit()

        # Try JSON endpoint first (most reliable for parsing)
        url = "https://www.thegazette.co.uk/all-notices/notice/data.json"
        params = {
            "text": f'"{clean_name}"',
            "categorycode": "G206000000",  # Corporate insolvency
            "results-page-size": 10,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; Clearview/1.0)",
        }

        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            try:
                data = resp.json()
                notices = _parse_json_feed(data, company_name, company_number)
                if notices:
                    print(f"[Gazette] Found {len(notices)} notices for {company_name}")
                    return notices
            except (ValueError, TypeError):
                pass

        # Fallback: Atom feed
        if resp.status_code != 200:
            url = "https://www.thegazette.co.uk/all-notices/notice/data.feed"
            headers["Accept"] = "application/atom+xml"
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            if resp.status_code == 200 and len(resp.text) > 100:
                notices = _parse_atom_feed(resp.text, company_name, company_number)
                if notices:
                    print(f"[Gazette] Found {len(notices)} notices (Atom) for {company_name}")
                    return notices

        if resp.status_code == 403:
            print(f"[Gazette] 403 Forbidden — API may require different headers")
        elif resp.status_code != 200:
            print(f"[Gazette] HTTP {resp.status_code}")

    except Exception as e:
        print(f"[Gazette] Error searching for {company_name}: {e}")

    return notices


def _parse_atom_feed(xml_text, company_name, company_number):
    """Parse Atom XML feed from Gazette."""
    notices = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            published_el = entry.find("atom:published", ns)
            updated_el = entry.find("atom:updated", ns)
            link_el = entry.find("atom:link", ns)
            content_el = entry.find("atom:content", ns)

            title = title_el.text if title_el is not None and title_el.text else ""
            pub_date = (published_el.text if published_el is not None else
                       updated_el.text if updated_el is not None else "")
            link = link_el.get("href", "") if link_el is not None else ""
            content = content_el.text if content_el is not None else ""

            # Try to match company number in content
            if company_number and company_number.upper() not in (content + title).upper():
                # Looser match: check company name
                if company_name.upper()[:20] not in (content + title).upper():
                    continue

            notice = _classify_notice(title, pub_date[:10], link)
            notices.append(notice)

    except ET.ParseError:
        pass
    return notices[:10]


def _parse_json_feed(data, company_name, company_number):
    """Parse JSON feed from Gazette."""
    notices = []
    entries = data.get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]

    for entry in entries:
        title = entry.get("title", "")
        if isinstance(title, dict):
            title = title.get("#text", str(title))
        pub_date = str(entry.get("published", entry.get("updated", "")))[:10]
        link = ""
        links = entry.get("link", [])
        if isinstance(links, dict):
            links = [links]
        for l in links:
            if isinstance(l, dict) and l.get("@rel") == "alternate":
                link = l.get("@href", "")
                break

        notice = _classify_notice(str(title), pub_date, link)
        notices.append(notice)

    return notices[:10]


def _parse_html_results(html, company_name, company_number):
    """Parse HTML search results as last resort."""
    notices = []
    # Simple regex extraction from Gazette HTML results
    # Look for notice titles and dates
    notice_pattern = re.compile(
        r'class="[^"]*notice-title[^"]*"[^>]*>([^<]+)</[^>]+>.*?'
        r'(\d{1,2}\s+\w+\s+\d{4})',
        re.DOTALL | re.IGNORECASE
    )
    for match in notice_pattern.finditer(html):
        title = match.group(1).strip()
        date = match.group(2).strip()
        notice = _classify_notice(title, date, "")
        notices.append(notice)

    return notices[:10]


def _classify_notice(title, date, url):
    """Classify a gazette notice by severity."""
    title_lower = title.lower()

    # Determine notice type and severity
    if "winding" in title_lower and "petition" in title_lower:
        severity = "critical"
        notice_type = "Winding-up petition"
    elif "winding" in title_lower and "order" in title_lower:
        severity = "critical"
        notice_type = "Winding-up order"
    elif "administration" in title_lower:
        severity = "critical"
        notice_type = "Administration"
    elif "liquidat" in title_lower:
        severity = "critical"
        notice_type = "Liquidation"
    elif "receiver" in title_lower:
        severity = "critical"
        notice_type = "Receivership"
    elif "voluntary arrangement" in title_lower:
        severity = "high"
        notice_type = "Voluntary arrangement"
    elif "striking off" in title_lower or "strike off" in title_lower:
        severity = "high"
        notice_type = "Striking off"
    elif "dismissal" in title_lower:
        severity = "positive"
        notice_type = "Petition dismissed"
    else:
        severity = "warning"
        notice_type = "Insolvency notice"

    return {
        "type": notice_type,
        "title": title[:200],
        "date": date,
        "url": url,
        "severity": severity,
    }


# ═══════════════════════════════════════════════════════════
#  CONTRACTS FINDER — Government Contract Awards
# ═══════════════════════════════════════════════════════════

def search_contracts(company_name, company_number=None):
    """Search Contracts Finder for government contracts awarded to a company.

    Uses the free Contracts Finder API (no auth needed).
    Returns: {active: [...], total_value: float, total_contracts: int, latest: str}
    """
    global _last_contracts
    results = {
        "contracts": [],
        "total_value": 0,
        "total_contracts": 0,
        "latest_award": None,
        "earliest_award": None,
        "buyers": [],
    }

    # Clean name
    clean_name = company_name.strip()
    for suffix in [" LIMITED", " LTD", " PLC", " LLP", " CIC"]:
        if clean_name.upper().endswith(suffix):
            clean_name = clean_name[:len(clean_name)-len(suffix)].strip()

    if len(clean_name) < 3:
        return results

    try:
        # Rate limit
        elapsed = time.time() - _last_contracts
        if elapsed < 1:
            time.sleep(1 - elapsed)
        _last_contracts = time.time()

        # Search awarded contracts
        url = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"
        params = {
            "supplier": clean_name,
            "stages": "award",
            "size": 20,
            "publishedFrom": (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d"),
        }

        resp = requests.get(url, params=params, timeout=15, headers={
            "Accept": "application/json",
            "User-Agent": "Clearview/1.0",
        })

        if resp.status_code == 200:
            data = resp.json()
            releases = data.get("releases", [])
            buyers = set()

            for release in releases[:20]:
                try:
                    awards = release.get("awards", [])
                    tender = release.get("tender", {})
                    buyer = release.get("buyer", {})
                    buyer_name = buyer.get("name", "Unknown")

                    for award in awards:
                        suppliers = award.get("suppliers", [])
                        # Check if this company is actually a supplier
                        matched = False
                        for s in suppliers:
                            s_name = s.get("name", "").upper()
                            if (clean_name.upper() in s_name or
                                s_name in clean_name.upper() or
                                (company_number and
                                 s.get("id", "").endswith(company_number))):
                                matched = True
                                break

                        if not matched and suppliers:
                            continue

                        value = award.get("value", {})
                        amount = value.get("amount", 0) or 0
                        currency = value.get("currency", "GBP")
                        award_date = award.get("date", "")[:10]
                        title = tender.get("title", release.get("tag", [""])[0] if release.get("tag") else "")

                        contract = {
                            "title": str(title)[:200],
                            "buyer": buyer_name,
                            "value": amount,
                            "currency": currency,
                            "date": award_date,
                            "status": award.get("status", "active"),
                        }

                        results["contracts"].append(contract)
                        results["total_value"] += amount
                        results["total_contracts"] += 1
                        buyers.add(buyer_name)

                        if award_date:
                            if not results["latest_award"] or award_date > results["latest_award"]:
                                results["latest_award"] = award_date
                            if not results["earliest_award"] or award_date < results["earliest_award"]:
                                results["earliest_award"] = award_date

                except (KeyError, TypeError, ValueError) as e:
                    continue

            results["buyers"] = list(buyers)[:10]

            # Sort contracts by date (newest first)
            results["contracts"].sort(key=lambda c: c.get("date", ""), reverse=True)
            results["contracts"] = results["contracts"][:10]

            if results["total_contracts"] > 0:
                print(f"[Contracts] Found {results['total_contracts']} government contracts for {company_name} "
                      f"(£{results['total_value']:,.0f})")

    except Exception as e:
        print(f"[Contracts] Error searching for {company_name}: {e}")

    return results


# ═══════════════════════════════════════════════════════════
#  Combined external data fetch
# ═══════════════════════════════════════════════════════════

def fetch_external_data(company_name, company_number=None):
    """Fetch all external data sources for a company.

    Returns: {gazette: [...], contracts: {...}}
    """
    gazette = []
    contracts = {"contracts": [], "total_value": 0, "total_contracts": 0}

    try:
        gazette = search_gazette(company_name, company_number)
    except Exception as e:
        print(f"[External] Gazette failed: {e}")

    try:
        contracts = search_contracts(company_name, company_number)
    except Exception as e:
        print(f"[External] Contracts failed: {e}")

    return {
        "gazette_notices": gazette,
        "government_contracts": contracts,
    }
