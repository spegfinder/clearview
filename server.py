"""Clearview — Company financial health assessment.

Usage:
    python server.py

Then open http://localhost:5001 in your browser.
"""

import os
import json
import sys
import traceback
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from ch_api import CompaniesHouseClient, build_company_data
from accounts_parser import extract_financials_from_ixbrl, format_for_frontend

# ── Config ──
API_KEY = os.environ.get("CH_API_KEY", "ea83a39a-c244-403e-be11-f45e9d664965")
PORT = int(os.environ.get("PORT", 5001))

app = Flask(__name__, static_folder="static")
CORS(app)

client = CompaniesHouseClient(API_KEY)

# ── Simple cache to avoid re-fetching ──
_company_cache = {}


# ── SIC code descriptions (common ones) ──
SIC_DESCRIPTIONS = {}
# We'll populate this from the API responses themselves


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/search")
def search():
    """Search for companies by name or number."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"items": []})

    try:
        results = client.search(q, items_per_page=15)
        # Simplify for frontend
        items = []
        for r in results:
            addr = r.get("address", {})
            items.append({
                "company_name": r.get("title", ""),
                "company_number": r.get("company_number", ""),
                "company_status": r.get("company_status", ""),
                "type": r.get("company_type", ""),
                "date_of_creation": r.get("date_of_creation", ""),
                "address_snippet": r.get("address_snippet", ""),
                "sic_codes": r.get("sic_codes"),
                "locality": addr.get("locality", ""),
            })
        return jsonify({"items": items})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/company/<number>")
def company(number):
    """Get full company data including parsed financials."""
    number = number.strip().upper()

    # Check cache
    if number in _company_cache:
        return jsonify(_company_cache[number])

    try:
        print(f"[Clearview] Fetching data for {number}...")
        data = build_company_data(client, number)
        if not data:
            return jsonify({"error": "Company not found"}), 404

        # ── Attempt to parse financials from iXBRL filings ──
        financials = []
        filings = data.get("accounts_filings", [])

        print(f"[Clearview] Found {len(filings)} accounts filings, attempting iXBRL parse...")

        for filing in filings[:4]:  # Parse up to 4 years
            doc_meta_link = filing.get("links", {}).get("document_metadata")
            if not doc_meta_link:
                continue

            try:
                content, content_type = client.get_document_content(doc_meta_link)
                if content and "html" in (content_type or "").lower() or "xml" in (content_type or "").lower():
                    parsed = extract_financials_from_ixbrl(content)
                    if parsed:
                        financials.extend(parsed)
                        print(f"  ✓ Parsed {len(parsed)} period(s) from {filing.get('date', 'unknown')}")
                    else:
                        print(f"  ✗ No financial data extracted from {filing.get('date', 'unknown')}")
                elif content:
                    print(f"  ⊘ Document is {content_type}, skipping (PDF/image)")
                else:
                    print(f"  ✗ Could not download document for {filing.get('date', 'unknown')}")
            except Exception as e:
                print(f"  ✗ Error parsing {filing.get('date', 'unknown')}: {e}")
                continue

        # Deduplicate by year and format
        seen_years = set()
        unique_financials = []
        for f in financials:
            yr = f["year"]
            if yr not in seen_years:
                seen_years.add(yr)
                unique_financials.append(f)

        data["financials"] = format_for_frontend(
            sorted(unique_financials, key=lambda x: x["year"], reverse=True)[:4]
        )

        # Remove raw filings data before caching
        data.pop("accounts_filings", None)

        # Add SIC description
        sic_codes = data.get("sic_codes", [])
        data["sic_desc"] = get_sic_description(sic_codes[0]) if sic_codes else ""

        print(f"[Clearview] Done. {len(data['financials'])} years of financials extracted.")
        for f in data["financials"]:
            fields = [k for k, v in f.items() if v is not None and k not in ("year", "period_end")]
            print(f"  {f['year']}: {', '.join(fields)}")

        # Cache result
        _company_cache[number] = data

        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear")
def clear_cache():
    """Clear the company data cache."""
    _company_cache.clear()
    return jsonify({"status": "cleared"})


def get_sic_description(code):
    """Get a human-readable SIC code description."""
    # Common SIC codes - in production this would be a full lookup table
    SIC = {
        "01110": "Growing of cereals",
        "10710": "Manufacture of bread; manufacture of fresh pastry goods and cakes",
        "11050": "Manufacture of beer",
        "41100": "Development of building projects",
        "41201": "Construction of commercial buildings",
        "41202": "Construction of domestic buildings",
        "43320": "Joinery installation",
        "43999": "Other specialised construction activities",
        "45111": "Sale of new cars and light motor vehicles",
        "46900": "Non-specialised wholesale trade",
        "47110": "Retail sale in non-specialised stores with food",
        "47190": "Other retail sale in non-specialised stores",
        "55100": "Hotels and similar accommodation",
        "56101": "Licensed restaurants",
        "56102": "Unlicensed restaurants and cafes",
        "56301": "Licensed clubs",
        "56302": "Public houses and bars",
        "62011": "Ready-made interactive leisure and entertainment software development",
        "62012": "Business and domestic software development",
        "62020": "Information technology consultancy activities",
        "62090": "Other information technology service activities",
        "64110": "Central banking",
        "64191": "Banks",
        "64209": "Activities of other holding companies",
        "66220": "Activities of insurance agents and brokers",
        "68100": "Buying and selling of own real estate",
        "68201": "Renting and operating of Housing Association real estate",
        "68202": "Letting and operating of conference and exhibition centres",
        "68209": "Other letting and operating of own or leased real estate",
        "69102": "Tax consultancy",
        "69201": "Accounting and auditing activities",
        "70100": "Activities of head offices",
        "70229": "Management consultancy activities other than financial management",
        "73110": "Advertising agencies",
        "74100": "Specialised design activities",
        "82990": "Other business support service activities",
        "85100": "Pre-primary education",
        "85200": "Primary education",
        "86101": "Hospital activities",
        "86210": "General medical practice activities",
        "86220": "Specialist medical practice activities",
        "86230": "Dental practice activities",
        "86900": "Other human health activities",
        "87100": "Residential nursing care activities",
        "93110": "Operation of sports facilities",
        "93130": "Fitness facilities",
        "93290": "Other amusement and recreation activities",
        "96020": "Hairdressing and other beauty treatment",
    }
    return SIC.get(code, f"SIC {code}")


if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════╗
║           C L E A R V I E W  v0.3            ║
║     Company Financial Health Assessment      ║
╠══════════════════════════════════════════════╣
║  Server:  http://localhost:{PORT}              ║
║  API Key: {API_KEY[:8]}...{API_KEY[-4:]}                    ║
╚══════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=PORT, debug=True)
