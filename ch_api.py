"""Companies House REST API client."""
import requests
import time
import base64
from functools import lru_cache

BASE = "https://api.companieshouse.gov.uk"
DOC_API = "https://document-api.companieshouse.gov.uk"
FRONTEND_DOC_API = "https://frontend-doc-api.company-information.service.gov.uk"
CONVERT_IXBRL_API = "https://convert-ixbrl.co.uk"


class CompaniesHouseClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update({"Accept": "application/json"})
        self._last_call = 0

        # Separate session for document API (needs different auth handling)
        self._doc_session = requests.Session()
        encoded_key = base64.b64encode(f"{api_key}:".encode()).decode()
        self._doc_session.headers.update({
            "Authorization": f"Basic {encoded_key}",
        })

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_call = time.time()

    def _get(self, url, **kwargs):
        self._rate_limit()
        resp = self.session.get(url, **kwargs)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            print("[CH API] Rate limited, waiting 5s...")
            time.sleep(5)
            return self._get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def search(self, query, items_per_page=20):
        data = self._get(f"{BASE}/search/companies", params={
            "q": query, "items_per_page": items_per_page
        })
        if not data:
            return []
        return data.get("items", [])

    def get_profile(self, number):
        return self._get(f"{BASE}/company/{number}")

    def get_officers(self, number, active_only=False):
        data = self._get(f"{BASE}/company/{number}/officers", params={
            "items_per_page": 50
        })
        if not data:
            return []
        items = data.get("items", [])
        if active_only:
            items = [o for o in items if not o.get("resigned_on")]
        return items

    def get_psc(self, number):
        data = self._get(f"{BASE}/company/{number}/persons-with-significant-control")
        if not data:
            return []
        return data.get("items", [])

    def get_charges(self, number):
        data = self._get(f"{BASE}/company/{number}/charges")
        if not data:
            return {"total_count": 0, "satisfied_count": 0, "part_satisfied_count": 0, "unfiltered_count": 0, "items": []}
        return data

    def get_insolvency(self, number):
        """Get insolvency cases from Companies House API (free endpoint)."""
        try:
            data = self._get(f"{BASE}/company/{number}/insolvency")
            if not data:
                return {"cases": [], "status": None}
            return data
        except Exception as e:
            print(f"[CH API] Insolvency lookup failed for {number}: {e}")
            return {"cases": [], "status": None}

    def get_gazette_notices(self, company_name, company_number):
        """Search The Gazette for insolvency notices about a company.
        Best-effort — fails silently if Gazette API is unavailable.
        Returns list of notice summaries.
        """
        notices = []
        try:
            # Try the JSON feed endpoint
            url = "https://www.thegazette.co.uk/all-notices/notice/data.json"
            params = {
                "text": f'"{company_name}"',
                "categorycode": "G206000000",  # Corporate insolvency
                "results-page-size": 5,
            }
            self._rate_limit()
            resp = requests.get(url, params=params, timeout=8, headers={
                "Accept": "application/json",
                "User-Agent": "Clearview/1.0"
            })
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    entries = data.get("entry", [])
                    if isinstance(entries, dict):
                        entries = [entries]
                    for r in entries[:5]:
                        title = r.get("title", "Insolvency notice")
                        if isinstance(title, dict):
                            title = title.get("#text", str(title))
                        pub_date = r.get("published", r.get("updated", ""))
                        notices.append({
                            "title": str(title)[:120],
                            "date": str(pub_date)[:10] if pub_date else "",
                            "url": "",
                        })
                except (ValueError, KeyError, TypeError):
                    pass
        except Exception as e:
            print(f"[Gazette] Search failed for {company_number}: {e}")
        return notices

    def get_filing_history(self, number, category=None, items_per_page=25):
        params = {"items_per_page": items_per_page}
        if category:
            params["category"] = category
        data = self._get(f"{BASE}/company/{number}/filing-history", params=params)
        if not data:
            return []
        return data.get("items", [])

    def get_accounts_filings(self, number, count=5):
        """Get the most recent accounts filings."""
        filings = self.get_filing_history(number, category="accounts", items_per_page=count)
        return filings

    def get_document_content(self, document_metadata_url):
        """Download iXBRL/XHTML or PDF document content.

        Filing history gives URLs on document-api.company-information.service.gov.uk
        but metadata must be fetched from frontend-doc-api.company-information.service.gov.uk

        Returns: (content_bytes, content_type_string) or (None, None)
        """
        # Rewrite the URL to use the correct domain for metadata
        meta_url = document_metadata_url
        meta_url = meta_url.replace(
            "https://document-api.company-information.service.gov.uk",
            "https://frontend-doc-api.company-information.service.gov.uk"
        )
        if meta_url.startswith("/"):
            meta_url = f"https://frontend-doc-api.company-information.service.gov.uk{meta_url}"

        print(f"    [doc] Fetching metadata: {meta_url[:90]}...")
        self._rate_limit()

        try:
            resp = self._doc_session.get(meta_url, headers={"Accept": "application/json"}, timeout=15)
            print(f"    [doc] Metadata response: HTTP {resp.status_code}")
            if resp.status_code != 200:
                return None, None
            meta = resp.json()
        except Exception as e:
            print(f"    [doc] Metadata error: {e}")
            return None, None

        resources = meta.get("resources", {})
        print(f"    [doc] Available formats: {list(resources.keys())}")

        # Content URL: use the self link + /content
        content_url = meta.get("links", {}).get("self", meta_url)
        if not content_url.startswith("http"):
            content_url = f"https://frontend-doc-api.company-information.service.gov.uk{content_url}"
        if not content_url.endswith("/content"):
            content_url = content_url.rstrip("/") + "/content"

        # Prefer iXBRL, fall back to PDF
        if "application/xhtml+xml" in resources:
            accept = "application/xhtml+xml"
        elif "application/pdf" in resources:
            accept = "application/pdf"
        else:
            print(f"    [doc] No iXBRL or PDF available")
            return None, None

        print(f"    [doc] Downloading ({accept}): {content_url[:80]}...")
        self._rate_limit()

        try:
            resp = self._doc_session.get(content_url, headers={
                "Accept": accept
            }, allow_redirects=True, timeout=30)

            if resp.status_code == 200:
                ct = resp.headers.get("Content-Type", "")
                print(f"    [doc] Downloaded {len(resp.content)} bytes ({ct})")
                # Return the actual content type from the response
                if "pdf" in ct.lower() or accept == "application/pdf":
                    return resp.content, "application/pdf"
                return resp.content, "application/xhtml+xml"
            else:
                print(f"    [doc] Download failed: HTTP {resp.status_code}")
                return None, None
        except Exception as e:
            print(f"    [doc] Download error: {e}")
            return None, None

    def get_financials_from_convert_ixbrl(self, number):
        """Fallback: fetch pre-parsed financials from convert-ixbrl.co.uk.

        This free API has already parsed all Companies House iXBRL filings.
        Returns parsed financial data or None.
        """
        url = f"{CONVERT_IXBRL_API}/api/company/{number}"
        print(f"  [convert-ixbrl] Trying fallback: {url}")

        try:
            resp = requests.get(url, timeout=15, headers={
                "Accept": "application/json"
            })
            if resp.status_code != 200:
                print(f"  [convert-ixbrl] HTTP {resp.status_code}")
                return None

            data = resp.json()
            accounts = data.get("accounts", [])
            if not accounts:
                print(f"  [convert-ixbrl] No accounts data")
                return None

            print(f"  [convert-ixbrl] Got {len(accounts)} period(s)")

            results = []
            for acc in accounts[:4]:
                bs = acc.get("balanceSheet", {})
                pl = acc.get("profitAndLoss", {})
                period = acc.get("period", {})

                record = {
                    "year": str(period.get("endDate", ""))[:4],
                    "period_end": period.get("endDate"),
                    "turnover": pl.get("turnover") or pl.get("revenue"),
                    "cost_of_sales": pl.get("costOfSales"),
                    "gross_profit": pl.get("grossProfitLoss") or pl.get("grossProfit"),
                    "ebit": pl.get("operatingProfitLoss") or pl.get("operatingProfit"),
                    "net_profit": pl.get("profitLossForPeriod") or pl.get("profitLossForYear"),
                    "total_assets": bs.get("totalAssets"),
                    "current_assets": bs.get("currentAssets"),
                    "total_liabilities": bs.get("totalLiabilities"),
                    "current_liabilities": bs.get("creditorsDueWithinOneYear") or bs.get("currentLiabilities"),
                    "net_assets": bs.get("netAssetsLiabilities") or bs.get("netAssets"),
                    "retained_earnings": bs.get("retainedEarningsAccumulatedLosses") or bs.get("profitLossAccountReserve"),
                    "cash": bs.get("cashBankInHand") or bs.get("cashCashEquivalents"),
                    "creditors_due_within_year": bs.get("creditorsDueWithinOneYear") or bs.get("currentLiabilities"),
                    "employees": acc.get("employees") or acc.get("averageNumberEmployees"),
                }

                # Only include if we have at least some data
                if any(v is not None for k, v in record.items() if k not in ("year", "period_end")):
                    results.append(record)

            return results if results else None

        except Exception as e:
            print(f"  [convert-ixbrl] Error: {e}")
            return None


def build_company_data(client, number):
    """Fetch all data for a company and structure it for the frontend."""
    profile = client.get_profile(number)
    if not profile:
        return None

    officers_raw = client.get_officers(number)
    psc_raw = client.get_psc(number)
    charges_raw = client.get_charges(number)
    accounts_filings = client.get_accounts_filings(number, count=5)
    insolvency_raw = client.get_insolvency(number)
    try:
        gazette_notices = client.get_gazette_notices(profile.get("company_name", ""), number)
    except Exception as e:
        print(f"[Gazette] Failed: {e}")
        gazette_notices = []

    # ── Profile ──
    addr = profile.get("registered_office_address", {})
    accounts = profile.get("accounts", {})
    last_acc = accounts.get("last_accounts", {})
    next_due = accounts.get("next_due")
    overdue = accounts.get("overdue", False)
    conf = profile.get("confirmation_statement", {})

    # ── Officers ──
    officers = []
    for o in officers_raw:
        officers.append({
            "name": o.get("name", "Unknown"),
            "role": o.get("officer_role", "director").replace("_", " ").title(),
            "appointed": o.get("appointed_on"),
            "resigned": o.get("resigned_on"),
            "nationality": o.get("nationality"),
            "dob": o.get("date_of_birth"),
        })

    # ── PSC ──
    psc = []
    for p in psc_raw:
        kind = p.get("kind", "")
        name = p.get("name") or p.get("name_elements", {}).get("surname", "")
        natures = p.get("natures_of_control", [])
        # Parse ownership from natures_of_control
        ownership = None
        for n in natures:
            if "75-to-100" in n:
                ownership = "75-100%"
            elif "50-to-75" in n:
                ownership = "50-75%"
            elif "25-to-50" in n:
                ownership = "25-50%"
        psc.append({
            "name": name,
            "ownership": ownership,
            "kind": "individual" if "individual" in kind else "corporate",
            "natures": natures,
        })

    if not psc:
        psc = [{"kind": "No PSC data available", "name": None, "ownership": None}]

    # ── Charges ──
    total_charges = charges_raw.get("total_count", 0) if isinstance(charges_raw, dict) else 0
    satisfied = 0
    outstanding = 0
    if isinstance(charges_raw, dict):
        for c in charges_raw.get("items", []):
            status = c.get("status", "")
            if status in ("fully-satisfied", "satisfied"):
                satisfied += 1
            else:
                outstanding += 1

    # ── Filing history types (for filing behaviour analysis) ──
    filing_types = []
    for f in accounts_filings:
        desc = (f.get("description", "") + " " + f.get("type", "")).lower()
        if "micro" in desc:
            filing_types.append("micro-entity")
        elif "small" in desc or "abridged" in desc:
            filing_types.append("small")
        elif "medium" in desc:
            filing_types.append("medium")
        else:
            filing_types.append("full")

    # ── Account type ──
    acc_type = last_acc.get("type", "unknown")
    # Normalise
    if "micro" in acc_type:
        acc_type = "micro-entity"
    elif "small" in acc_type or "abridged" in acc_type:
        acc_type = "small"
    elif "medium" in acc_type:
        acc_type = "medium"
    elif "full" in acc_type or "group" in acc_type:
        acc_type = "full"

    # ── Insolvency ──
    insolvency_cases = []
    has_active = False
    try:
        raw_cases = insolvency_raw.get("cases", [])
        if isinstance(raw_cases, list):
            for case in raw_cases:
                if not isinstance(case, dict):
                    continue
                dates = case.get("dates", []) or []
                practitioners = case.get("practitioners", []) or []
                insolvency_cases.append({
                    "type": case.get("type", "unknown"),
                    "number": case.get("number"),
                    "dates": [{"date": d.get("date"), "type": d.get("type")} for d in dates if isinstance(d, dict)],
                    "practitioners": [{"name": p.get("name"), "role": p.get("role")} for p in practitioners[:2] if isinstance(p, dict)],
                    "notes": case.get("notes", []),
                })
            if insolvency_cases:
                has_active = insolvency_raw.get("status") not in (None, "closed", "discharged")
    except Exception as e:
        print(f"[Insolvency] Parse error: {e}")
        insolvency_cases = []

    return {
        "company_name": profile.get("company_name", ""),
        "company_number": number,
        "company_status": profile.get("company_status", ""),
        "type": profile.get("type", ""),
        "date_of_creation": profile.get("date_of_creation", ""),
        "sic_codes": profile.get("sic_codes", []),
        "registered_office_address": addr,
        "accounts": {
            "overdue": overdue,
            "last_accounts": {
                "made_up_to": last_acc.get("made_up_to", ""),
                "type": acc_type,
            },
            "next_due": next_due,
        },
        "confirmation_statement": {
            "overdue": conf.get("overdue", False),
        },
        "officers": officers,
        "psc": psc,
        "charges": {
            "total": total_charges,
            "satisfied": satisfied,
            "outstanding": outstanding,
        },
        "insolvency": {
            "cases": insolvency_cases,
            "status": insolvency_raw.get("status"),
            "has_active_case": has_active,
        },
        "gazette_notices": gazette_notices,
        "filing_history_types": filing_types,
        "accounts_filings": accounts_filings,
        "financials": [],
    }
