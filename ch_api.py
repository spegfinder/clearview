"""Companies House REST API client."""
import requests
import time
from functools import lru_cache

BASE = "https://api.companieshouse.gov.uk"
DOC_BASE = "https://frontend-doc-api.company-information.service.gov.uk"


class CompaniesHouseClient:
    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update({"Accept": "application/json"})
        self._last_call = 0

    def _get(self, url, **kwargs):
        # Simple rate limiting: 100ms between calls (well within 600/5min)
        elapsed = time.time() - self._last_call
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_call = time.time()

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
        """Download the actual document content (iXBRL/XHTML).

        document_metadata_url is typically like /document/abc-123
        Returns (content_bytes, content_type) or (None, None)
        """
        # First get the document metadata to find the content URL
        meta_url = document_metadata_url
        if meta_url.startswith("/"):
            meta_url = f"{BASE}{meta_url}"

        meta = self._get(meta_url)
        if not meta:
            return None, None

        # The metadata contains links to the actual content
        resources = meta.get("resources", {})

        # Prefer iXBRL (application/xhtml+xml) over PDF
        for content_type in ["application/xhtml+xml", "application/xml", "text/html"]:
            if content_type in resources:
                content_url = meta.get("links", {}).get("document", "")
                if content_url:
                    if content_url.startswith("/"):
                        content_url = f"{DOC_BASE}{content_url}"
                    # Fetch with the right Accept header
                    elapsed = time.time() - self._last_call
                    if elapsed < 0.1:
                        time.sleep(0.1 - elapsed)
                    self._last_call = time.time()

                    resp = self.session.get(content_url, headers={
                        "Accept": content_type
                    }, allow_redirects=True)
                    if resp.status_code == 200:
                        return resp.content, content_type
        return None, None


def build_company_data(client, number):
    """Fetch all data for a company and structure it for the frontend."""
    profile = client.get_profile(number)
    if not profile:
        return None

    officers_raw = client.get_officers(number)
    psc_raw = client.get_psc(number)
    charges_raw = client.get_charges(number)
    accounts_filings = client.get_accounts_filings(number, count=5)

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
        "filing_history_types": filing_types,
        "accounts_filings": accounts_filings,  # raw filings for document parsing
        "financials": [],  # populated by accounts parser
    }
