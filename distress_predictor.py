"""Clearview Insolvency Prediction — Production Module.

Loads the pre-trained model weights from clearview_model.json and predicts
the probability of company financial distress.

No sklearn dependency needed — uses pure Python with exported lookup tables.
"""

import os
import json
import math
from datetime import datetime

_model = None


def _load_model():
    """Load model weights from JSON file. Returns default rates if not available."""
    global _model
    if _model is not None:
        return _model

    model_path = os.path.join(os.path.dirname(__file__), "clearview_model.json")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            _model = json.load(f)
        print(f"[Predictor] Loaded ML model v{_model.get('version', '?')} "
              f"(trained {_model.get('trained_on', 'unknown')[:10]})")
    else:
        # Fallback: use published UK SME insolvency statistics as base rates
        # Source: ONS business demography, Insolvency Service quarterly stats
        print("[Predictor] No ML model file — using published UK base rates")
        _model = {
            "version": "fallback",
            "baseline_prob": 0.015,  # ~1.5% annual failure rate for established SMEs
            "base_rates": {},
            "adjustments": {
                "accounts_overdue": 2.5,
                "num_outstanding_charges": 1.6,
                "days_since_filing_800": 1.4,
                "days_since_filing_200": 0.85,
                "num_charges_5": 1.2,
                "num_charges_10": 1.5,
            },
            "sector_rates": {
                "41": 0.035, "42": 0.032, "43": 0.038,  # Construction
                "56": 0.028, "55": 0.025,  # Hospitality
                "47": 0.022, "45": 0.020,  # Retail
                "68": 0.018,  # Real estate
                "62": 0.012, "63": 0.012,  # Tech
                "69": 0.008, "70": 0.010,  # Professional services
                "86": 0.006,  # Health
            },
            "age_buckets": [0.5, 1, 2, 3, 5, 8, 12, 20, 50],
        }
        # Generate base rates from published stats
        # Young companies fail much more: ~10% in year 1, ~5% in year 2-3, settling to ~1-2%
        age_probs = {
            0.5: 0.12, 1: 0.08, 2: 0.05, 3: 0.035,
            5: 0.02, 8: 0.015, 12: 0.012, 20: 0.010, 50: 0.008
        }
        for age_b in _model["age_buckets"]:
            for hr in [0, 1]:
                for acc in ["dormant", "micro", "small", "full"]:
                    base = age_probs.get(age_b, 0.015)
                    if hr:
                        base *= 1.8
                    if acc == "dormant":
                        base *= 0.6
                    elif acc == "full":
                        base *= 0.7  # larger, more established
                    key = f"{age_b}_{hr}_{acc}"
                    _model["base_rates"][key] = round(base, 6)

    return _model


def predict_distress(company_data, financials=None):
    """Predict the probability of financial distress for a company.

    Args:
        company_data: dict with company profile data (from build_company_data)
        financials: list of financial year records (newest first), optional

    Returns:
        dict with:
            probability: float 0-1 (chance of distress in next 24 months)
            risk_band: str ("very_low", "low", "moderate", "elevated", "high", "very_high")
            factors: list of (description, impact) tuples
            confidence: str ("high", "medium", "low")
        or None if model not available
    """
    model = _load_model()
    if model is None:
        return None

    factors = []
    confidence = "medium"

    # ── Extract features from company data ──

    # Age
    doc = company_data.get("date_of_creation", "")
    try:
        inc_date = datetime.strptime(doc[:10], "%Y-%m-%d")
        age_years = (datetime.now() - inc_date).days / 365.25
    except Exception:
        age_years = 5  # default
        confidence = "low"

    # SIC code
    sic_codes = company_data.get("sic_codes", [])
    sic_2digit = 0
    if sic_codes:
        try:
            sic_2digit = int(str(sic_codes[0])[:2])
        except (ValueError, IndexError):
            pass

    # Account type
    acc_type_raw = company_data.get("accounts", {}).get("last_accounts", {}).get("type", "")
    acc_type = "micro"  # default
    if "dormant" in acc_type_raw.lower():
        acc_type = "dormant"
    elif "small" in acc_type_raw.lower():
        acc_type = "small"
    elif "medium" in acc_type_raw.lower() or "full" in acc_type_raw.lower():
        acc_type = "full"

    # High risk sector
    high_risk_sectors = [41, 42, 43, 56, 68, 47, 49]
    high_risk = 1 if sic_2digit in high_risk_sectors else 0

    # Charges
    charges = company_data.get("charges", {})
    num_charges = charges.get("total", 0) or 0
    num_outstanding = charges.get("outstanding", 0) or 0

    # Filing behaviour
    accounts_overdue = 1 if company_data.get("accounts", {}).get("overdue", False) else 0
    conf_overdue = 1 if company_data.get("confirmation_statement", {}).get("overdue", False) else 0

    last_made_up = company_data.get("accounts", {}).get("last_accounts", {}).get("made_up_to", "")
    days_since_filing = 400  # default
    try:
        last_date = datetime.strptime(last_made_up[:10], "%Y-%m-%d")
        days_since_filing = (datetime.now() - last_date).days
    except Exception:
        pass

    # Insolvency history
    insolvency = company_data.get("insolvency", {})
    has_active_insolvency = insolvency.get("has_active_case", False)
    past_insolvency_cases = len(insolvency.get("cases", []))

    # ── Look up base rate ──

    base_rates = model.get("base_rates", {})
    age_buckets = model.get("age_buckets", [0.5, 1, 2, 3, 5, 8, 12, 20, 50])

    # Find age bucket
    age_bucket = age_buckets[-1]
    for bucket in age_buckets:
        if age_years <= bucket:
            age_bucket = bucket
            break

    key = f"{age_bucket}_{high_risk}_{acc_type}"
    base_prob = base_rates.get(key, model.get("baseline_prob", 0.02))

    # Sector-specific rate
    sector_rates = model.get("sector_rates", {})
    sector_prob = sector_rates.get(str(sic_2digit))
    if sector_prob is not None:
        # Blend sector rate with base rate
        base_prob = (base_prob + sector_prob) / 2

    prob = base_prob

    # ── Apply adjustments ──

    adjustments = model.get("adjustments", {})

    # Accounts overdue
    if accounts_overdue:
        mult = adjustments.get("accounts_overdue", 2.0)
        prob *= mult
        factors.append(("Accounts are overdue", "increases_risk"))

    # Confirmation statement overdue
    if conf_overdue:
        prob *= 1.3
        factors.append(("Confirmation statement overdue", "increases_risk"))

    # Outstanding charges
    if num_outstanding > 0:
        mult = adjustments.get("num_outstanding_charges", 1.5)
        prob *= mult
        if num_outstanding >= 3:
            factors.append((f"{num_outstanding} outstanding charges", "increases_risk"))

    # Days since filing
    if days_since_filing > 800:
        mult = adjustments.get("days_since_filing_800", 1.3)
        prob *= mult
        factors.append(("Very old accounts on file", "increases_risk"))
    elif days_since_filing < 300:
        mult = adjustments.get("days_since_filing_200", 0.9)
        prob *= mult

    # Charges count
    if num_charges >= 5:
        mult = adjustments.get("num_charges_5", 1.2)
        prob *= mult
    elif num_charges >= 10:
        mult = adjustments.get("num_charges_10", 1.4)
        prob *= mult

    # ── Financial ratio adjustments (when we have data) ──

    if financials and len(financials) > 0:
        confidence = "high"
        fin = financials[0]

        na = fin.get("net_assets")
        ca = fin.get("current_assets")
        cl = fin.get("current_liabilities") or fin.get("creditors_due_within_year")
        re = fin.get("retained_earnings")
        cash = fin.get("cash")

        # Negative net assets — strong failure signal
        if na is not None and na < 0:
            prob *= 3.0
            factors.append(("Negative net assets", "major_risk"))

        # Current ratio
        if ca is not None and cl is not None and cl > 0:
            current_ratio = ca / cl
            if current_ratio < 0.5:
                prob *= 2.5
                factors.append(("Current ratio below 0.5 — severe liquidity risk", "major_risk"))
            elif current_ratio < 0.8:
                prob *= 1.8
                factors.append(("Current ratio below 0.8 — liquidity concern", "increases_risk"))
            elif current_ratio < 1.0:
                prob *= 1.3
                factors.append(("Current ratio below 1.0", "slight_risk"))
            elif current_ratio > 2.0:
                prob *= 0.7
                factors.append(("Strong current ratio", "reduces_risk"))

        # Negative retained earnings
        if re is not None and re < 0:
            prob *= 1.6
            factors.append(("Accumulated losses (negative retained earnings)", "increases_risk"))

        # Low cash
        if cash is not None and cl is not None and cl > 0:
            cash_ratio = cash / cl
            if cash_ratio < 0.05:
                prob *= 1.5
                factors.append(("Almost no cash relative to liabilities", "increases_risk"))

        # Multi-year trend analysis
        if len(financials) >= 2:
            prev = financials[1]
            prev_na = prev.get("net_assets")

            if na is not None and prev_na is not None:
                if prev_na > 0:
                    na_change = (na - prev_na) / prev_na
                    if na_change < -0.3:
                        prob *= 1.8
                        factors.append(("Net assets dropped >30% year-on-year", "major_risk"))
                    elif na_change < -0.1:
                        prob *= 1.3
                        factors.append(("Net assets declining", "increases_risk"))
                    elif na_change > 0.1:
                        prob *= 0.85
                        factors.append(("Net assets growing", "reduces_risk"))

            prev_cash = prev.get("cash")
            if cash is not None and prev_cash is not None and prev_cash > 0:
                cash_change = (cash - prev_cash) / prev_cash
                if cash_change < -0.5:
                    prob *= 1.4
                    factors.append(("Cash halved year-on-year", "increases_risk"))

    else:
        factors.append(("Limited financial data — using company profile only", "note"))
        confidence = "low"

    # ── Active insolvency = near-certain ──
    if has_active_insolvency:
        prob = 0.95
        factors = [("Active insolvency proceeding", "critical")]
        confidence = "high"

    if past_insolvency_cases > 0 and not has_active_insolvency:
        prob *= 2.0
        factors.append(("Past insolvency history", "increases_risk"))

    # ── Age-based factors ──
    if age_years < 2:
        factors.append(("Company less than 2 years old — higher base failure rate", "increases_risk"))
    elif age_years > 15:
        factors.append(("Established company (15+ years)", "reduces_risk"))
        prob *= 0.8

    if high_risk:
        sector_names = {41: "Construction", 42: "Civil engineering", 43: "Specialist construction",
                        56: "Food & beverage", 68: "Real estate", 47: "Retail", 49: "Transport"}
        name = sector_names.get(sic_2digit, "this sector")
        factors.append((f"{name} has above-average failure rates", "increases_risk"))

    # ── Clamp probability ──
    prob = max(0.001, min(0.95, prob))

    # ── Risk band ──
    if prob < 0.02:
        risk_band = "very_low"
    elif prob < 0.05:
        risk_band = "low"
    elif prob < 0.10:
        risk_band = "moderate"
    elif prob < 0.20:
        risk_band = "elevated"
    elif prob < 0.40:
        risk_band = "high"
    else:
        risk_band = "very_high"

    return {
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 1),
        "risk_band": risk_band,
        "factors": factors,
        "confidence": confidence,
    }
