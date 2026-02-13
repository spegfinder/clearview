"""Clearview Credit Assessment Model v1.0

Multi-factor credit scoring for UK small companies using balance-sheet data
and non-financial signals from the Companies House register.

Based on:
- Altman Z''-Score (1995) — balance-sheet-focused bankruptcy prediction
- Company Watch H-Score methodology — retained earnings as profit proxy
- Altman, Sabato & Wilson (2010) — non-financial SME risk factors
- Peel (1992) — UK small firm industry effects
"""

from datetime import datetime, date


# ═══════════════════════════════════════════════════════════════════
#  HELPER: interpolate a value within scored bands
# ═══════════════════════════════════════════════════════════════════

def _score_band(value, bands):
    """Score a value against a list of (threshold, score) bands.

    bands = [(threshold_1, score_1), (threshold_2, score_2), ...]
    Must be sorted by threshold ascending.
    Returns linearly interpolated score between bands.
    """
    if value is None:
        return None

    if value <= bands[0][0]:
        return bands[0][1]
    if value >= bands[-1][0]:
        return bands[-1][1]

    for i in range(len(bands) - 1):
        low_thresh, low_score = bands[i]
        high_thresh, high_score = bands[i + 1]
        if low_thresh <= value <= high_thresh:
            # Linear interpolation
            if high_thresh == low_thresh:
                return high_score
            ratio = (value - low_thresh) / (high_thresh - low_thresh)
            return low_score + ratio * (high_score - low_score)

    return bands[-1][1]


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 1: FINANCIAL HEALTH (60% of total)
# ═══════════════════════════════════════════════════════════════════

# Each ratio has scored bands: (value, score) pairs sorted ascending.
# Score is linearly interpolated between bands.

RATIO_BANDS = {
    # Net Assets / Total Assets — solvency
    "net_assets_ratio": [
        (-0.5, 0), (0.0, 10), (0.10, 25), (0.25, 45),
        (0.50, 65), (0.75, 85), (1.0, 100),
    ],
    # Current Assets / Current Liabilities — liquidity
    "current_ratio": [
        (0.0, 0), (0.5, 10), (0.8, 30), (1.2, 50),
        (2.0, 75), (4.0, 90), (8.0, 100),
    ],
    # Cash / Current Liabilities — immediate liquidity
    "cash_ratio": [
        (0.0, 0), (0.05, 15), (0.20, 35), (0.50, 60),
        (1.0, 80), (2.0, 100),
    ],
    # Total Liabilities / Total Assets — leverage (inverted: lower = better)
    "debt_ratio": [
        (0.0, 100), (0.20, 85), (0.40, 65), (0.60, 45),
        (0.80, 25), (1.0, 10), (1.5, 0),
    ],
    # Retained Earnings / Total Assets — cumulative profitability
    "retained_earnings_ratio": [
        (-1.0, 0), (-0.5, 5), (0.0, 20), (0.15, 40),
        (0.35, 60), (0.60, 80), (1.0, 100),
    ],
    # Working Capital / Total Assets — Altman X1
    "working_capital_ratio": [
        (-0.5, 0), (-0.2, 10), (0.0, 25), (0.15, 45),
        (0.35, 70), (0.60, 90), (1.0, 100),
    ],
}

# Weights for each ratio in the financial health pillar
RATIO_WEIGHTS = {
    "net_assets_ratio": 0.25,
    "current_ratio": 0.20,
    "debt_ratio": 0.20,
    "retained_earnings_ratio": 0.15,
    "cash_ratio": 0.10,
    "working_capital_ratio": 0.10,
}


def _safe_div(a, b):
    """Safe division returning None if either value is None or b is 0."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def calc_financial_ratios(fin):
    """Calculate financial ratios from a single year's data.

    fin: dict with keys like total_assets, current_assets, etc.
    Returns: dict of ratio_name -> value (or None if not calculable)
    """
    ta = fin.get("total_assets")
    ca = fin.get("current_assets")
    cl = fin.get("current_liabilities") or fin.get("creditors_due_within_year")
    na = fin.get("net_assets")
    re = fin.get("retained_earnings")
    cash = fin.get("cash")

    # Derive total_liabilities if not directly available
    tl = fin.get("total_liabilities")
    if tl is None and ta is not None and na is not None:
        tl = ta - na  # total_liabilities = total_assets - net_assets

    # Working capital = current_assets - current_liabilities
    wc = None
    if ca is not None and cl is not None:
        wc = ca - cl

    ratios = {
        "net_assets_ratio": _safe_div(na, ta),
        "current_ratio": _safe_div(ca, cl),
        "cash_ratio": _safe_div(cash, cl),
        "debt_ratio": _safe_div(tl, ta),
        "retained_earnings_ratio": _safe_div(re, ta),
        "working_capital_ratio": _safe_div(wc, ta),
    }

    return ratios


def score_financial_health(fin):
    """Score a single year's financials on 0-100 scale.

    Returns: (score, details_dict)
    """
    ratios = calc_financial_ratios(fin)

    scored = {}
    total_weight = 0
    weighted_sum = 0

    for name, value in ratios.items():
        s = _score_band(value, RATIO_BANDS[name])
        scored[name] = {
            "value": round(value, 4) if value is not None else None,
            "score": round(s, 1) if s is not None else None,
            "weight": RATIO_WEIGHTS[name],
        }
        if s is not None:
            weighted_sum += s * RATIO_WEIGHTS[name]
            total_weight += RATIO_WEIGHTS[name]

    if total_weight == 0:
        return 50, scored  # No data — neutral

    # Normalise weights to sum to 1
    financial_health = weighted_sum / total_weight

    return round(financial_health, 1), scored


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 2: STABILITY SIGNALS (25% of total)
# ═══════════════════════════════════════════════════════════════════

def score_stability(company_data):
    """Score non-financial stability signals on 0-100 scale.

    company_data: the full company dict from build_company_data()
    Returns: (score, signals_list)
    """
    signals = []
    adjustment = 0

    # 2.1 Company Age
    doc = company_data.get("date_of_creation")
    if doc:
        try:
            created = datetime.strptime(doc, "%Y-%m-%d").date()
            age_years = (date.today() - created).days / 365.25
            if age_years < 2:
                adjustment -= 20
                signals.append(("Company age < 2 years", -20, "high_risk"))
            elif age_years < 3:
                adjustment -= 10
                signals.append(("Company age 2-3 years", -10, "risk"))
            elif age_years < 5:
                adjustment -= 5
                signals.append(("Company age 3-5 years", -5, "caution"))
            elif age_years < 10:
                adjustment += 5
                signals.append(("Company age 5-10 years", +5, "positive"))
            elif age_years < 20:
                adjustment += 10
                signals.append(("Company age 10-20 years", +10, "positive"))
            else:
                adjustment += 15
                signals.append((f"Established company ({int(age_years)} years)", +15, "positive"))
        except (ValueError, TypeError):
            pass

    # 2.2 Filing Behaviour
    acc = company_data.get("accounts", {})
    cs = company_data.get("confirmation_statement", {})
    acc_overdue = acc.get("overdue", False)
    cs_overdue = cs.get("overdue", False)

    if acc_overdue and cs_overdue:
        adjustment -= 30
        signals.append(("Accounts AND confirmation statement overdue", -30, "high_risk"))
    elif acc_overdue:
        adjustment -= 25
        signals.append(("Accounts overdue", -25, "high_risk"))
    elif cs_overdue:
        adjustment -= 10
        signals.append(("Confirmation statement overdue", -10, "risk"))
    else:
        signals.append(("Filing up to date", +5, "positive"))
        adjustment += 5

    # 2.3 Filing Type Changes (downgrades)
    filing_types = company_data.get("filing_history_types", [])
    if len(filing_types) >= 2:
        type_rank = {"micro-entity": 0, "small": 1, "medium": 2, "full": 3}
        recent = type_rank.get(filing_types[0], 1)
        prev = type_rank.get(filing_types[1], 1)

        if recent < prev:
            adjustment -= 10
            signals.append(("Filing downgrade detected", -10, "risk"))
        elif recent > prev:
            adjustment += 5
            signals.append(("Filing upgrade (more transparency)", +5, "positive"))

    # 2.4 Director Stability
    officers = company_data.get("officers", [])
    today = date.today()
    active_directors = []
    recent_resignations = 0

    for o in officers:
        is_director = "director" in (o.get("role", "")).lower()
        if not is_director:
            continue

        if o.get("resigned"):
            try:
                resigned_date = datetime.strptime(o["resigned"], "%Y-%m-%d").date()
                months_since = (today - resigned_date).days / 30.44
                if months_since <= 24:
                    recent_resignations += 1
            except (ValueError, TypeError):
                pass
        else:
            active_directors.append(o)

    if len(active_directors) <= 1:
        adjustment -= 5
        signals.append(("Only 1 active director", -5, "caution"))

    if recent_resignations >= 3:
        adjustment -= 15
        signals.append((f"{recent_resignations} directors resigned in 24 months", -15, "high_risk"))
    elif recent_resignations >= 1:
        penalty = min(recent_resignations * 5, 15)
        adjustment -= penalty
        signals.append((f"{recent_resignations} director resignation(s) recently", -penalty, "risk"))
    elif len(active_directors) >= 2:
        adjustment += 10
        signals.append(("Stable board", +10, "positive"))

    # 2.5 Outstanding Charges
    charges = company_data.get("charges", {})
    outstanding = charges.get("outstanding", 0)
    if outstanding >= 3:
        adjustment -= 10
        signals.append((f"{outstanding} outstanding charges", -10, "risk"))
    elif outstanding >= 1:
        adjustment -= 5
        signals.append((f"{outstanding} outstanding charge(s)", -5, "caution"))
    else:
        adjustment += 5
        signals.append(("No outstanding charges", +5, "positive"))

    score = max(0, min(100, 50 + adjustment))
    return score, signals


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 3: TREND SCORE (15% of total)
# ═══════════════════════════════════════════════════════════════════

def _pct_change(new, old):
    """Calculate percentage change, handling None and zero."""
    if new is None or old is None:
        return None
    if old == 0:
        return 1.0 if new > 0 else (-1.0 if new < 0 else 0)
    return (new - old) / abs(old)


def score_trends(financials):
    """Score year-on-year trends on 0-100 scale.

    financials: list of financial records sorted newest first
    Returns: (score, trends_list)
    """
    trends = []
    adjustment = 0

    if len(financials) < 2:
        return 50, [("Insufficient data for trend analysis", 0, "neutral")]

    curr = financials[0]
    prev = financials[1]

    # 3.1 Retained Earnings Change (profit proxy)
    re_change = _pct_change(curr.get("retained_earnings"), prev.get("retained_earnings"))
    if re_change is not None:
        if re_change > 0.05:
            adjustment += 20
            trends.append(("Retained earnings growing (profit proxy)", +20, "positive"))
        elif re_change >= -0.05:
            adjustment += 10
            trends.append(("Retained earnings stable", +10, "neutral"))
        elif re_change >= -0.20:
            adjustment -= 10
            trends.append(("Retained earnings declining", -10, "risk"))
        else:
            adjustment -= 20
            trends.append(("Significant retained earnings decline", -20, "high_risk"))

        # Calculate the estimated profit figure for display
        if curr.get("retained_earnings") is not None and prev.get("retained_earnings") is not None:
            est_profit = curr["retained_earnings"] - prev["retained_earnings"]
            trends.append((f"Estimated annual profit/loss: £{est_profit:,.0f}", 0, "info"))

    # 3.2 Net Assets Trend
    na_change = _pct_change(curr.get("net_assets"), prev.get("net_assets"))
    if na_change is not None:
        if na_change > 0.05:
            adjustment += 15
            trends.append(("Net assets growing", +15, "positive"))
        elif na_change >= -0.05:
            adjustment += 5
            trends.append(("Net assets stable", +5, "neutral"))
        else:
            adjustment -= 15
            trends.append(("Net assets declining", -15, "risk"))

    # 3.3 Current Ratio Trend
    cr_curr = _safe_div(curr.get("current_assets"),
                        curr.get("current_liabilities") or curr.get("creditors_due_within_year"))
    cr_prev = _safe_div(prev.get("current_assets"),
                        prev.get("current_liabilities") or prev.get("creditors_due_within_year"))
    if cr_curr is not None and cr_prev is not None:
        cr_delta = cr_curr - cr_prev
        if cr_delta > 0.1:
            adjustment += 10
            trends.append(("Current ratio improving", +10, "positive"))
        elif cr_delta >= -0.1:
            adjustment += 5
            trends.append(("Current ratio stable", +5, "neutral"))
        else:
            adjustment -= 10
            trends.append(("Current ratio deteriorating", -10, "risk"))

    # 3.4 Cash Trend
    cash_change = _pct_change(curr.get("cash"), prev.get("cash"))
    if cash_change is not None:
        if cash_change > 0.1:
            adjustment += 10
            trends.append(("Cash position improving", +10, "positive"))
        elif cash_change >= -0.1:
            adjustment += 5
            trends.append(("Cash position stable", +5, "neutral"))
        else:
            adjustment -= 10
            trends.append(("Cash position declining", -10, "risk"))

    score = max(0, min(100, 50 + adjustment))
    return score, trends


# ═══════════════════════════════════════════════════════════════════
#  ALTMAN Z''-SCORE (when full P&L available)
# ═══════════════════════════════════════════════════════════════════

def calc_altman_z(fin):
    """Calculate Altman Z''-Score for non-manufacturing private firms.

    Z'' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BV Equity/TL)

    Returns: (z_score, zone, components) or (None, None, None) if insufficient data
    """
    ta = fin.get("total_assets")
    ca = fin.get("current_assets")
    cl = fin.get("current_liabilities") or fin.get("creditors_due_within_year")
    re = fin.get("retained_earnings")
    ebit = fin.get("ebit")
    na = fin.get("net_assets")

    # We need total_liabilities
    tl = fin.get("total_liabilities")
    if tl is None and ta is not None and na is not None:
        tl = ta - na

    # Check minimum data requirements
    if ta is None or ta == 0:
        return None, None, None

    # Working capital
    wc = None
    if ca is not None and cl is not None:
        wc = ca - cl

    components = {}

    # X1: WC/TA
    x1 = _safe_div(wc, ta)
    components["wc_ta"] = x1

    # X2: RE/TA
    x2 = _safe_div(re, ta)
    components["re_ta"] = x2

    # X3: EBIT/TA (requires P&L data)
    x3 = _safe_div(ebit, ta)
    components["ebit_ta"] = x3

    # X4: Book Equity / Total Liabilities
    x4 = _safe_div(na, tl) if tl and tl > 0 else None
    components["equity_tl"] = x4

    # If we don't have EBIT, try using retained earnings change as proxy
    # (This makes it a "Modelled Z-Score")
    modelled = False
    if x3 is None:
        modelled = True
        # We can't calculate a proper Z-score without EBIT
        # But we can calculate a partial score using 3 of 4 components
        if x1 is not None and x2 is not None and x4 is not None:
            # Use the 3 available components with adjusted weights
            z = 6.56 * x1 + 3.26 * x2 + 1.05 * x4
            # This is missing the 6.72*EBIT/TA component
            # Add a rough estimate: assume EBIT/TA ≈ change in RE/TA (crude proxy)
            zone = "safe" if z > 2.6 else ("grey" if z > 1.1 else "distress")
            components["modelled"] = True
            components["note"] = "Missing EBIT — partial score using 3 of 4 components"
            return round(z, 2), zone, components
        return None, None, None

    # Full Z''-Score
    if any(v is None for v in [x1, x2, x3, x4]):
        return None, None, None

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    zone = "safe" if z > 2.6 else ("grey" if z > 1.1 else "distress")
    components["modelled"] = modelled

    return round(z, 2), zone, components


# ═══════════════════════════════════════════════════════════════════
#  CONFIDENCE LEVEL
# ═══════════════════════════════════════════════════════════════════

def calc_confidence(financials, company_data):
    """Determine confidence level based on data completeness.

    Returns: ("high" | "medium" | "low", reason_string)
    """
    n_years = len(financials)

    if n_years == 0:
        return "low", "No financial data available"

    # Count how many key fields are present in most recent year
    latest = financials[0]
    key_fields = ["total_assets", "current_assets", "current_liabilities",
                  "net_assets", "retained_earnings", "cash"]
    available = sum(1 for f in key_fields if latest.get(f) is not None)
    completeness = available / len(key_fields)

    has_officers = len(company_data.get("officers", [])) > 0
    has_age = company_data.get("date_of_creation") is not None

    if n_years >= 3 and completeness >= 0.8 and has_officers and has_age:
        return "high", f"{n_years} years data, {available}/{len(key_fields)} balance sheet fields"
    elif n_years >= 2 and completeness >= 0.5:
        return "medium", f"{n_years} years data, {available}/{len(key_fields)} balance sheet fields"
    else:
        return "low", f"{n_years} year(s) data, {available}/{len(key_fields)} balance sheet fields"


# ═══════════════════════════════════════════════════════════════════
#  COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════

PILLAR_WEIGHTS = {
    "financial_health": 0.60,
    "stability": 0.25,
    "trend": 0.15,
}

RATING_BANDS = [
    (80, "A", "Strong", "Low risk. Well capitalised, stable, positive trends."),
    (65, "B", "Good", "Moderate-low risk. Fundamentally sound."),
    (50, "C", "Fair", "Some concerns. Monitor closely."),
    (35, "D", "Weak", "Elevated risk. Significant concerns present."),
    (20, "E", "Poor", "High risk. Multiple warning signs."),
    (0,  "F", "Critical", "Very high risk. May be insolvent."),
]


def get_rating(score):
    """Convert a 0-100 score into a rating band."""
    for threshold, grade, label, desc in RATING_BANDS:
        if score >= threshold:
            return {"grade": grade, "label": label, "description": desc}
    return {"grade": "F", "label": "Critical", "description": "Very high risk. May be insolvent."}


def assess_company(company_data, financials):
    """Run the full Clearview assessment.

    company_data: dict from build_company_data()
    financials: list of financial records, sorted newest first

    Returns: dict with full assessment results
    """
    # ── Pillar 1: Financial Health ──
    if financials:
        fh_score, fh_details = score_financial_health(financials[0])
    else:
        fh_score, fh_details = 50, {}

    # ── Pillar 2: Stability Signals ──
    stab_score, stab_signals = score_stability(company_data)

    # ── Pillar 3: Trends ──
    trend_score, trend_details = score_trends(financials)

    # ── Composite Score ──
    composite = (
        fh_score * PILLAR_WEIGHTS["financial_health"]
        + stab_score * PILLAR_WEIGHTS["stability"]
        + trend_score * PILLAR_WEIGHTS["trend"]
    )
    composite = round(composite, 1)

    # ── Rating ──
    rating = get_rating(composite)

    # ── Confidence ──
    confidence, conf_reason = calc_confidence(financials, company_data)

    # ── Altman Z-Score ──
    altman = {"z_score": None, "zone": None, "components": None}
    if financials:
        z, zone, comps = calc_altman_z(financials[0])
        altman = {"z_score": z, "zone": zone, "components": comps}

    return {
        "clearview_score": composite,
        "rating": rating,
        "confidence": confidence,
        "confidence_reason": conf_reason,
        "pillars": {
            "financial_health": {
                "score": fh_score,
                "weight": PILLAR_WEIGHTS["financial_health"],
                "ratios": fh_details,
            },
            "stability": {
                "score": stab_score,
                "weight": PILLAR_WEIGHTS["stability"],
                "signals": stab_signals,
            },
            "trend": {
                "score": trend_score,
                "weight": PILLAR_WEIGHTS["trend"],
                "details": trend_details,
            },
        },
        "altman_z": altman,
    }
