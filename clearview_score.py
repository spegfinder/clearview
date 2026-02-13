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


def _score_single_year(fin):
    """Score a single year's financials. Returns (score, ratios_dict) or (None, {})."""
    ratios = calc_financial_ratios(fin)
    total_weight = 0
    weighted_sum = 0
    scored = {}

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
        return None, scored
    return round(weighted_sum / total_weight, 1), scored


def score_financial_health(financials):
    """Score financial health using ALL available years, weighting recent years more.

    financials: list of financial records sorted newest first (up to 4 years)
    Returns: (score, details_dict)

    Weighting: Year 0 (latest) = 50%, Year 1 = 25%, Year 2 = 15%, Year 3 = 10%
    This means a company with 3 years of strong ratios scores higher than one
    with just 1 strong year, and sustained decline is penalised more heavily.
    """
    if not financials:
        return 50, {}

    # Weight recent years more heavily
    year_weights = [0.50, 0.25, 0.15, 0.10]

    year_scores = []
    latest_details = {}

    for i, fin in enumerate(financials[:4]):
        score, details = _score_single_year(fin)
        if i == 0:
            latest_details = details  # Show most recent year's breakdown to user
        if score is not None:
            w = year_weights[i] if i < len(year_weights) else 0.05
            year_scores.append((score, w))

    if not year_scores:
        return 50, latest_details

    # Normalise weights and compute weighted average
    total_w = sum(w for _, w in year_scores)
    blended = sum(s * w for s, w in year_scores) / total_w

    # Add year count info to details
    latest_details["_years_used"] = len(year_scores)
    latest_details["_year_scores"] = [
        {"year": financials[i].get("year", "?"), "score": round(s, 1)}
        for i, (s, _) in enumerate(year_scores)
    ]

    return round(blended, 1), latest_details


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

    # 2.6 Insolvency History
    insolvency = company_data.get("insolvency", {})
    insolvency_cases = insolvency.get("cases", [])
    if insolvency_cases:
        # Has insolvency history — very serious
        case_types = [c.get("type", "") for c in insolvency_cases]
        active_types = []
        for c in insolvency_cases:
            ctype = c.get("type", "unknown").replace("-", " ").replace("_", " ").title()
            active_types.append(ctype)

        if insolvency.get("has_active_case"):
            adjustment -= 40
            signals.append((f"Active insolvency: {active_types[0]}", -40, "high_risk"))
        else:
            adjustment -= 20
            signals.append((f"Past insolvency history ({len(insolvency_cases)} case(s))", -20, "high_risk"))
    else:
        signals.append(("No insolvency history", +5, "positive"))
        adjustment += 5

    # 2.7 Gazette Notices (winding-up petitions etc)
    gazette = company_data.get("gazette_notices", [])
    if gazette:
        adjustment -= 15
        signals.append((f"{len(gazette)} Gazette insolvency notice(s) found", -15, "high_risk"))

    # 2.8 Company Status
    status = company_data.get("company_status", "active")
    if status in ("dissolved", "liquidation", "receivership", "administration"):
        adjustment -= 50
        signals.append((f"Company status: {status}", -50, "high_risk"))

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
    """Score trends using ALL available years on 0-100 scale.

    Analyses year-on-year changes across the full history.
    Sustained trends (3+ years in same direction) get amplified.

    financials: list of financial records sorted newest first
    Returns: (score, trends_list)
    """
    trends = []
    adjustment = 0

    if len(financials) < 2:
        return 50, [("Insufficient data for trend analysis", 0, "neutral")]

    n_years = len(financials)
    trends.append((f"Analysed {n_years} years of accounts", 0, "info"))

    # ── 3.1 Retained Earnings trajectory (profit proxy) ──
    re_values = [(f.get("year", "?"), f.get("retained_earnings")) for f in financials]
    re_valid = [(y, v) for y, v in re_values if v is not None]

    if len(re_valid) >= 2:
        # Calculate year-on-year changes (newest to oldest)
        re_changes = []
        for i in range(len(re_valid) - 1):
            new_v = re_valid[i][1]
            old_v = re_valid[i + 1][1]
            ch = _pct_change(new_v, old_v)
            if ch is not None:
                re_changes.append(ch)

        if re_changes:
            # Average annual change
            avg_re = sum(re_changes) / len(re_changes)
            # Count direction consistency
            up_years = sum(1 for c in re_changes if c > 0.02)
            down_years = sum(1 for c in re_changes if c < -0.02)

            if avg_re > 0.05 and down_years == 0:
                adjustment += 25
                label = f"Retained earnings growing consistently ({len(re_changes)} years)"
                trends.append((label, +25, "positive"))
            elif avg_re > 0.05:
                adjustment += 15
                trends.append(("Retained earnings growing overall", +15, "positive"))
            elif avg_re >= -0.05:
                adjustment += 8
                trends.append(("Retained earnings broadly stable", +8, "neutral"))
            elif avg_re >= -0.20 or down_years < len(re_changes):
                adjustment -= 12
                trends.append(("Retained earnings declining", -12, "risk"))
            else:
                adjustment -= 25
                label = f"Sustained decline in retained earnings ({down_years} consecutive years)"
                trends.append((label, -25, "high_risk"))

        # Estimated profit from most recent pair
        curr_re = financials[0].get("retained_earnings")
        prev_re = financials[1].get("retained_earnings")
        if curr_re is not None and prev_re is not None:
            est_profit = curr_re - prev_re
            trends.append((f"Estimated annual profit/loss: \u00a3{est_profit:,.0f}", 0, "info"))

        # Multi-year profit trajectory if 3+ years
        if len(re_valid) >= 3:
            profits = []
            for i in range(len(re_valid) - 1):
                p = re_valid[i][1] - re_valid[i + 1][1]
                profits.append((re_valid[i][0], p))
            loss_years = sum(1 for _, p in profits if p < 0)
            if loss_years == 0 and len(profits) >= 2:
                trends.append((f"Profitable in all {len(profits)} years analysed", 0, "info"))
            elif loss_years == len(profits):
                adjustment -= 10
                trends.append((f"Loss-making in all {len(profits)} years analysed", -10, "high_risk"))
            elif loss_years > 0:
                trends.append((f"Loss-making in {loss_years} of {len(profits)} years", 0, "info"))

    # ── 3.2 Net Assets trajectory ──
    na_values = [(f.get("year", "?"), f.get("net_assets")) for f in financials]
    na_valid = [(y, v) for y, v in na_values if v is not None]

    if len(na_valid) >= 2:
        na_changes = []
        for i in range(len(na_valid) - 1):
            ch = _pct_change(na_valid[i][1], na_valid[i + 1][1])
            if ch is not None:
                na_changes.append(ch)

        if na_changes:
            avg_na = sum(na_changes) / len(na_changes)
            down_na = sum(1 for c in na_changes if c < -0.05)

            if avg_na > 0.05 and down_na == 0:
                adjustment += 15
                trends.append(("Net assets growing consistently", +15, "positive"))
            elif avg_na > 0.02:
                adjustment += 10
                trends.append(("Net assets growing", +10, "positive"))
            elif avg_na >= -0.05:
                adjustment += 5
                trends.append(("Net assets stable", +5, "neutral"))
            elif down_na == len(na_changes):
                adjustment -= 18
                trends.append((f"Net assets declining every year ({len(na_changes)} years)", -18, "high_risk"))
            else:
                adjustment -= 12
                trends.append(("Net assets declining", -12, "risk"))

    # ── 3.3 Liquidity trend (current ratio) ──
    cr_values = []
    for fin in financials:
        cr = _safe_div(fin.get("current_assets"),
                       fin.get("current_liabilities") or fin.get("creditors_due_within_year"))
        cr_values.append(cr)

    cr_valid = [v for v in cr_values if v is not None]
    if len(cr_valid) >= 2:
        # Compare newest vs oldest for overall direction
        overall_delta = cr_valid[0] - cr_valid[-1]
        # Also check most recent change
        recent_delta = cr_valid[0] - cr_valid[1]

        if overall_delta > 0.2 and recent_delta > 0:
            adjustment += 10
            trends.append(("Liquidity improving over time", +10, "positive"))
        elif overall_delta > 0:
            adjustment += 5
            trends.append(("Liquidity slightly improved", +5, "positive"))
        elif overall_delta > -0.15:
            adjustment += 3
            trends.append(("Liquidity broadly stable", +3, "neutral"))
        else:
            adjustment -= 10
            trends.append(("Liquidity deteriorating over time", -10, "risk"))

    # ── 3.4 Cash trend ──
    cash_values = [(f.get("year", "?"), f.get("cash")) for f in financials]
    cash_valid = [(y, v) for y, v in cash_values if v is not None]

    if len(cash_valid) >= 2:
        cash_changes = []
        for i in range(len(cash_valid) - 1):
            ch = _pct_change(cash_valid[i][1], cash_valid[i + 1][1])
            if ch is not None:
                cash_changes.append(ch)

        if cash_changes:
            avg_cash = sum(cash_changes) / len(cash_changes)
            if avg_cash > 0.1:
                adjustment += 8
                trends.append(("Cash position improving", +8, "positive"))
            elif avg_cash >= -0.1:
                adjustment += 4
                trends.append(("Cash position stable", +4, "neutral"))
            else:
                adjustment -= 8
                trends.append(("Cash position declining", -8, "risk"))

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
    """Determine confidence level based on data completeness across all years.

    Returns: ("high" | "medium" | "low", reason_string)
    """
    n_years = len(financials)

    if n_years == 0:
        return "low", "No financial data available"

    # Count how many key fields are present across years
    key_fields = ["total_assets", "current_assets", "current_liabilities",
                  "net_assets", "retained_earnings", "cash"]

    # Check most recent year
    latest = financials[0]
    available = sum(1 for f in key_fields if latest.get(f) is not None)
    completeness = available / len(key_fields)

    # Check consistency across years (do all years have the same fields?)
    multi_year_quality = 0
    if n_years >= 2:
        for yr in financials[1:]:
            yr_avail = sum(1 for f in key_fields if yr.get(f) is not None)
            multi_year_quality += yr_avail / len(key_fields)
        multi_year_quality /= (n_years - 1)

    has_officers = len(company_data.get("officers", [])) > 0
    has_age = company_data.get("date_of_creation") is not None

    if n_years >= 3 and completeness >= 0.8 and multi_year_quality >= 0.6 and has_officers and has_age:
        return "high", f"{n_years} years data, {available}/{len(key_fields)} fields, consistent history"
    elif n_years >= 2 and completeness >= 0.5:
        return "medium", f"{n_years} years data, {available}/{len(key_fields)} fields"
    else:
        reason = f"{n_years} year(s) data, {available}/{len(key_fields)} fields"
        if n_years < 2:
            reason += " (no trend data)"
        return "low", reason


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


def calc_credit_limit(score, financials, company_data):
    """Calculate a suggested credit limit based on score and financials.

    Uses a conservative approach:
    - Base limit derived from net assets (capped at percentage)
    - Adjusted by Clearview score (higher score = higher multiplier)
    - Capped at reasonable maximums for small company context
    - Zero if score is very low or company has insolvency history

    Returns: dict with limit, basis, and confidence
    """
    # No limit for very poor companies or active insolvency
    insolvency = company_data.get("insolvency", {})
    if score < 20 or insolvency.get("has_active_case"):
        return {"limit": 0, "basis": "Not recommended — high risk or active insolvency", "confidence": "low"}

    if not financials:
        return {"limit": 500, "basis": "Minimal data available — conservative limit", "confidence": "low"}

    f = financials[0]
    net_assets = f.get("net_assets")
    total_assets = f.get("total_assets")
    cash = f.get("cash")
    current_assets = f.get("current_assets")

    # Start with net assets as the base
    if net_assets is not None and net_assets > 0:
        # Allow credit up to a % of net assets depending on score
        if score >= 80:
            base = net_assets * 0.10  # 10% of net assets
        elif score >= 65:
            base = net_assets * 0.07
        elif score >= 50:
            base = net_assets * 0.05
        elif score >= 35:
            base = net_assets * 0.03
        else:
            base = net_assets * 0.02
        basis = "Based on net assets"
    elif total_assets is not None and total_assets > 0:
        # Fall back to total assets at lower percentages
        if score >= 65:
            base = total_assets * 0.03
        elif score >= 50:
            base = total_assets * 0.02
        else:
            base = total_assets * 0.01
        basis = "Based on total assets (net assets not available)"
    elif cash is not None and cash > 0:
        base = cash * 0.15
        basis = "Based on cash position"
    else:
        # Very limited data
        if score >= 65:
            base = 2000
        elif score >= 50:
            base = 1000
        else:
            base = 500
        basis = "Limited financial data — conservative estimate"

    # Apply score multiplier
    if score >= 80:
        multiplier = 1.5
    elif score >= 65:
        multiplier = 1.2
    elif score >= 50:
        multiplier = 1.0
    elif score >= 35:
        multiplier = 0.7
    else:
        multiplier = 0.4

    limit = base * multiplier

    # Past insolvency = halve it
    if insolvency.get("cases"):
        limit *= 0.5
        basis += " (reduced — past insolvency)"

    # Cap at reasonable maximums
    limit = max(250, min(limit, 500000))

    # Round to nice numbers
    if limit < 1000:
        limit = round(limit / 50) * 50  # Round to nearest 50
    elif limit < 10000:
        limit = round(limit / 250) * 250  # Round to nearest 250
    elif limit < 100000:
        limit = round(limit / 1000) * 1000  # Round to nearest 1000
    else:
        limit = round(limit / 5000) * 5000  # Round to nearest 5000

    # Confidence
    conf = "medium"
    if net_assets is not None and len(financials) >= 2 and score >= 50:
        conf = "high"
    elif net_assets is None or len(financials) < 2:
        conf = "low"

    return {"limit": int(limit), "basis": basis, "confidence": conf}


def assess_company(company_data, financials):
    """Run the full Clearview assessment.

    company_data: dict from build_company_data()
    financials: list of financial records, sorted newest first

    Returns: dict with full assessment results
    """
    # ── Pillar 1: Financial Health (multi-year weighted) ──
    if financials:
        fh_score, fh_details = score_financial_health(financials)
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

    # ── Credit Limit ──
    credit_limit = calc_credit_limit(composite, financials, company_data)

    # ── Insolvency Summary ──
    insolvency = company_data.get("insolvency", {})
    gazette = company_data.get("gazette_notices", [])

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
        "credit_limit": credit_limit,
        "insolvency": {
            "cases": len(insolvency.get("cases", [])),
            "active": insolvency.get("has_active_case", False),
            "gazette_notices": len(gazette),
        },
    }
