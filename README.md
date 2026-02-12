# Clearview v0.3

Company financial health assessment for UK companies. Uses Companies House API data, Altman Z'-Score modelling, and SIC code benchmarks.

## Quick Start

```bash
cd clearview
pip install -r requirements.txt
python server.py
```

Then open **http://localhost:5001** in your browser.

## How It Works

1. **Search** — queries the Companies House search API
2. **Profile** — fetches company details, officers, PSC, charges
3. **Financials** — downloads iXBRL accounts filings and parses financial data
4. **Analysis** — runs Altman Z'-Score (real or modelled) + Clearview scoring
5. **Display** — presents credit rating, risk signals, and plain English verdict

## Scoring Tiers

| Data Available | Method | Rating Suffix |
|---|---|---|
| Full accounts (all P&L) | Altman Z'-Score | AA, A, BBB etc. |
| Partial P&L (turnover + profit) | Hybrid Z' (EBIT derived) | AA~, A~ etc. |
| No P&L but has employees | Modelled Z' (SIC benchmarks) | AA~, A~ etc. |
| Balance sheet only | Clearview score | AA*, A* etc. |

## API Key

The Companies House API key is set in `server.py`. To change it:

```bash
export CH_API_KEY="your-key-here"
python server.py
```

## Files

- `server.py` — Flask web server + API routes
- `ch_api.py` — Companies House REST API client
- `accounts_parser.py` — iXBRL financial data extractor
- `static/index.html` — React frontend (single file, no build step)

## Notes

- iXBRL parsing works for ~75% of accounts filings. PDF-only filers will show "no financial data"
- Rate limit: 600 requests per 5 minutes (Companies House)
- Financial data is cached in memory per session
- SIC benchmarks are currently hardcoded; in production these would be generated from bulk filing data
