"""
Stage 2: Parse bulk iXBRL accounts into structured data.

Reads all iXBRL/HTML/XML files from the extracted accounts directories
and extracts balance sheet data into a single Parquet file.

For each filing we extract:
  - company_number (from filename or content)
  - period_end (balance sheet date)
  - net_assets
  - total_assets
  - current_assets
  - current_liabilities (creditors due within 1 year)
  - non_current_liabilities
  - cash
  - retained_earnings
  - share_capital
  - fixed_assets
  - employees
  - turnover (if available — small/full accounts)
  - net_profit (if available)

Output: Parquet file with one row per company per filing period.
"""

import os
import re
import sys
import glob
import json
import time
import traceback
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  XBRL Concept Mappings
# ═══════════════════════════════════════════════════════════

# Maps internal field name → list of XBRL concept suffixes to match
CONCEPT_MAP = {
    "net_assets": [
        "NetAssetsLiabilities", "NetAssets", "TotalNetAssets",
        "NetAssetsIncludingPensionAssetLiability",
    ],
    "total_assets": [
        "TotalAssets",
    ],
    "current_assets": [
        "CurrentAssets", "TotalCurrentAssets",
    ],
    "fixed_assets": [
        "FixedAssets", "NonCurrentAssets",
    ],
    "current_liabilities": [
        "CreditorsDueWithinOneYear", "CurrentLiabilities",
        "CreditorAmountsFallingDueWithinOneYear",
        "CreditorAmountsFallingDueWithinOneYear",
    ],
    "non_current_liabilities": [
        "CreditorsDueAfterOneYear", "NonCurrentLiabilities",
        "CreditorsAmountsFallingDueAfterMoreThanOneYear",
        "CreditorAmountsFallingDueAfterOneYear",
    ],
    "cash": [
        "CashBankInHand", "CashCashEquivalents",
        "CashAtBankInHand", "CashBankOnHand",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedLosses",
        "ProfitLossAccountReserve", "RetainedEarnings",
    ],
    "share_capital": [
        "CalledUpShareCapital", "ShareCapital",
    ],
    "employees": [
        "AverageNumberEmployeesDuringPeriod",
        "EntityAverageNumberOfEmployees",
        "AverageNumberOfEmployees",
    ],
    "turnover": [
        "Turnover", "TurnoverRevenue", "Revenue",
        "TurnoverGrossIncome",
    ],
    "net_profit": [
        "ProfitLossForPeriod", "ProfitLossForYear",
        "ProfitLossForFinancialYear", "ProfitLoss",
    ],
    "total_liabilities": [
        "TotalLiabilities",
    ],
}

# Build reverse lookup: concept_suffix → field_name
_CONCEPT_LOOKUP = {}
for field, concepts in CONCEPT_MAP.items():
    for c in concepts:
        _CONCEPT_LOOKUP[c.lower()] = field


# ═══════════════════════════════════════════════════════════
#  iXBRL Parser — Lightweight & Fast
# ═══════════════════════════════════════════════════════════

def extract_company_number(filepath, content_head):
    """Extract company number from filename or file content."""
    # Try filename first — CH bulk files often named like: Prod224_1234_00012345_20240331.html
    basename = os.path.basename(filepath)
    # Match 8-digit company number pattern
    m = re.search(r"_(\d{8})_", basename)
    if m:
        return m.group(1)

    # Also try: CompanyNumber_12345678 in filename
    m = re.search(r"(\d{8})", basename)
    if m:
        return m.group(1)

    # Try content — look for company number in XBRL entity identifier
    m = re.search(r"CompanyNumber[>\s:]+(\d{6,8})", content_head, re.I)
    if m:
        return m.group(1).zfill(8)

    # Try UKCompaniesHouseRegisteredNumber or similar
    m = re.search(r"RegisteredNumber[>\s:]+(\d{6,8})", content_head, re.I)
    if m:
        return m.group(1).zfill(8)

    # Look in entity identifier tags
    m = re.search(r'<[^>]*identifier[^>]*>(\d{6,8})<', content_head, re.I)
    if m:
        return m.group(1).zfill(8)

    return None


def parse_value(text, scale_str="0", sign=""):
    """Parse a numeric value from iXBRL text."""
    cleaned = text.replace(",", "").replace(" ", "").replace("\xa0", "")
    has_brackets = "(" in text and ")" in text
    cleaned = re.sub(r"[^\d.\-]", "", cleaned)

    if not cleaned or cleaned in (".", "-"):
        return None

    try:
        val = float(cleaned)
    except ValueError:
        return None

    # Apply scale (e.g., scale="3" means thousands, "6" means millions)
    try:
        scale = int(scale_str)
        if scale != 0:
            val *= 10 ** scale
    except (ValueError, TypeError):
        pass

    # Apply sign
    if sign == "-" or has_brackets:
        val = -abs(val)

    return val


def parse_ixbrl_fast(filepath):
    """Fast iXBRL parser optimised for bulk processing.

    Returns list of dicts: [{company_number, period_end, field: value, ...}]
    One dict per reporting period found in the file.
    """
    try:
        # Read file
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()

        if len(content) < 200:
            return []

        # Extract company number from first 5KB
        company_number = extract_company_number(filepath, content[:5000])
        if not company_number:
            return []

        # Use BeautifulSoup for robust parsing
        soup = BeautifulSoup(content, "lxml")

        # ── Parse contexts (period definitions) ──
        contexts = {}
        for ctx in soup.find_all(re.compile(r"context$", re.I)):
            ctx_id = ctx.get("id")
            if not ctx_id:
                continue

            period = ctx.find(re.compile(r"period$", re.I))
            if not period:
                continue

            info = {}
            instant = period.find(re.compile(r"instant$", re.I))
            end = period.find(re.compile(r"enddate$", re.I))

            if instant:
                info["date"] = instant.get_text(strip=True)[:10]
            elif end:
                info["date"] = end.get_text(strip=True)[:10]
            else:
                continue

            # Skip contexts with dimensions (consolidated, segment etc)
            segment = ctx.find(re.compile(r"segment$", re.I))
            if segment and segment.find(re.compile(r"explicitmember|typedmember", re.I)):
                info["has_dimension"] = True

            contexts[ctx_id] = info

        if not contexts:
            return []

        # ── Parse facts ──
        # Collect all values by (context_date, field)
        by_period = {}  # date → {field: value}

        for tag in soup.find_all(re.compile(r"nonfraction$", re.I)):
            name = tag.get("name", "")
            ctx_ref = tag.get("contextref") or tag.get("contextRef", "")

            # Get local concept name
            local_name = name.split(":")[-1] if ":" in name else name
            field = _CONCEPT_LOOKUP.get(local_name.lower())
            if not field:
                continue

            # Resolve context
            ctx = contexts.get(ctx_ref)
            if not ctx or ctx.get("has_dimension"):
                continue

            date = ctx["date"]

            # Parse value
            text = tag.get_text(strip=True)
            sign = tag.get("sign", "")
            scale = tag.get("scale", "0")
            val = parse_value(text, scale, sign)
            if val is None:
                continue

            if date not in by_period:
                by_period[date] = {"company_number": company_number, "period_end": date}

            # Only keep first value for each field per period (avoid double-counting)
            if field not in by_period[date]:
                by_period[date][field] = val

        results = list(by_period.values())

        # Filter out periods with almost no data
        results = [r for r in results if len(r) > 3]  # At least company_number + period_end + 1 financial field

        return results

    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
#  Bulk Processing
# ═══════════════════════════════════════════════════════════

def find_all_account_files(base_dir):
    """Recursively find all iXBRL/HTML/XML account files."""
    extensions = (".html", ".htm", ".xml", ".xhtml")
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                files.append(os.path.join(root, fname))
    return files


def process_batch(file_batch):
    """Process a batch of files (for multiprocessing)."""
    results = []
    for filepath in file_batch:
        rows = parse_ixbrl_fast(filepath)
        results.extend(rows)
    return results


def parse_all_accounts(accounts_dir, output_parquet, workers=8):
    """Parse all iXBRL files into a single Parquet dataset.

    Args:
        accounts_dir: Directory containing extracted accounts folders
        output_parquet: Output Parquet file path
        workers: Number of parallel worker processes
    """
    print(f"\n[Stage 2] Parsing iXBRL accounts")
    print(f"  Source: {accounts_dir}")
    print(f"  Output: {output_parquet}")
    print(f"  Workers: {workers}")

    # Check if already parsed
    if os.path.exists(output_parquet):
        existing = pd.read_parquet(output_parquet)
        print(f"  [exists] {len(existing):,} rows already parsed")
        resp = input("  Re-parse? (y/N): ").strip().lower()
        if resp != "y":
            return

    # Find all files
    print("\n[2.1] Scanning for account files...")
    all_files = find_all_account_files(accounts_dir)
    # Filter out non-account files (readmes, indexes, etc)
    all_files = [f for f in all_files if os.path.getsize(f) > 500]
    print(f"  Found {len(all_files):,} files to parse")

    if not all_files:
        print("[!!] No account files found. Check that Stage 1 completed.")
        return

    # Split into batches for multiprocessing
    batch_size = 500
    batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    print(f"  Split into {len(batches)} batches of ~{batch_size} files")

    # Process with multiprocessing
    print(f"\n[2.2] Parsing with {workers} workers...")
    all_rows = []
    parsed_count = 0
    failed_count = 0
    t0 = time.time()

    with Pool(workers) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(process_batch, batches),
            total=len(batches),
            desc="  Parsing",
            ncols=80,
            unit="batch",
        ):
            all_rows.extend(batch_results)
            parsed_count += batch_size

            # Progress update every 50 batches
            if len(all_rows) % 25000 == 0 and all_rows:
                elapsed = time.time() - t0
                rate = parsed_count / elapsed
                remaining = (len(all_files) - parsed_count) / max(rate, 1)
                print(f"    {len(all_rows):,} rows extracted | "
                      f"{rate:.0f} files/sec | "
                      f"~{remaining / 60:.0f} min remaining")

    elapsed = time.time() - t0
    print(f"\n[2.3] Parsing complete")
    print(f"  Files processed: {len(all_files):,}")
    print(f"  Rows extracted: {len(all_rows):,}")
    print(f"  Time: {elapsed / 60:.1f} minutes ({len(all_files) / elapsed:.0f} files/sec)")

    if not all_rows:
        print("[!!] No data extracted. Check file formats.")
        return

    # Build DataFrame
    print("\n[2.4] Building DataFrame...")
    df = pd.DataFrame(all_rows)

    # Clean up
    # Parse period_end as date
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end", "company_number"])

    # Derive year
    df["year"] = df["period_end"].dt.year

    # Remove obviously bad data
    df = df[(df["year"] >= 2000) & (df["year"] <= 2026)]

    # Deduplicate: keep one row per company per year (prefer the one with most data)
    df["data_count"] = df.notna().sum(axis=1)
    df = df.sort_values("data_count", ascending=False).drop_duplicates(
        subset=["company_number", "year"], keep="first"
    ).drop(columns=["data_count"])

    # Sort
    df = df.sort_values(["company_number", "year"]).reset_index(drop=True)

    # Stats
    print(f"\n[2.5] Dataset summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique companies: {df['company_number'].nunique():,}")
    print(f"  Year range: {df['year'].min()} — {df['year'].max()}")
    print(f"  Fields with data:")
    financial_fields = [c for c in df.columns if c not in ("company_number", "period_end", "year")]
    for field in financial_fields:
        non_null = df[field].notna().sum()
        pct = non_null / len(df) * 100
        print(f"    {field:30s} {non_null:>10,} ({pct:.1f}%)")

    # Save as Parquet (very efficient for columnar data)
    print(f"\n[2.6] Saving to {output_parquet}...")
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    size_mb = os.path.getsize(output_parquet) / 1024 / 1024
    print(f"  [ok] Saved ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("accounts_dir", help="Directory with extracted accounts")
    parser.add_argument("output", help="Output parquet path")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    parse_all_accounts(args.accounts_dir, args.output, workers=args.workers)
