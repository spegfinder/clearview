"""
═══════════════════════════════════════════════════════════════
  Clearview ML Pipeline — Full Accounts Training
═══════════════════════════════════════════════════════════════

  Trains a gradient-boosted insolvency prediction model using:
    - Company profile data (5.6M companies)
    - Parsed balance sheet data from iXBRL filings
    - Multi-year financial trajectory features

  Stages:
    1. Download bulk accounts from Companies House
    2. Parse iXBRL → extract balance sheets → Parquet
    3. Build features (profile + financial trajectories)
    4. Train model + calibrate + export

  Usage:
    source ~/clearview_env/bin/activate
    python pipeline.py [--stage N] [--workers 8] [--months 24]

  Requirements:
    - ~200GB disk (accounts zips + extracted)
    - 16GB RAM recommended
    - 4-8 CPU cores for parallel parsing
    - ~4-8 hours total runtime

═══════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
from datetime import datetime

# ── Config ──
DATA_DIR = os.path.expanduser("~/clearview/data")
OUTPUT_DIR = os.path.expanduser("~/clearview/output")
ACCOUNTS_RAW = os.path.join(DATA_DIR, "accounts_raw")
ACCOUNTS_PARSED = os.path.join(DATA_DIR, "accounts_parsed")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")

PARSED_PARQUET = os.path.join(ACCOUNTS_PARSED, "all_accounts.parquet")
FEATURES_PARQUET = os.path.join(OUTPUT_DIR, "training_features.parquet")
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, "clearview_model_v2.json")


def stage_1_download(months=24):
    """Download bulk accounts zips from Companies House."""
    from download_accounts import download_bulk_accounts
    download_bulk_accounts(ACCOUNTS_RAW, months=months)


def stage_2_parse(workers=8):
    """Parse all iXBRL files into a single Parquet dataset."""
    from parse_accounts_bulk import parse_all_accounts
    parse_all_accounts(ACCOUNTS_RAW, PARSED_PARQUET, workers=workers)


def stage_3_features():
    """Build training features from profile + accounts data."""
    from build_features import build_training_data
    profile_csv = None
    for f in os.listdir(PROFILES_DIR):
        if f.startswith("Basic") and f.endswith(".csv"):
            profile_csv = os.path.join(PROFILES_DIR, f)
            break
    if not profile_csv:
        print("[!!] No BasicCompanyData CSV found in", PROFILES_DIR)
        print("     Download from http://download.companieshouse.gov.uk/en_output.html")
        sys.exit(1)
    build_training_data(profile_csv, PARSED_PARQUET, FEATURES_PARQUET)


def stage_4_train():
    """Train the enhanced model."""
    from train_model_v2 import train_and_export
    train_and_export(FEATURES_PARQUET, MODEL_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description="Clearview ML Pipeline")
    parser.add_argument("--stage", type=int, default=0,
                        help="Start from stage N (1=download, 2=parse, 3=features, 4=train, 0=all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for parsing")
    parser.add_argument("--months", type=int, default=24,
                        help="Months of accounts data to download")
    args = parser.parse_args()

    os.makedirs(ACCOUNTS_RAW, exist_ok=True)
    os.makedirs(ACCOUNTS_PARSED, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()
    print("=" * 60)
    print("  Clearview ML Pipeline — Full Accounts Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workers: {args.workers} | Months: {args.months}")
    print("=" * 60)

    stages = [
        (1, "Download bulk accounts", lambda: stage_1_download(args.months)),
        (2, "Parse iXBRL → Parquet", lambda: stage_2_parse(args.workers)),
        (3, "Build training features", stage_3_features),
        (4, "Train model", stage_4_train),
    ]

    for num, name, fn in stages:
        if args.stage > num:
            continue
        print(f"\n{'─' * 60}")
        print(f"  STAGE {num}: {name}")
        print(f"{'─' * 60}")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  ✓ Stage {num} complete ({elapsed / 60:.1f} minutes)")

    total = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {total / 60:.1f} minutes total")
    print(f"  Model: {MODEL_OUTPUT}")
    print(f"  Copy to your project: scp {MODEL_OUTPUT} your-server:clearview/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
