"""
Stage 3: Build training features from profile + accounts data.

This is where the magic happens. We take:
  - Profile data (5.6M companies with status, age, SIC, charges, filing dates)
  - Parsed accounts data (balance sheet figures per company per year)

And build trajectory features:
  - net_assets_latest: most recent net assets value
  - net_assets_trend: year-on-year change in net assets
  - net_assets_trend_accel: is the decline accelerating or decelerating
  - years_declining: consecutive years of net asset decline
  - equity_negative: has equity turned negative
  - current_ratio: current assets / current liabilities
  - current_ratio_trend: is the current ratio improving or worsening
  - leverage_ratio: total liabilities / total assets
  - cash_position: cash relative to current liabilities
  - asset_shrinkage: are total assets declining
  - liability_growth: are liabilities growing

These trajectory features capture FINANCIAL deterioration that the
profile-only model can't see. A company filing on time with no charges
but whose net assets have gone from £100K → £30K → -£10K is a very
different risk than one going £100K → £120K → £150K.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


def load_profiles(csv_path):
    """Load and process the BasicCompanyData CSV.

    Returns DataFrame with one row per company plus profile features.
    (Same logic as train_profile.py but returns more columns)
    """
    print(f"\n[3.1] Loading profile data: {csv_path}")
    now = datetime(2026, 2, 1)
    chunk_size = 200000
    all_rows = []
    total = 0

    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, low_memory=False,
                                                    encoding="latin-1",
                                                    chunksize=chunk_size)):
        total += len(chunk)

        # Normalise column names
        col_map = {c: c.strip().replace(" ", "").replace(".", "_") for c in chunk.columns}
        chunk.rename(columns=col_map, inplace=True)

        # Find columns flexibly
        def fc(cands):
            for c in cands:
                m = [col for col in chunk.columns if c.lower() in col.lower()]
                if m:
                    return m[0]
            return None

        num_col = fc(["CompanyNumber"])
        status_col = fc(["CompanyStatus"])
        inc_col = fc(["IncorporationDate"])
        cat_col = fc(["CompanyCategory"])
        acc_cat_col = fc(["AccountCategory"])
        sic_col = fc(["SicText_1", "SICCode_SicText_1"])
        mort_col = fc(["NumMortCharges", "Mortgages_NumMortCharges"])
        mort_out_col = fc(["NumMortOutstanding", "Mortgages_NumMortOutstanding"])
        acc_due_col = fc(["NextDueDate", "Accounts_NextDueDate"])
        acc_made_col = fc(["LastMadeUpDate", "Accounts_LastMadeUpDate"])
        conf_due_col = fc(["ConfStmtNextDueDate"])

        if not status_col or not inc_col or not num_col:
            if chunk_num == 0:
                print("[!!] Missing essential columns")
                return None
            continue

        if chunk_num == 0:
            print(f"  Columns detected: num={num_col}, status={status_col}")

        # Company number
        chunk["company_number"] = chunk[num_col].astype(str).str.strip().str.zfill(8)

        # Status → failure label
        status = chunk[status_col].fillna("").str.lower()
        chunk["failed"] = status.isin([
            "liquidation", "receivership", "administration",
            "voluntary arrangement", "insolvency proceedings"
        ]).astype(int)

        # Also count dissolved-with-outstanding-charges as failure
        if mort_out_col:
            dwd = (status == "dissolved") & (
                pd.to_numeric(chunk[mort_out_col], errors="coerce").fillna(0) > 0
            )
            chunk.loc[dwd, "failed"] = 1

        chunk["company_status"] = status

        # Age
        min_date = pd.Timestamp("1900-01-01")
        max_date = pd.Timestamp("2026-02-01")
        inc_dates = pd.to_datetime(chunk[inc_col], dayfirst=True, errors="coerce")
        inc_dates = inc_dates.where((inc_dates >= min_date) & (inc_dates <= max_date))
        chunk["age_years"] = (pd.Timestamp(now) - inc_dates).dt.days / 365.25

        # SIC code
        if sic_col:
            chunk["sic_2digit"] = (
                chunk[sic_col].fillna("").astype(str)
                .str.extract(r"(\d{2})", expand=False)
                .fillna("0").astype(int)
            )
        else:
            chunk["sic_2digit"] = 0

        # Account category flags
        if acc_cat_col:
            ar = chunk[acc_cat_col].fillna("").str.lower()
            chunk["acc_dormant"] = ar.str.contains("dormant").astype(int)
            chunk["acc_micro"] = ar.str.contains("micro").astype(int)
            chunk["acc_small"] = ar.str.contains("small").astype(int)
            chunk["acc_full"] = ar.str.contains("full|group|audit").astype(int)
        else:
            for c in ["acc_dormant", "acc_micro", "acc_small", "acc_full"]:
                chunk[c] = 0

        # Company type
        if cat_col:
            cr = chunk[cat_col].fillna("").str.lower()
            chunk["is_plc"] = cr.str.contains("public").astype(int)
            chunk["is_llp"] = cr.str.contains("llp|partnership").astype(int)
        else:
            chunk["is_plc"] = 0
            chunk["is_llp"] = 0

        # Charges
        chunk["num_charges"] = pd.to_numeric(
            chunk.get(mort_col, pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        chunk["num_outstanding"] = pd.to_numeric(
            chunk.get(mort_out_col, pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)

        # Accounts overdue
        if acc_due_col:
            due = pd.to_datetime(chunk[acc_due_col], dayfirst=True, errors="coerce")
            due = due.where((due >= min_date) & (due <= max_date))
            chunk["accounts_overdue"] = (due < pd.Timestamp(now)).fillna(False).astype(int)
        else:
            chunk["accounts_overdue"] = 0

        # Days since last filing
        if acc_made_col:
            made = pd.to_datetime(chunk[acc_made_col], dayfirst=True, errors="coerce")
            made = made.where((made >= min_date) & (made <= max_date))
            chunk["days_since_filing"] = (pd.Timestamp(now) - made).dt.days.clip(0, 3650).fillna(999)
        else:
            chunk["days_since_filing"] = 999

        # Confirmation statement overdue
        if conf_due_col:
            conf_due = pd.to_datetime(chunk[conf_due_col], dayfirst=True, errors="coerce")
            conf_due = conf_due.where((conf_due >= min_date) & (conf_due <= max_date))
            chunk["conf_overdue"] = (conf_due < pd.Timestamp(now)).fillna(False).astype(int)
        else:
            chunk["conf_overdue"] = 0

        # High risk sector
        chunk["high_risk_sector"] = chunk["sic_2digit"].isin([41, 42, 43, 56, 68, 47, 49]).astype(int)

        # Keep relevant columns
        keep = [
            "company_number", "failed", "company_status",
            "age_years", "sic_2digit", "acc_dormant", "acc_micro", "acc_small", "acc_full",
            "is_plc", "is_llp", "num_charges", "num_outstanding",
            "accounts_overdue", "days_since_filing", "conf_overdue", "high_risk_sector",
        ]
        valid = chunk[chunk["age_years"].notna() & (chunk["age_years"] > 0)][keep]
        all_rows.append(valid)

        if chunk_num % 5 == 0:
            print(f"    {total:,} rows processed...")

    print(f"[3.1] Combining {len(all_rows)} chunks...")
    df = pd.concat(all_rows, ignore_index=True)
    del all_rows

    print(f"  Total companies: {len(df):,}")
    print(f"  Failed: {df['failed'].sum():,} ({df['failed'].mean() * 100:.2f}%)")
    return df


def build_financial_features(accounts_parquet):
    """Build per-company financial trajectory features from accounts data.

    For each company, we look at all available years and compute:
    - Latest values
    - Trends (direction and rate of change)
    - Consecutive decline counts
    - Ratios and their trends
    """
    print(f"\n[3.2] Loading parsed accounts: {accounts_parquet}")
    accts = pd.read_parquet(accounts_parquet)
    print(f"  Rows: {len(accts):,} | Companies: {accts['company_number'].nunique():,}")

    # Ensure company_number is zero-padded string
    accts["company_number"] = accts["company_number"].astype(str).str.strip().str.zfill(8)

    # Sort by company + year for trajectory calculation
    accts = accts.sort_values(["company_number", "year"]).reset_index(drop=True)

    print("\n[3.3] Computing per-company financial features...")
    features = []
    grouped = accts.groupby("company_number")
    total_groups = len(grouped)

    for i, (co_num, group) in enumerate(grouped):
        if i % 100000 == 0 and i > 0:
            print(f"    {i:,}/{total_groups:,} companies processed...")

        row = {"company_number": co_num}
        years_data = group.sort_values("year")
        n_years = len(years_data)

        # ── Latest values ──
        latest = years_data.iloc[-1]
        row["fin_years_available"] = n_years
        row["latest_year"] = int(latest.get("year", 0))

        for field in ["net_assets", "total_assets", "current_assets",
                      "current_liabilities", "cash", "retained_earnings",
                      "turnover", "net_profit", "employees",
                      "non_current_liabilities", "fixed_assets",
                      "share_capital", "total_liabilities"]:
            val = latest.get(field)
            if pd.notna(val):
                row[f"fin_{field}"] = float(val)

        # ── Net assets trajectory ──
        na_series = years_data["net_assets"].dropna()
        if len(na_series) >= 2:
            vals = na_series.values
            # Year-on-year changes
            changes = np.diff(vals)
            row["na_latest_change"] = float(changes[-1])
            row["na_avg_change"] = float(np.mean(changes))

            # Is it declining?
            row["na_declining"] = int(changes[-1] < 0)

            # Consecutive years of decline
            consecutive = 0
            for c in reversed(changes):
                if c < 0:
                    consecutive += 1
                else:
                    break
            row["na_years_declining"] = consecutive

            # Rate of change (percentage)
            if abs(vals[-2]) > 100:  # Avoid division by tiny numbers
                row["na_pct_change"] = float(changes[-1] / abs(vals[-2]) * 100)
            else:
                row["na_pct_change"] = 0.0

            # Acceleration: is decline getting worse?
            if len(changes) >= 2:
                row["na_accelerating"] = int(changes[-1] < changes[-2])

            # Has equity turned negative?
            row["na_negative"] = int(vals[-1] < 0)
            row["na_was_positive_now_negative"] = int(vals[-1] < 0 and vals[0] > 0)

        # ── Current ratio trajectory ──
        ca = years_data["current_assets"].dropna()
        cl = years_data["current_liabilities"].dropna()
        if len(ca) >= 1 and len(cl) >= 1:
            # Latest current ratio
            ca_latest = latest.get("current_assets")
            cl_latest = latest.get("current_liabilities")
            if pd.notna(ca_latest) and pd.notna(cl_latest) and abs(cl_latest) > 0:
                row["fin_current_ratio"] = float(ca_latest / abs(cl_latest))

        # Compute current ratio per year for trend
        cr_values = []
        for _, yr_row in years_data.iterrows():
            ca_v = yr_row.get("current_assets")
            cl_v = yr_row.get("current_liabilities")
            if pd.notna(ca_v) and pd.notna(cl_v) and abs(cl_v) > 0:
                cr_values.append(ca_v / abs(cl_v))
        if len(cr_values) >= 2:
            row["cr_trend"] = float(cr_values[-1] - cr_values[-2])
            row["cr_declining"] = int(cr_values[-1] < cr_values[-2])

        # ── Cash position ──
        cash_v = latest.get("cash")
        cl_v = latest.get("current_liabilities")
        if pd.notna(cash_v) and pd.notna(cl_v) and abs(cl_v) > 0:
            row["fin_cash_ratio"] = float(cash_v / abs(cl_v))

        # ── Leverage ratio ──
        ta = latest.get("total_assets")
        tl = latest.get("total_liabilities")
        if pd.notna(ta) and pd.notna(tl) and abs(ta) > 0:
            row["fin_leverage"] = float(abs(tl) / abs(ta))

        # ── Asset shrinkage ──
        ta_series = years_data["total_assets"].dropna()
        if len(ta_series) >= 2:
            ta_vals = ta_series.values
            row["ta_shrinking"] = int(ta_vals[-1] < ta_vals[-2])
            if abs(ta_vals[-2]) > 100:
                row["ta_pct_change"] = float((ta_vals[-1] - ta_vals[-2]) / abs(ta_vals[-2]) * 100)

        # ── Retained earnings trajectory ──
        re_series = years_data["retained_earnings"].dropna()
        if len(re_series) >= 2:
            re_vals = re_series.values
            row["re_declining"] = int(re_vals[-1] < re_vals[-2])
            row["re_negative"] = int(re_vals[-1] < 0)

        # ── Turnover trajectory (if available — small/full accounts) ──
        to_series = years_data["turnover"].dropna()
        if len(to_series) >= 2:
            to_vals = to_series.values
            row["to_declining"] = int(to_vals[-1] < to_vals[-2])
            if abs(to_vals[-2]) > 100:
                row["to_pct_change"] = float((to_vals[-1] - to_vals[-2]) / abs(to_vals[-2]) * 100)

        # ── Employees trajectory ──
        emp_series = years_data["employees"].dropna()
        if len(emp_series) >= 2:
            emp_vals = emp_series.values
            row["emp_declining"] = int(emp_vals[-1] < emp_vals[-2])

        features.append(row)

    print(f"  Computed features for {len(features):,} companies")
    fin_df = pd.DataFrame(features)

    # Summary
    print(f"\n[3.3] Financial feature summary:")
    print(f"  Companies with net_assets data: {fin_df['fin_net_assets'].notna().sum():,}")
    print(f"  Companies with trajectory (2+ years): {(fin_df['na_latest_change'].notna()).sum():,}")
    print(f"  Companies with turnover: {fin_df.get('fin_turnover', pd.Series()).notna().sum():,}")
    print(f"  Companies with current ratio: {fin_df.get('fin_current_ratio', pd.Series()).notna().sum():,}")

    return fin_df


def build_training_data(profile_csv, accounts_parquet, output_path):
    """Join profile + financial features and produce training dataset."""

    # Load profile data
    profiles = load_profiles(profile_csv)
    if profiles is None:
        return

    # Load and compute financial features
    fin_features = build_financial_features(accounts_parquet)

    # Join on company_number
    print(f"\n[3.4] Joining profile ({len(profiles):,}) with financials ({len(fin_features):,})...")
    merged = profiles.merge(fin_features, on="company_number", how="left")

    # Stats on join
    has_fin = merged["fin_net_assets"].notna().sum()
    has_trajectory = merged["na_latest_change"].notna().sum()
    print(f"  Companies with financial data: {has_fin:,} ({has_fin / len(merged) * 100:.1f}%)")
    print(f"  Companies with trajectory: {has_trajectory:,} ({has_trajectory / len(merged) * 100:.1f}%)")

    # Fill NaN financial features with sentinel values
    # The model will learn that NaN (= no accounts data) is itself informative
    fin_cols = [c for c in merged.columns if c.startswith(("fin_", "na_", "cr_", "ta_", "re_", "to_", "emp_"))]
    for col in fin_cols:
        if merged[col].dtype in ["float64", "float32", "int64"]:
            # Use -999 as sentinel for "no data" — model can learn this
            merged[col] = merged[col].fillna(-999)

    # Boolean flags: NaN → 0
    bool_cols = [c for c in merged.columns if c.endswith(("_declining", "_negative", "_accelerating", "_shrinking"))]
    for col in bool_cols:
        merged[col] = merged[col].fillna(0).astype(int)

    # Add has_accounts flag
    merged["has_accounts_data"] = (merged["fin_net_assets"] != -999).astype(int)
    merged["has_trajectory"] = (merged["na_latest_change"] != -999).astype(int)

    # Drop non-feature columns
    merged = merged.drop(columns=["company_number", "company_status"], errors="ignore")

    # Fill any remaining NaN
    merged = merged.fillna(0)

    print(f"\n[3.5] Final training dataset:")
    print(f"  Total companies: {len(merged):,}")
    print(f"  Features: {merged.shape[1] - 1}")  # -1 for target
    print(f"  Failed: {merged['failed'].sum():,} ({merged['failed'].mean() * 100:.2f}%)")
    print(f"  With accounts: {merged['has_accounts_data'].sum():,}")
    print(f"  With trajectory: {merged['has_trajectory'].sum():,}")

    # Save
    print(f"\n[3.6] Saving to {output_path}...")
    merged.to_parquet(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  [ok] Saved ({size_mb:.1f} MB)")

    # Print feature list
    feature_cols = [c for c in merged.columns if c != "failed"]
    print(f"\n  All {len(feature_cols)} features:")
    for c in sorted(feature_cols):
        print(f"    {c}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("profile_csv")
    p.add_argument("accounts_parquet")
    p.add_argument("output")
    args = p.parse_args()
    build_training_data(args.profile_csv, args.accounts_parquet, args.output)
