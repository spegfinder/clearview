# Clearview ML Pipeline — Full Accounts Training

Train an insolvency prediction model on **5.6M UK companies** using both profile data and multi-year balance sheet trajectories from Companies House iXBRL filings.

## What it does

| Stage | What | Time | Output |
|-------|------|------|--------|
| 1. Download | Bulk accounts zips from Companies House | ~2-3 hours | ~100GB of iXBRL files |
| 2. Parse | Extract balance sheets from iXBRL | ~2-3 hours | `all_accounts.parquet` |
| 3. Features | Join profiles + compute trajectories | ~30 min | `training_features.parquet` |
| 4. Train | Gradient boosting + calibration | ~30 min | `clearview_model_v2.json` |

**Total: ~5-7 hours, ~£2-3 in EC2 costs.**

## Features (v2 vs v1)

**v1 (profile only, 13 features):** accounts overdue, age, charges, SIC sector, filing gap, company type. AUC: 0.967

**v2 adds ~20 financial trajectory features:**
- `fin_net_assets`, `na_latest_change`, `na_years_declining` — net asset trajectory
- `fin_current_ratio`, `cr_trend` — liquidity trend
- `fin_cash_ratio`, `fin_leverage` — balance sheet health
- `na_negative`, `na_was_positive_now_negative` — equity turning negative
- `ta_shrinking`, `re_declining` — asset/earnings deterioration
- `to_declining`, `to_pct_change` — revenue trajectory (small/full accounts)
- `has_accounts_data`, `has_trajectory` — model knows when data is missing

**Expected AUC: 0.98+** — the key gain is catching companies that file on time but are financially deteriorating.

## Quick Start

### 1. Launch EC2

```
Instance: c5.2xlarge (8 vCPU, 16GB RAM) — $0.34/hr
Storage: 200GB gp3 EBS
AMI: Ubuntu 22.04 LTS
```

### 2. Upload and run

```bash
# Upload pipeline files
scp -r ec2_pipeline/* ubuntu@your-ec2:~/clearview/

# SSH in
ssh ubuntu@your-ec2

# Setup
cd ~/clearview
chmod +x setup_ec2.sh
./setup_ec2.sh

# Run full pipeline
source ~/clearview_env/bin/activate
python pipeline.py --workers 8 --months 24
```

### 3. Resume from a stage (if interrupted)

```bash
python pipeline.py --stage 2  # Skip download, start from parse
python pipeline.py --stage 3  # Skip download+parse, start from features
python pipeline.py --stage 4  # Just retrain
```

### 4. Deploy the model

```bash
# Copy model back to your machine
scp ubuntu@your-ec2:~/clearview/output/clearview_model_v2.json ~/Downloads/clearview/

# Deploy to Railway
cd ~/Downloads/clearview
cp clearview_model_v2.json clearview_model.json
git add . && git commit -m "v2 model with accounts data" && git push
```

The server's `distress_predictor.py` auto-detects v2 and uses financial features when available.

## File Structure

```
ec2_pipeline/
├── setup_ec2.sh              # EC2 setup (packages, dirs, profile CSV)
├── pipeline.py               # Main orchestrator
├── download_accounts.py      # Stage 1: Bulk download from Companies House
├── parse_accounts_bulk.py    # Stage 2: iXBRL → Parquet
├── build_features.py         # Stage 3: Profile + accounts → training data
├── train_model_v2.py         # Stage 4: Train + calibrate + export
└── README.md                 # This file
```

## Disk Space Requirements

| What | Size |
|------|------|
| Accounts zips (24 months) | ~60-80GB |
| Extracted iXBRL files | ~100-150GB |
| Parsed Parquet | ~2-5GB |
| Profile CSV | ~1.5GB |
| Training features | ~500MB |
| Final model | ~10KB |

After training, you can delete the raw accounts and zips. The Parquet files are worth keeping for retraining.

## Monthly Retraining

For ongoing updates:

```bash
# Download just the latest month
python pipeline.py --stage 1 --months 1

# Re-parse everything (incremental would be better but this is simpler)
python pipeline.py --stage 2

# Rebuild features and retrain
python pipeline.py --stage 3
```

Or schedule as a cron job / Lambda trigger once a month.

## Notes

- Companies House bulk accounts data: http://download.companieshouse.gov.uk/en_accountsdata.html
- Profile CSV: http://download.companieshouse.gov.uk/en_output.html
- The model JSON format is backwards-compatible with v1
- Even micro-entity accounts (net assets + equity only) contribute useful trajectory features
- The `-999` sentinel value tells the model "no financial data available" — this is itself informative
