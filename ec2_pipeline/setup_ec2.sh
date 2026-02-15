#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Clearview ML Pipeline — EC2 Setup
# ═══════════════════════════════════════════════════════════
#
#  Recommended instance: c5.2xlarge (8 vCPU, 16GB RAM)
#  Storage: 200GB gp3 EBS volume
#  OS: Ubuntu 22.04 LTS or Amazon Linux 2023
#  Cost: ~$0.34/hr → full pipeline ~$2-3
#
#  Usage:
#    chmod +x setup_ec2.sh
#    ./setup_ec2.sh
#
# ═══════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════"
echo "  Clearview ML Pipeline — Setup"
echo "═══════════════════════════════════════════"

# ── System dependencies ──
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv unzip wget htop

# ── Python virtual environment ──
echo "[2/4] Creating Python environment..."
python3 -m venv ~/clearview_env
source ~/clearview_env/bin/activate

pip install --upgrade pip
pip install \
    pandas==2.2.1 \
    numpy==1.26.4 \
    scikit-learn==1.4.1 \
    beautifulsoup4==4.12.3 \
    lxml==5.1.0 \
    pyarrow==15.0.0 \
    tqdm==4.66.2 \
    requests==2.31.0 \
    joblib==1.3.2

echo "[3/4] Creating directories..."
mkdir -p ~/clearview/data/accounts_raw
mkdir -p ~/clearview/data/accounts_parsed
mkdir -p ~/clearview/data/profiles
mkdir -p ~/clearview/output

# ── Download bulk company profile CSV ──
echo "[4/4] Downloading company profile data..."
cd ~/clearview/data/profiles
if [ ! -f BasicCompanyDataAsOneFile*.csv ]; then
    wget -q "http://download.companieshouse.gov.uk/en_output.html" -O index.html 2>/dev/null || true
    # Direct download link
    wget --content-disposition "http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-02-01.zip" \
         -O profiles.zip 2>/dev/null || \
    wget --content-disposition "http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-01-01.zip" \
         -O profiles.zip 2>/dev/null || \
    echo "[!!] Could not auto-download profile CSV."
    echo "     Go to http://download.companieshouse.gov.uk/en_output.html"
    echo "     Download BasicCompanyDataAsOneFile and place in ~/clearview/data/profiles/"

    if [ -f profiles.zip ]; then
        unzip profiles.zip
        rm profiles.zip
        echo "[ok] Profile CSV extracted"
    fi
else
    echo "[ok] Profile CSV already present"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    source ~/clearview_env/bin/activate"
echo "    cd ~/clearview"
echo "    python pipeline.py"
echo "═══════════════════════════════════════════"
