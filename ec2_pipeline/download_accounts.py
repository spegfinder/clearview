"""
Stage 1: Download bulk accounts data from Companies House.

Companies House publishes monthly zip files containing all accounts filings.
Each zip contains thousands of iXBRL (HTML) and XML files.

Source: http://download.companieshouse.gov.uk/en_accountsdata.html

Each monthly zip is ~2-5GB compressed, ~10-20GB extracted.
We download the most recent N months (default 24).
"""

import os
import re
import sys
import time
import zipfile
import requests
from datetime import datetime, timedelta
from tqdm import tqdm


# Companies House bulk download base URL
BASE_URL = "http://download.companieshouse.gov.uk"

# Three sources of accounts data:
# 1. Monthly (last 12 months): /en_monthlyaccountsdata.html
# 2. Historic monthly (2008-onwards): /historicmonthlyaccountsdata.html
# 3. Daily (last 60 days): /en_accountsdata.html
MONTHLY_INDEX = f"{BASE_URL}/en_monthlyaccountsdata.html"
HISTORIC_INDEX = f"{BASE_URL}/historicmonthlyaccountsdata.html"
DAILY_INDEX = f"{BASE_URL}/en_accountsdata.html"


def get_available_zips():
    """Scrape CH download pages for available accounts zip URLs.

    Checks monthly (last 12 months) and historic (2008+) pages.
    """
    print("[1.1] Fetching available accounts files from Companies House...")
    all_zips = []

    for label, url in [("monthly", MONTHLY_INDEX), ("historic", HISTORIC_INDEX)]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # Parse zip links — typical: Accounts_Monthly_Data-YYYY-MM.zip
            pattern = re.compile(r'href=["\']([^"\']*\.zip)["\']', re.I)
            matches = pattern.findall(resp.text)
            for match in matches:
                full_url = match if match.startswith("http") else f"{BASE_URL}/{match.lstrip('/')}"
                if full_url not in all_zips:
                    all_zips.append(full_url)
            print(f"  [{label}] Found {len(matches)} zip files")
        except Exception as e:
            print(f"  [{label}] Could not fetch: {e}")

    if not all_zips:
        print("[!!] No zip links found, using fallback URLs")
        return generate_fallback_urls()

    # Sort newest first (filenames contain dates)
    all_zips.sort(reverse=True)
    print(f"[ok] Total available: {len(all_zips)} zip files")
    return all_zips


def generate_fallback_urls():
    """Generate expected URLs for recent months if scraping fails."""
    urls = []
    now = datetime.now()
    for months_back in range(0, 36):
        d = now - timedelta(days=months_back * 30)
        # CH monthly accounts naming convention
        date_str = d.strftime("%Y-%m")
        urls.append(f"{BASE_URL}/Accounts_Monthly_Data-{date_str}.zip")
    return urls


def download_file(url, dest_path):
    """Download a file with progress bar and resume support."""
    # Check if already downloaded
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        # Check remote size
        try:
            head = requests.head(url, timeout=10)
            remote_size = int(head.headers.get("content-length", 0))
            if remote_size > 0 and local_size >= remote_size:
                print(f"  [skip] Already downloaded: {os.path.basename(dest_path)}")
                return True
        except:
            pass

    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        filename = os.path.basename(dest_path)

        with open(dest_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {filename}",
                      ncols=80) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True

    except Exception as e:
        print(f"  [!!] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, extract_dir):
    """Extract a zip file, skipping already-extracted files."""
    basename = os.path.splitext(os.path.basename(zip_path))[0]
    target_dir = os.path.join(extract_dir, basename)

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 100:
        print(f"  [skip] Already extracted: {basename}")
        return target_dir

    os.makedirs(target_dir, exist_ok=True)
    print(f"  Extracting {basename}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"  {basename}", ncols=80, unit="files"):
                try:
                    zf.extract(member, target_dir)
                except (zipfile.BadZipFile, KeyError, OSError):
                    continue
        print(f"  [ok] {len(members):,} files extracted to {basename}")
        return target_dir
    except zipfile.BadZipFile:
        print(f"  [!!] Corrupt zip: {zip_path}")
        return None


def download_bulk_accounts(output_dir, months=24):
    """Download and extract bulk accounts data.

    Args:
        output_dir: Where to save zips and extracted files
        months: How many months of data to download
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_dir = os.path.join(output_dir, "zips")
    os.makedirs(zip_dir, exist_ok=True)

    print(f"\n[Stage 1] Downloading bulk accounts — last {months} months")
    print(f"  Output: {output_dir}")
    print(f"  Warning: Each month is 2-5GB. Total may be 50-100GB.\n")

    available = get_available_zips()

    downloaded = 0
    for url in available[:months]:
        filename = os.path.basename(url)
        zip_path = os.path.join(zip_dir, filename)

        print(f"\n[{downloaded + 1}/{months}] {filename}")

        # Download
        if download_file(url, zip_path):
            # Extract
            extract_zip(zip_path, output_dir)
            downloaded += 1

            # Optionally delete zip after extraction to save space
            # os.remove(zip_path)
        else:
            print(f"  [skip] Not available: {filename}")

        if downloaded >= months:
            break

        # Be polite to CH servers
        time.sleep(2)

    print(f"\n[ok] Downloaded and extracted {downloaded} months of accounts data")
    print(f"     Location: {output_dir}")

    # Count total files
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        total_files += len([f for f in files if f.endswith((".html", ".xml", ".htm"))])
    print(f"     Total account files: {total_files:,}")

    return downloaded
