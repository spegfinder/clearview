"""
Clearview Insolvency Prediction Model - Enhanced Training Script
================================================================
Trains on BOTH company profile data AND parsed financial accounts.

Run on your local machine:
    pip install pandas scikit-learn requests beautifulsoup4 python-dateutil
    python train_model.py

Expected runtime: 30-60 minutes (mostly downloading + parsing iXBRL)
"""
import os, io, re, sys, json, zipfile, requests, warnings, numpy as np, pandas as pd, traceback
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import itertools

warnings.filterwarnings("ignore")

def download_company_csv():
    existing = [f for f in os.listdir(".") if f.startswith("Basic") and f.endswith(".csv")]
    if existing:
        print(f"[ok] Found: {existing[0]}")
        return existing[0]
    print("[->] Downloading bulk company data (~400MB)...")
    for mo in range(0, 6):
        target = datetime.now().replace(day=1) - timedelta(days=30*mo)
        url = f"http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-{target.strftime('%Y-%m')}-01.zip"
        try:
            r = requests.head(url, timeout=10)
            if r.status_code != 200: continue
            r = requests.get(url, stream=True, timeout=600)
            total = int(r.headers.get("content-length",0))
            dl = 0
            with open("co_bulk.zip","wb") as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk); dl += len(chunk)
                    if total: print(f"\r  {dl//(1024*1024)}MB / {total//(1024*1024)}MB ({dl*100//total}%)", end="", flush=True)
            print()
            with zipfile.ZipFile("co_bulk.zip") as z:
                csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
                z.extract(csv_name)
                os.remove("co_bulk.zip")
                print(f"[ok] Extracted {csv_name}")
                return csv_name
        except Exception as e:
            print(f"  Failed: {e}"); continue
    print("[!!] Download manually from https://download.companieshouse.gov.uk/en_output.html")
    return None

CONCEPT_MAP = {
    "turnover": ["Turnover","TurnoverRevenue","Revenue","TurnoverGrossIncome"],
    "net_profit": ["ProfitLossForPeriod","ProfitLoss","ProfitLossForYear","ProfitLossOnOrdinaryActivitiesAfterTax"],
    "ebit": ["OperatingProfitLoss","OperatingProfit","ProfitLossBeforeTax"],
    "total_assets": ["TotalAssetsLessCurrentLiabilitiesPlusCurrentLiabilities","TotalAssets","FixedAssetsPlusCurrentAssets"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities","CreditorsDueWithinOneYear"],
    "net_assets": ["NetAssetsLiabilities","NetCurrentAssetsLiabilities","TotalAssetsLessCurrentLiabilities","NetAssets"],
    "cash": ["CashBankOnHand","CashBankInHand","CashCashEquivalents"],
    "retained_earnings": ["RetainedEarningsAccumulatedLosses","ProfitLossAccountReserve","RetainedEarnings"],
    "employees": ["AverageNumberEmployeesDuringPeriod","EmployeesTotal"],
}

def _match(local, field):
    for s in CONCEPT_MAP.get(field,[]): 
        if local.endswith(s): return True
    return False

def _pval(text):
    if not text: return None
    t = re.sub(r"[pounds$euro,\s()]","",text.strip()).replace("\u2212","-").replace("\u2013","-")
    if not t or t=="-": return None
    try: return float(t)
    except: return None

def parse_single_ixbrl(content):
    try:
        if isinstance(content, bytes): content = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
    except: return None
    co_num = None
    for tag in soup.find_all("xbrli:identifier"):
        v = tag.get_text(strip=True)
        if v and len(v)<=10: co_num = v.upper().zfill(8); break
    ctxs = {}
    for ctx in soup.find_all(re.compile(r"xbrli:context", re.I)):
        cid = ctx.get("id","")
        if not cid: continue
        if ctx.find(re.compile(r"xbrldi:explicitmember|xbrli:segment", re.I)): continue
        period = ctx.find(re.compile(r"xbrli:period", re.I))
        if not period: continue
        end = period.find(re.compile(r"xbrli:enddate|xbrli:instant", re.I))
        if end: ctxs[cid] = end.get_text(strip=True)
    if not ctxs: return None
    latest = sorted(set(ctxs.values()), reverse=True)[0]
    latest_ids = {k for k,v in ctxs.items() if v==latest}
    result = {"company_number": co_num, "period_end": latest, "year": latest[:4]}
    for tag in soup.find_all(re.compile(r"ix:nonfraction", re.I)):
        if tag.get("contextref","") not in latest_ids: continue
        name = tag.get("name",""); local = name.split(":")[-1] if ":" in name else name
        val = _pval(tag.get_text(strip=True))
        if val is None: continue
        scale = int(tag.get("scale","0") or "0")
        if scale: val *= (10**scale)
        if tag.get("sign","") == "-": val = -val
        for field in CONCEPT_MAP:
            if _match(local, field):
                result[field] = int(round(val)) if field != "employees" else int(val)
                break
    has = any(k in result for k in ["total_assets","net_assets","current_assets","turnover"])
    return result if has else None

def download_and_parse_accounts(max_months=4):
    cache = "parsed_accounts.json"
    if os.path.exists(cache):
        print(f"[ok] Cached accounts: {cache}")
        with open(cache) as f: return json.load(f)
    accounts = {}; parsed = 0; errs = 0
    now = datetime.now()
    for mo in range(1, max_months+1):
        target = now.replace(day=1) - timedelta(days=30*mo)
        url = f"http://download.companieshouse.gov.uk/Accounts_Monthly_Data-{target.strftime('%Y-%m')}.zip"
        print(f"\n[->] Accounts for {target.strftime('%Y-%m')}...")
        try:
            r = requests.head(url, timeout=10)
            if r.status_code != 200: print(f"  Not available ({r.status_code})"); continue
            r = requests.get(url, stream=True, timeout=600)
            total = int(r.headers.get("content-length",0))
            dl = 0; zdata = io.BytesIO()
            for chunk in r.iter_content(1024*1024):
                zdata.write(chunk); dl += len(chunk)
                if total: print(f"\r  {dl//(1024*1024)}MB / {total//(1024*1024)}MB ({dl*100//total}%)", end="", flush=True)
            print()
            zdata.seek(0)
            with zipfile.ZipFile(zdata) as zf:
                names = [n for n in zf.namelist() if n.endswith((".html",".xml",".xhtml"))]
                print(f"  Parsing {len(names)} files...")
                for i,name in enumerate(names):
                    if i>0 and i%5000==0: print(f"  {i}/{len(names)} ({parsed} ok, {errs} err)")
                    try:
                        result = parse_single_ixbrl(zf.read(name))
                        if result and result.get("company_number"):
                            cn = result["company_number"]
                            if cn not in accounts or result.get("year","") >= accounts[cn].get("year",""):
                                accounts[cn] = result
                            parsed += 1
                    except: errs += 1
            print(f"  [ok] Total: {len(accounts)} companies")
        except Exception as e:
            print(f"  Error: {e}"); continue
    print(f"\n[ok] {len(accounts)} companies with accounts ({parsed} filings, {errs} errors)")
    with open(cache, "w") as f: json.dump(accounts, f)
    return accounts

def process_data(csv_path, accounts_data):
    print(f"\n[->] Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    print(f"[ok] {len(df):,} companies")
    col_map = {c: c.strip().replace(" ","").replace(".",  "_") for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    def fc(cands):
        for c in cands:
            m = [col for col in df.columns if c.lower() in col.lower()]
            if m: return m[0]
        return None
    status_col=fc(["CompanyStatus"]); inc_col=fc(["IncorporationDate"])
    cat_col=fc(["CompanyCategory"]); acc_cat_col=fc(["AccountCategory"])
    sic_col=fc(["SicText_1","SICCode_SicText_1"])
    mort_col=fc(["NumMortCharges","Mortgages_NumMortCharges"])
    mort_out_col=fc(["NumMortOutstanding","Mortgages_NumMortOutstanding"])
    num_col=fc(["CompanyNumber"]); acc_due_col=fc(["NextDueDate","Accounts_NextDueDate"])
    acc_made_col=fc(["LastMadeUpDate","Accounts_LastMadeUpDate"])
    if not status_col or not inc_col: print("[!!] Missing columns"); return None,None
    status = df[status_col].fillna("").str.lower()
    df["failed"] = status.isin(["liquidation","receivership","administration","voluntary arrangement","insolvency proceedings"]).astype(int)
    if mort_out_col:
        dwd = (status=="dissolved") & (pd.to_numeric(df[mort_out_col],errors="coerce").fillna(0)>0)
        df.loc[dwd,"failed"] = 1
    print(f"[->] {df['failed'].sum():,} failed / {(df['failed']==0).sum():,} survived")
    # Merge accounts
    fin_fields = ["turnover","net_profit","ebit","total_assets","current_assets","current_liabilities","net_assets","cash","retained_earnings","employees"]
    if num_col and accounts_data:
        print(f"[->] Merging {len(accounts_data):,} accounts...")
        df["_cn"] = df[num_col].astype(str).str.strip().str.upper().str.zfill(8)
        rows = []
        for cn, acc in accounts_data.items():
            row = {"_cn": cn.zfill(8)}
            for f in fin_fields: row[f"acc_{f}"] = acc.get(f)
            rows.append(row)
        adf = pd.DataFrame(rows)
        df = df.merge(adf, on="_cn", how="left")
        matched = df[[c for c in df.columns if c.startswith("acc_")]].notna().any(axis=1).sum()
        print(f"[ok] Matched {matched:,}")
    else:
        for f in fin_fields: df[f"acc_{f}"] = np.nan
    now = datetime(2026,1,1)
    df["age_years"] = (now - pd.to_datetime(df[inc_col],dayfirst=True,errors="coerce")).dt.days/365.25
    if sic_col:
        df["sic_2digit"] = df[sic_col].fillna("").astype(str).str.extract(r"(\d{2})",expand=False).fillna("0").astype(int)
    else: df["sic_2digit"]=0
    if acc_cat_col:
        ar = df[acc_cat_col].fillna("").str.lower()
        df["acc_dormant"]=ar.str.contains("dormant").astype(int)
        df["acc_micro"]=ar.str.contains("micro").astype(int)
        df["acc_small"]=ar.str.contains("small").astype(int)
        df["acc_full"]=ar.str.contains("full|group|audit").astype(int)
    else:
        for c in ["acc_dormant","acc_micro","acc_small","acc_full"]: df[c]=0
    if cat_col:
        cr=df[cat_col].fillna("").str.lower()
        df["is_plc"]=cr.str.contains("public").astype(int)
        df["is_llp"]=cr.str.contains("llp|partnership").astype(int)
    else: df["is_plc"]=0; df["is_llp"]=0
    df["num_charges"]=pd.to_numeric(df.get(mort_col,pd.Series(dtype=float)),errors="coerce").fillna(0)
    df["num_outstanding"]=pd.to_numeric(df.get(mort_out_col,pd.Series(dtype=float)),errors="coerce").fillna(0)
    if acc_due_col: df["accounts_overdue"]=(pd.to_datetime(df[acc_due_col],dayfirst=True,errors="coerce")<now).astype(int)
    else: df["accounts_overdue"]=0
    if acc_made_col: df["days_since_filing"]=(now-pd.to_datetime(df[acc_made_col],dayfirst=True,errors="coerce")).dt.days.clip(0,3650).fillna(999)
    else: df["days_since_filing"]=999
    df["high_risk_sector"]=df["sic_2digit"].isin([41,42,43,56,68,47,49]).astype(int)
    for f in fin_fields: df[f"acc_{f}"]=pd.to_numeric(df[f"acc_{f}"],errors="coerce")
    df["current_ratio"]=np.where(df["acc_current_liabilities"].abs()>0,df["acc_current_assets"]/df["acc_current_liabilities"].abs(),np.nan).clip(-10,50)
    df["net_assets_negative"]=(df["acc_net_assets"]<0).astype(float).fillna(0)
    df["retained_negative"]=(df["acc_retained_earnings"]<0).astype(float).fillna(0)
    df["has_cash"]=(df["acc_cash"]>0).astype(float).fillna(0)
    df["log_net_assets"]=np.log1p(df["acc_net_assets"].clip(0).fillna(0))
    df["log_total_assets"]=np.log1p(df["acc_total_assets"].clip(0).fillna(0))
    df["log_turnover"]=np.log1p(df["acc_turnover"].clip(0).fillna(0))
    df["log_cash"]=np.log1p(df["acc_cash"].clip(0).fillna(0))
    df["profit_margin"]=np.where(df["acc_turnover"].abs()>0,df["acc_net_profit"]/df["acc_turnover"].abs(),np.nan)
    df["profit_margin"]=pd.to_numeric(df["profit_margin"],errors="coerce").clip(-5,5)
    df["has_accounts_data"]=df["acc_total_assets"].notna().astype(int)
    feat = ["age_years","sic_2digit","acc_dormant","acc_micro","acc_small","acc_full","is_plc","is_llp",
            "num_charges","num_outstanding","accounts_overdue","days_since_filing","high_risk_sector",
            "current_ratio","net_assets_negative","retained_negative","has_cash",
            "log_net_assets","log_total_assets","log_turnover","log_cash","profit_margin",
            "has_accounts_data","acc_employees"]
    df=df[df["age_years"].notna()&(df["age_years"]>0)].copy()
    print(f"\n[->] Dataset: {len(df):,} companies, {df['has_accounts_data'].sum():,} with accounts")
    print(f"[->] Failed: {df['failed'].sum():,} ({df['failed'].mean()*100:.2f}%)")
    return df[feat].fillna(0), df["failed"]

def train_model(X, y):
    print(f"\n[->] Training on {len(X):,} companies, {X.shape[1]} features...")
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f"[->] Train: {len(Xtr):,} | Test: {len(Xte):,} | Failure rate: {ytr.mean()*100:.2f}%")
    base = GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,subsample=0.8,min_samples_leaf=100,random_state=42)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    print("[->] Training (few minutes)...")
    model.fit(Xtr, ytr)
    yp = model.predict(Xte); yprob = model.predict_proba(Xte)[:,1]
    print(f"\n{'='*50}\nTEST RESULTS\n{'='*50}")
    print(classification_report(yte, yp, target_names=["Survived","Failed"]))
    auc = roc_auc_score(yte, yprob)
    print(f"AUC-ROC: {auc:.4f}\n{'='*50}")
    imps = np.zeros(X.shape[1])
    for ce in model.calibrated_classifiers_: imps += ce.estimator.feature_importances_
    imps /= len(model.calibrated_classifiers_)
    print("\nFeature Importance:")
    for n,i in sorted(zip(X.columns,imps),key=lambda x:-x[1])[:15]:
        print(f"  {n:30s} {i:.4f} {'#'*int(i*200)}")
    return model, X.columns.tolist(), auc

def export_model(model, feat, auc, n_total, n_acc, path="clearview_model.json"):
    print(f"\n[->] Exporting...")
    age_buckets = [0.5,1,2,3,5,8,12,20,50]
    base_rates = {}
    for ab,hr,at in itertools.product(age_buckets,[0,1],["dormant","micro","small","full"]):
        s = pd.DataFrame([{f:0 for f in feat}])
        s["age_years"]=ab*0.75; s["sic_2digit"]=43 if hr else 62; s["high_risk_sector"]=hr
        s[f"acc_{at}"]=1; s["days_since_filing"]=400
        s=s[feat]; base_rates[f"{ab}_{hr}_{at}"]=round(float(model.predict_proba(s)[0][1]),6)
    sector_rates = {}
    for sic in [1,10,20,25,41,42,43,45,46,47,49,55,56,62,64,66,68,69,70,71,73,74,77,78,80,82,85,86,93,96]:
        s=pd.DataFrame([{f:0 for f in feat}]); s["age_years"]=5; s["sic_2digit"]=sic; s["acc_micro"]=1
        s["high_risk_sector"]=1 if sic in [41,42,43,56,68,47,49] else 0; s["days_since_filing"]=400
        s=s[feat]; sector_rates[str(sic)]=round(float(model.predict_proba(s)[0][1]),6)
    bl=pd.DataFrame([{f:0 for f in feat}]); bl["age_years"]=5; bl["sic_2digit"]=62; bl["acc_micro"]=1; bl["days_since_filing"]=400
    bl=bl[feat]; bp=float(model.predict_proba(bl)[0][1])
    def gm(feature,value):
        m=pd.DataFrame([{f:0 for f in feat}]); m["age_years"]=5; m["sic_2digit"]=62; m["acc_micro"]=1; m["days_since_filing"]=400
        m[feature]=value; m=m[feat]; return round(float(model.predict_proba(m)[0][1])/max(bp,0.001),4)
    adj = {"accounts_overdue":gm("accounts_overdue",1),"num_outstanding_charges":gm("num_outstanding",3),
           "net_assets_negative":gm("net_assets_negative",1),"retained_negative":gm("retained_negative",1)}
    for d in [200,400,600,800,1200]: adj[f"days_since_filing_{d}"]=gm("days_since_filing",d)
    for c in [0,1,3,5,10]: adj[f"num_charges_{c}"]=gm("num_charges",c)
    for cr in [0.3,0.5,0.8,1.0,1.5,2.5]: adj[f"current_ratio_{cr}"]=gm("current_ratio",cr)
    out = {"version":"2.0","trained_on":datetime.now().isoformat(),"total_companies":n_total,
           "total_with_accounts":n_acc,"auc_roc":round(auc,4),"base_rates":base_rates,
           "adjustments":adj,"sector_rates":sector_rates,"baseline_prob":round(bp,6),
           "age_buckets":age_buckets,"features":feat}
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print(f"[ok] {path} ({os.path.getsize(path):,} bytes)")

if __name__=="__main__":
    print("="*60+"\n  Clearview ML Training v2 — Profile + Accounts\n"+"="*60)
    try: import sklearn; print(f"[ok] sklearn {sklearn.__version__}")
    except: print("[!!] pip install scikit-learn"); sys.exit(1)
    csv = download_company_csv()
    if not csv: sys.exit(1)
    print("\n"+"="*60+"\n  Downloading & parsing iXBRL accounts (20-40 min first run)\n"+"="*60)
    accounts = download_and_parse_accounts(max_months=4)
    X, y = process_data(csv, accounts)
    if X is None: sys.exit(1)
    model, feat, auc = train_model(X, y)
    export_model(model, feat, auc, len(X), int((X["has_accounts_data"]==1).sum()))
    print(f"\n{'='*60}\n  DONE! AUC: {auc:.4f}\n  Copy clearview_model.json to your project, commit & push\n{'='*60}")
