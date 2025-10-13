#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPMI   |   REM-sleep × Dopamine × Serotonin   (DEBUG version)
Reads:
  sleep_metrics/sleep_metrics.csv
  sleep_stages/*.csv   (stage column exactly SLEEP_STAGE_REM, etc.)
  LEDD, UPDRS-III, demographics, questionnaires, Olink proteomics
Outputs:
  ppmi_rem_ready.parquet  and a sanity scatter
Author: Diogo Sousa · Jun 2025
"""

import time, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
from tqdm import tqdm

# ─────────────────────────────────────  CONFIG  ──────────────────────────────
ROOT           = Path(r"C:\Users\Diogo Sousa\Downloads\Consicous\Final")

METRICS_CSV    = ROOT/"sleep_metrics"/"sleep_metrics.csv"
STAGES_DIR     = ROOT/"sleep_stages"

LEDD_CSV       = ROOT/"LEDD_Concomitant_Medication_Log_16Jun2025.csv"
UPDRS_CSV      = ROOT/"MDS-UPDRS_Part_III_Treatment_Determination_and_Part_III__Motor_Examination.csv"
DEMO_CSV       = ROOT/"Demographics_16Jun2025.csv"
STATUS_CSV     = ROOT/"Participant_Status_16Jun2025.csv"
PDSS_CSV       = ROOT/"Parkinson_s_Disease_Sleep_Scale_PDSS-2__Online__16Jun2025.csv"
RBDSQ_CSV      = ROOT/"REM_Sleep_Behavior_Disorder_Questionnaire_16Jun2025.csv"

OLINK_CSF_CSV  = ROOT/"PPMI_Project_196_CSF_NEU_NPX_20Jun2025.csv"
OLINK_PLA_CSV  = ROOT/"PPMI_Project_196_Plasma_NEURO_NPX_20Jun2025.csv"

OUT_PARQUET    = ROOT/"ppmi_rem_ready.parquet"

# ──────────────────────────────────  HELPERS  ────────────────────────────────
t0 = time.time()
def log(msg):
    global t0
    t1 = time.time()
    print(f"[{t1-t0:5.1f}s] {msg}")
    t0 = t1

def load_csv(path, **kw):
    if not path.exists():
        log(f"⚠️  {path.name} missing → skipped.")
        return pd.DataFrame()
    return pd.read_csv(path, on_bad_lines="skip", **kw)

def ensure_patno(df: pd.DataFrame, f: Path) -> pd.DataFrame:
    """Make sure df has a 'patno' column (int). Try infer from file-name if absent."""
    for col in df.columns:
        if col.lower() in {"patno","subjectid","subject_id","participant","participantid"}:
            df = df.rename(columns={col: "patno"})
            return df
    # no column → try prefix of filename
    m = re.match(r"(\d+)", f.stem)
    if m:
        df["patno"] = int(m.group(1))
        return df
    # give up
    raise KeyError("no patno in file or filename")

def concat_stages(folder: Path) -> pd.DataFrame:
    if not folder.exists():
        return pd.DataFrame()
    rows, skipped = [], []
    files = list(folder.glob("*.csv"))
    log(f"sleep_stages folder contains {len(files)} files")
    for f in tqdm(files, desc="Reading stage files"):
        try:
            df = ensure_patno(pd.read_csv(f), f)
            if df.empty or df.shape[1]==0:
                raise EmptyDataError
        except (EmptyDataError, KeyError):
            skipped.append(f.name); continue
        # add night_date
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        df["night_date"] = pd.to_datetime(m.group(1)) if m else pd.NaT
        rows.append(df)
    if skipped:
        log(f"⚠️  skipped {len(skipped)} empty/ID-less files")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def std_id(df):
    if df.empty: return df
    for c in df.columns:
        if c.lower().startswith("pat"):
            return df.rename(columns={c:"patno"})
    raise KeyError("ID col")

# ── helper: load a clinical table & normalise its date column ───────────────
def load_clinical(path: Path, date_keywords=("infodt", "visit", "date")) -> pd.DataFrame:
    """Read CSV, lower-case headers, rename first matching date-col → 'infodt'."""
    df = load_csv(path)           # already prints warning & returns empty if missing
    if df.empty:
        return df
    # 1) lower-case every header for uniform handling
    df.columns = [c.lower().strip() for c in df.columns]
    # 2) rename ID
    for c in df.columns:
        if c.startswith("pat"):
            df = df.rename(columns={c: "patno"})
            break
    # 3) find first column whose name contains any date keyword
    date_col = next((c for c in df.columns for kw in date_keywords if kw in c), None)
    if date_col:
        df = df.rename(columns={date_col: "infodt"})
        df["infodt"] = pd.to_datetime(df["infodt"], errors="coerce")
    else:
        df["infodt"] = pd.NaT
    return df

def load_demo(path: Path) -> pd.DataFrame:
    """
    Read Demographics CSV and guarantee columns:
      patno   sex   birthdt
    SEX is cast to int  (0 = Male, 1 = Female) per PPMI code-list.
    """
    df = load_csv(path)
    if df.empty:
        return df

    # lower-case every header
    df.columns = [c.lower().strip() for c in df.columns]

    # rename id
    for c in df.columns:
        if c.startswith("pat"):
            df = df.rename(columns={c: "patno"})
            break

    # normalise birth date
    if "birthdt" not in df.columns:
        # some old dumps use 'birth_date' or 'bdate'
        for c in df.columns:
            if "birth" in c and "dt" in c:
                df = df.rename(columns={c: "birthdt"})
                break
    df["birthdt"] = pd.to_datetime(df["birthdt"], errors="coerce")

    # normalise sex  (0/1 or M/F codes → int)
    if "sex" not in df.columns:
        for c in df.columns:
            if c.startswith("sex"):
                df = df.rename(columns={c: "sex"})
                break
    df["sex"] = df["sex"].replace({"M": 0, "F": 1}).astype("Int64")

    return df[["patno", "sex", "birthdt"]]

def load_questionnaire(path: Path,
                       score_keywords=("total", "score"),
                       date_keywords=("infodt", "visit", "date")) -> pd.DataFrame:
    """
    Load any PPMI questionnaire CSV (PDSS-2, RBDSQ, etc.) and guarantee:
        patno, infodt, score
    • Header case/underscores ignored.
    • If score or date column missing, it is created as <NA> so merges never fail.
    """
    df = load_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["patno", "infodt", "score"])

    # uniform headers
    df.columns = [c.lower().strip() for c in df.columns]

    # patno
    id_col = next((c for c in df.columns if c.startswith("pat")), None)
    if id_col:
        df = df.rename(columns={id_col: "patno"})
    else:
        raise KeyError(f"No PATNO in {path.name}")

    # date column
    date_col = next((c for c in df.columns
                     for kw in date_keywords if kw in c), None)
    if date_col:
        df = df.rename(columns={date_col: "infodt"})
        df["infodt"] = pd.to_datetime(df["infodt"], errors="coerce")
    else:
        df["infodt"] = pd.NaT

    # score column
    score_col = next((c for c in df.columns
                      if any(kw in c for kw in score_keywords)), None)
    if score_col:
        df = df.rename(columns={score_col: "score"})
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    else:
        df["score"] = pd.NA

    return df[["patno", "infodt", "score"]]





# ───────────────────────────── LEDD loader (robust) ─────────────────────────
def load_ledd(path: Path) -> pd.DataFrame:
    """Load any LEDD CSV, standardise to: patno, startdt, stopdt, ledd, drugname."""
    df = load_csv(path)                # warns & returns empty if missing
    if df.empty:
        return df

    # lower-case every header for uniform handling
    df.columns = [c.lower().strip() for c in df.columns]

    # rename ID col → patno
    for c in df.columns:
        if c.startswith("pat"):
            df = df.rename(columns={c: "patno"})
            break

    # normalise date columns
    date_map = {}
    for c in df.columns:
        if "start" in c and "dt" in c: date_map[c] = "startdt"
        if "stop"  in c and "dt" in c: date_map[c] = "stopdt"
    df = df.rename(columns=date_map)

    # make sure both date cols exist even if original names were odd
    for col in ("startdt", "stopdt"):
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
def load_status(path: Path) -> pd.DataFrame:
    """
    Read Participant-Status CSV and return columns:
        patno, cohort_definition, enroll_status
    – Header case / spacing doesn’t matter.
    – Missing columns are filled with <NA> so merge never crashes.
    """
    df = load_csv(path)                          # warns + empty-df if file missing
    if df.empty:
        return pd.DataFrame(columns=["patno", "cohort_definition", "enroll_status"])

    # lower-case headers for uniform handling
    df.columns = [c.lower().strip() for c in df.columns]

    # -------- patno -------------
    id_col = next((c for c in df.columns if c.startswith("pat")), None)
    if id_col:
        df = df.rename(columns={id_col: "patno"})
    else:
        raise KeyError("No PATNO column in Participant_Status file")

    # -------- cohort definition -------------
    cohort_col = next((c for c in df.columns if "cohort_def" in c or "cohort" == c), None)
    if cohort_col:
        df = df.rename(columns={cohort_col: "cohort_definition"})
    else:
        df["cohort_definition"] = pd.NA

    # -------- enrolment status -------------
    enroll_col = next((c for c in df.columns if "enroll_status" in c or
                       ("status" in c and "date" not in c)), None)
    if enroll_col:
        df = df.rename(columns={enroll_col: "enroll_status"})
    else:
        df["enroll_status"] = pd.NA

    return df[["patno", "cohort_definition", "enroll_status"]]

# ─────────────────────────  STEP 1  sleep_metrics  ───────────────────────────
metrics = load_csv(METRICS_CSV)
metrics = std_id(metrics)
metrics["night_date"] = pd.to_datetime(metrics["date"], errors="coerce")
metrics["rem_min"] = pd.to_numeric(metrics["total_rem_time_ms"], errors="coerce")/60000
metrics["key"] = metrics["patno"].astype(str)+"_"+metrics["night_date"].dt.strftime("%Y-%m-%d")
log(f"sleep_metrics rows: {len(metrics):,}, rem_min non-NA: {metrics['rem_min'].notna().sum():,}")

# ─────────────────────────  STEP 2  sleep_stages  ────────────────────────────
epochs = concat_stages(STAGES_DIR)
if epochs.empty:
    log("No usable epochs found → using metrics only.")
    rem = metrics.copy()
else:
    # clean stage labels
    epochs["stage"] = epochs["stage"].str.upper().str.strip()
    # keep only known labels
    keep_mask = epochs["stage"].isin([
        "SLEEP_STAGE_REM","SLEEP_STAGE_NREM_DEEP","SLEEP_STAGE_NREM_LIGHT",
        "SLEEP_STAGE_WAKE","SLEEP_STAGE_UNKNOWN"
    ])
    epochs = epochs[keep_mask]
    # DEBUG distribution
    print(epochs["stage"].value_counts())

    epochs["is_rem"] = epochs["stage"].eq("SLEEP_STAGE_REM")
    epochs["dur_min"] = epochs["duration_sec"]/60
    epochs["epoch_ts"] = pd.to_datetime(epochs["time_utc_ms"], unit="ms")
    # ----- derive nightly REM metrics from epochs -----
    grouped = epochs.groupby(["patno", "night_date"], sort=False, group_keys=False)


    def summarise(g):
        rem_rows = g[g["is_rem"]]
        if rem_rows.empty:
            return pd.Series({"rem_min": 0,
                              "rem_periods": 0,
                              "rem_latency_min": np.nan})
        rem_min = rem_rows["dur_min"].sum()
        # count REM bouts separated by >1 min
        rem_rows = rem_rows.sort_values("epoch_ts")
        bouts = (rem_rows["epoch_ts"].diff().dt.total_seconds().fillna(0) > 60).sum() + 1
        latency = (rem_rows["epoch_ts"].iloc[0] - g["epoch_ts"].min()).total_seconds() / 60
        return pd.Series({"rem_min": rem_min,
                          "rem_periods": bouts,
                          "rem_latency_min": latency})


    # include_groups=False keeps patno & night_date in result → no duplicate columns
    nightly = grouped.apply(summarise, include_groups=False).reset_index()
    # ──────────────────  BUILD FINAL REM TABLE  ────────────────────────────────
    # 1) guarantee 'nightly' exists
    if 'nightly' not in locals():
        nightly = pd.DataFrame(columns=["patno", "night_date", "rem_min",
                                        "rem_periods", "rem_latency_min"])

    # 2) add unique key to both frames
    metrics["key"] = metrics["patno"].astype(str) + "_" + metrics["night_date"].dt.strftime("%Y-%m-%d")
    nightly["key"] = nightly["patno"].astype(str) + "_" + nightly["night_date"].dt.strftime("%Y-%m-%d")

    # 3) use metrics as the base; fill in new nights from nightly
    nightly_new = nightly.loc[~nightly["key"].isin(metrics["key"])]

    rem = pd.concat([metrics, nightly_new], ignore_index=True)
    log(f"Total nights in REM table: {len(rem):,}")

# ─────────────────────────  STEP 3  clinical merge  ──────────────────────────
ledd = load_ledd(LEDD_CSV)
log(f"Loaded LEDD rows: {len(ledd):,}")
# ── LOAD *ALL* MDS-UPDRS PART-III CSVs ───────────────────────────────────────
UPDRS_FILES = sorted(ROOT.glob("MDS-UPDRS_Part_III*_*.csv"))

if not UPDRS_FILES:
    log("⚠️  No Part-III files found; UPDRS_III will stay NaN.")
    updrs = pd.DataFrame(columns=["patno", "infodt", "np3tot"])

else:
    frames = []
    for f in UPDRS_FILES:
        df = load_csv(f)                     # read raw
        if df.empty:
            continue

        # 1) force every header to lower-case → uniform
        df.columns = [c.lower().strip() for c in df.columns]

        # 2) rename ID col to patno if necessary
        for c in df.columns:
            if c.startswith("pat"):
                df = df.rename(columns={c: "patno"})
                break

        # 3) some releases use 'visitdate' instead of 'infodt'
        if "infodt" not in df.columns and "visitdate" in df.columns:
            df = df.rename(columns={"visitdate": "infodt"})

        # 4) keep only the columns we need
        needed = [c for c in ("patno", "infodt", "np3tot") if c in df.columns]
        frames.append(df[needed])

    updrs = pd.concat(frames, ignore_index=True)

    # 5) make sure date column is datetime64
    updrs["infodt"] = pd.to_datetime(updrs["infodt"], errors="coerce")

    log(f"Loaded UPDRS Part-III rows: {len(updrs):,}")

demo   = load_demo(DEMO_CSV)
log(f"Loaded Demographics rows: {len(demo):,}")

status = load_status(STATUS_CSV)
log(f"Loaded Status rows: {len(status):,}")

pdss2  = load_questionnaire(PDSS_CSV, score_keywords=("pdss2", "total"))
rbdsq  = load_questionnaire(RBDSQ_CSV, score_keywords=("rbdsq", "total"))
log(f"Loaded PDSS-2 rows: {len(pdss2):,}")
log(f"Loaded RBDSQ rows: {len(rbdsq):,}")


log("Clinical CSVs loaded")

# LEDD + SSRI
ssri_re = re.compile("|".join(["citalopram","escitalopram","fluoxetine","paroxetine","sertraline",
                               "venlafaxine","duloxetine","fluvoxamine","desvenlafaxine","mirtazapine","trazodone"]), re.I)
ledd.sort_values(["patno","startdt"], inplace=True)

def ledd_flag(row):
    sub = ledd[(ledd.patno==row.patno)&
               (ledd.startdt<=row.night_date)&
               ((ledd.stopdt.isna())|(ledd.stopdt>=row.night_date))]
    if sub.empty:
        return pd.Series({"LEDD":0,"SSRI_flag":0})
    return pd.Series({
        "LEDD": pd.to_numeric(sub["ledd"], errors='coerce').sum(),
        "SSRI_flag": int(sub["drugname"].fillna("").str.contains(ssri_re).any()) if "drugname" in sub.columns else 0
    })

log("Calculating LEDD/SSRI flags ...")
rem[["LEDD","SSRI_flag"]] = rem.apply(ledd_flag, axis=1)

# demographics
rem = (rem.merge(demo[["patno","sex","birthdt"]], on="patno", how="left")
          .merge(status[["patno","cohort_definition","enroll_status"]], on="patno", how="left"))
rem["age"] = (rem["night_date"] - pd.to_datetime(rem["birthdt"]))\
             .dt.days/365.25

# latest UPDRS-III
if not updrs.empty:
    updrs_last = (updrs.sort_values(["patno","infodt"])
                       .groupby("patno").tail(1)[["patno","np3tot"]])
    rem = rem.merge(updrs_last.rename(columns={"np3tot":"UPDRS_III"}), on="patno", how="left")

# PDSS & RBDSQ via asof merge
def attach_scale(df, col, new):
    if df.empty: return
    df = df.dropna(subset=["infodt", col]).sort_values(["patno","infodt"])
    rem.sort_values("night_date", inplace=True)
    merged = pd.merge_asof(rem, df[["patno","infodt",col]],
                           left_on="night_date", right_on="infodt",
                           by="patno", direction="backward")
    rem[new] = merged[col]

attach_scale(pdss2, "score", "PDSS2_total")   # <— uses the unified “score”
attach_scale(rbdsq, "score", "RBDSQ_total")


log("Merged clinical & questionnaires")

# ─────────────────────────  STEP 4  proteomics  ─────────────────────────────
def latest_olink(path: Path, tag: str) -> pd.DataFrame:
    """
    Load any Olink Explore CSV and return one row per patno with
    serotonin-related NPX values.  Works even if headers differ:
        id col  → patno
        assay   → assay
        value   → npx
        date    → infodt
    """
    df = load_csv(path)
    if df.empty:
        return pd.DataFrame()

    # lower-case headers
    df.columns = [c.lower().strip() for c in df.columns]

    # ---- id column ----
    id_col = next((c for c in df.columns if c.startswith("pat")), None)
    if not id_col:
        raise KeyError(f"No PATNO column in {path.name}")
    df = df.rename(columns={id_col: "patno"})

    # ---- assay column ----
    assay_col = next((c for c in df.columns if "assay" in c or "protein" in c or "analyte" in c), None)
    if not assay_col:
        raise KeyError(f"No assay / protein column in {path.name}")
    df = df.rename(columns={assay_col: "assay"})

    # ---- NPX / value column ----
    value_col = next((c for c in df.columns if c in {"npx","value","normvalue"}), None)
    if not value_col:
        raise KeyError(f"No NPX/value column in {path.name}")
    df = df.rename(columns={value_col: "npx"})

    # ---- date column (optional) ----
    date_col = next((c for c in df.columns if "infodt" in c or ("date" in c and "sampl" not in c)), None)
    if date_col:
        df = df.rename(columns={date_col: "infodt"})
        df["infodt"] = pd.to_datetime(df["infodt"], errors="coerce")
    else:
        df["infodt"] = pd.NaT

    # ---- keep serotonin-related assays ----
    df = df[df["assay"].str.contains("HTR2A|TPH|MAO", case=False, na=False)]
    if df.empty:
        return pd.DataFrame()

    # ---- pivot to wide ----
    last = (df.sort_values(["patno","assay","infodt"])
              .groupby(["patno","assay"]).tail(1)
              .pivot(index="patno", columns="assay", values="npx"))

    last.columns = [c.replace(" ","_")+f"_{tag}" for c in last.columns]
    return last.reset_index()


for fpath, tag in [(OLINK_CSF_CSV,"csf"),(OLINK_PLA_CSV,"plasma")]:
    wide = latest_olink(fpath, tag)
    if not wide.empty:
        rem = rem.merge(wide, on="patno", how="left")
        log(f"Merged {len(wide.columns)-1} Olink ({tag}) columns")

# ─────────────────────────  STEP 5  save & plot  ────────────────────────────

# 1) keep only one copy of any duplicated column name
rem = rem.loc[:, ~rem.columns.duplicated(keep="first")]

# 2) optionally: drop epoch-level columns that make no sense at night-level
cols_to_drop = {"duration_sec","confidence","imu_ppg_availability",
                "is_rem","dur_min","epoch_ts","time_utc_ms","date"}
rem = rem.drop(columns=cols_to_drop & set(rem.columns), errors="ignore")

rem.to_parquet(OUT_PARQUET, index=False)
log(f"Saved final parquet: {OUT_PARQUET.name}  →  {len(rem):,} rows")

if "LEDD" in rem.columns and "rem_min" in rem.columns:
    plt.scatter(rem["LEDD"], rem["rem_min"], s=8, alpha=0.3)
    plt.xlabel("LEDD (mg levodopa eq.)")
    plt.ylabel("REM minutes")
    plt.title("LEDD vs nightly REM duration")
    plt.tight_layout(); plt.show()
else:
    log("Plot skipped (columns missing)")
